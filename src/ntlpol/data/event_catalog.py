from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ntlpol.io_utils import ensure_parent, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.event_catalog")

EVENT_COLUMNS = [
    "event_id",
    "event_name",
    "event_type",
    "country",
    "state_names",
    "start_date",
    "end_date",
    "event_year",
    "event_month",
    "source_catalog",
    "ibtracs_sid",
    "primary_hazard",
    "notes",
]

EVENT_TEMPLATE_COLUMNS = EVENT_COLUMNS + [
    "source_event_id",
    "raw_event_type",
    "raw_source_file",
]

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "event_id": ("event_id", "event id", "dis no", "disno", "dis_no", "id"),
    "event_name": (
        "event_name",
        "event name",
        "name",
        "disaster name",
        "storm name",
        "cyclone name",
        "event",
    ),
    "event_type": (
        "event_type",
        "event type",
        "disaster type",
        "disaster subgroup",
        "hazard type",
        "type",
    ),
    "country": ("country", "country name", "location country"),
    "state_names": (
        "state_names",
        "state name",
        "state",
        "states",
        "location",
        "admin1",
        "province",
        "region",
    ),
    "start_date": ("start_date", "start date", "date_start", "began", "start"),
    "end_date": ("end_date", "end date", "date_end", "ended", "end"),
    "event_year": ("event_year", "year", "start year", "start_year"),
    "event_month": ("event_month", "month", "start month", "start_month"),
    "source_catalog": ("source_catalog", "source", "catalog"),
    "ibtracs_sid": ("ibtracs_sid", "sid", "storm_id", "ibtracs id", "serial_num"),
    "primary_hazard": ("primary_hazard", "primary hazard", "hazard", "main_hazard"),
    "notes": ("notes", "note", "comments", "comment"),
}

EVENT_TYPE_MAP = {
    "tropical_cyclone": "tropical_cyclone",
    "cyclone": "tropical_cyclone",
    "tropical cyclone": "tropical_cyclone",
    "storm": "tropical_cyclone",
    "tropical storm": "tropical_cyclone",
    "typhoon": "tropical_cyclone",
    "hurricane": "tropical_cyclone",
    "urban_flooding": "urban_flooding",
    "urban flooding": "urban_flooding",
    "flood": "urban_flooding",
    "flash flood": "urban_flooding",
    "riverine flood": "urban_flooding",
    "coastal flood": "urban_flooding",
}


def _norm_col(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _slug(value: object, max_len: int = 32) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "event").strip().lower()).strip("_")
    text = re.sub(r"_+", "_", text)
    return (text[:max_len].strip("_") or "event")


def _column_lookup(columns: Iterable[str]) -> dict[str, str]:
    normalized = {_norm_col(c): c for c in columns}
    result: dict[str, str] = {}
    for target, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_norm = _norm_col(alias)
            if alias_norm in normalized:
                result[target] = normalized[alias_norm]
                break
    return result


def _clean_event_type(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = _norm_col(value)
    if text in EVENT_TYPE_MAP:
        return EVENT_TYPE_MAP[text]
    for key, mapped in EVENT_TYPE_MAP.items():
        if key in text:
            return mapped
    return None


def _parse_date_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col and col in df.columns:
        return pd.to_datetime(df[col], errors="coerce")
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def _date_from_ymd(
    year: pd.Series,
    month: pd.Series | None = None,
    day: pd.Series | None = None,
) -> pd.Series:
    y = pd.to_numeric(year, errors="coerce")
    m = pd.to_numeric(month, errors="coerce") if month is not None else pd.Series(1, index=year.index)
    d = pd.to_numeric(day, errors="coerce") if day is not None else pd.Series(1, index=year.index)
    out = pd.to_datetime(
        {
            "year": y.fillna(1900).astype(int),
            "month": m.fillna(1).clip(1, 12).astype(int),
            "day": d.fillna(1).clip(1, 28).astype(int),
        },
        errors="coerce",
    )
    out[y.isna()] = pd.NaT
    return out


def _infer_dates(raw: pd.DataFrame, lookup: dict[str, str]) -> tuple[pd.Series, pd.Series]:
    start = _parse_date_series(raw, lookup.get("start_date"))
    end = _parse_date_series(raw, lookup.get("end_date"))

    if start.isna().all() and lookup.get("event_year"):
        year = raw[lookup["event_year"]]
        month = raw[lookup["event_month"]] if lookup.get("event_month") else None
        # EM-DAT often has explicit start day with many column variations.
        day_col = None
        for c in raw.columns:
            if _norm_col(c) in {"start day", "start_day", "day"}:
                day_col = c
                break
        start = _date_from_ymd(year, month, raw[day_col] if day_col else None)

    if end.isna().all():
        end = start.copy()

    return start, end


def _series_or_default(raw: pd.DataFrame, lookup: dict[str, str], key: str, default: object) -> pd.Series:
    col = lookup.get(key)
    if col and col in raw.columns:
        return raw[col]
    return pd.Series(default, index=raw.index)


def normalize_event_frame(
    raw: pd.DataFrame,
    *,
    source_catalog: str,
    raw_source_file: str | None = None,
    country_filter: str = "India",
    include_types: Iterable[str] = ("tropical_cyclone", "urban_flooding"),
) -> pd.DataFrame:
    """Normalize one event catalog dataframe to the project schema.

    The function is intentionally permissive because EM-DAT exports, official
    reports, and manually curated tables tend to use different column names.
    """
    if raw.empty:
        return pd.DataFrame(columns=EVENT_TEMPLATE_COLUMNS)

    lookup = _column_lookup(raw.columns)
    out = pd.DataFrame(index=raw.index)

    start, end = _infer_dates(raw, lookup)
    out["start_date"] = start.dt.date.astype("string")
    out["end_date"] = end.dt.date.astype("string")
    out["event_year"] = start.dt.year.astype("Int64")
    out["event_month"] = start.dt.month.astype("Int64")

    out["event_name"] = _series_or_default(raw, lookup, "event_name", "unnamed_event").fillna(
        "unnamed_event"
    )
    raw_event_type = _series_or_default(raw, lookup, "event_type", None)
    out["raw_event_type"] = raw_event_type
    out["event_type"] = raw_event_type.map(_clean_event_type)

    out["country"] = _series_or_default(raw, lookup, "country", country_filter).fillna(country_filter)
    out["state_names"] = _series_or_default(raw, lookup, "state_names", "").fillna("")
    out["source_catalog"] = _series_or_default(raw, lookup, "source_catalog", source_catalog).fillna(
        source_catalog
    )
    out["ibtracs_sid"] = _series_or_default(raw, lookup, "ibtracs_sid", pd.NA)
    out["primary_hazard"] = _series_or_default(raw, lookup, "primary_hazard", out["event_type"])
    out["notes"] = _series_or_default(raw, lookup, "notes", "").fillna("")
    out["source_event_id"] = _series_or_default(raw, lookup, "event_id", pd.NA)
    out["raw_source_file"] = raw_source_file or ""

    # Country and type filters.
    country_mask = out["country"].astype(str).str.contains(country_filter, case=False, na=False)
    include_types = set(include_types)
    type_mask = out["event_type"].isin(include_types)
    out = out[country_mask & type_mask].copy()

    # Drop rows without usable date.
    out = out[out["event_year"].notna() & out["event_month"].notna()].copy()

    # Create stable project event IDs.
    type_code = out["event_type"].map({"tropical_cyclone": "TC", "urban_flooding": "UF"}).fillna("EV")
    month = out["event_month"].astype(int).astype(str).str.zfill(2)
    year = out["event_year"].astype(int).astype(str)
    name_slug = out["event_name"].map(_slug)
    event_id_from_source = out["source_event_id"].astype("string").fillna("").map(_slug)
    out["event_id"] = (
        "IND_" + type_code.astype(str) + "_" + year + month + "_" + name_slug
    )
    has_source_id = event_id_from_source.ne("") & event_id_from_source.ne("nan")
    out.loc[has_source_id, "event_id"] = (
        "IND_" + type_code[has_source_id].astype(str) + "_" + year[has_source_id] + "_" + event_id_from_source[has_source_id]
    )

    # Ensure unique IDs even if two sources produce the same record.
    out["event_id"] = _deduplicate_ids(out["event_id"])

    return out[EVENT_TEMPLATE_COLUMNS].reset_index(drop=True)


def _deduplicate_ids(ids: pd.Series) -> pd.Series:
    counts: dict[str, int] = {}
    result: list[str] = []
    for value in ids.astype(str):
        n = counts.get(value, 0)
        result.append(value if n == 0 else f"{value}_{n + 1}")
        counts[value] = n + 1
    return pd.Series(result, index=ids.index, dtype="string")


def read_raw_event_files(raw_events_dir: Path) -> list[tuple[pd.DataFrame, str, Path]]:
    candidates = [
        ("manual_event_catalog.csv", "manual"),
        ("emdat_india_events.csv", "EM_DAT"),
        ("india_official_events.csv", "India_official"),
    ]
    frames: list[tuple[pd.DataFrame, str, Path]] = []
    for filename, source in candidates:
        path = raw_events_dir / filename
        if path.exists():
            frames.append((pd.read_csv(path), source, path))
    return frames


def build_event_catalog(
    *,
    raw_events_dir: str | Path,
    output_path: str | Path,
    template_path: str | Path | None = None,
    country_filter: str = "India",
    include_types: Iterable[str] = ("tropical_cyclone", "urban_flooding"),
) -> pd.DataFrame:
    raw_events_dir = Path(raw_events_dir)
    raw_events_dir.mkdir(parents=True, exist_ok=True)

    raw_frames = read_raw_event_files(raw_events_dir)
    if not raw_frames:
        template = pd.DataFrame(columns=EVENT_TEMPLATE_COLUMNS)
        if template_path is not None:
            ensure_parent(template_path)
            template.to_csv(template_path, index=False)
            LOGGER.warning("No raw event catalog found. Wrote template: %s", template_path)
        clean = pd.DataFrame(columns=EVENT_COLUMNS)
        write_table(clean, output_path, index=False)
        LOGGER.warning("Wrote empty event catalog. Add raw CSVs and rerun.")
        return clean

    normalized = []
    for raw, source, path in raw_frames:
        LOGGER.info("Reading raw event file: %s", path)
        normalized.append(
            normalize_event_frame(
                raw,
                source_catalog=source,
                raw_source_file=path.name,
                country_filter=country_filter,
                include_types=include_types,
            )
        )

    all_events = pd.concat(normalized, ignore_index=True) if normalized else pd.DataFrame()
    if all_events.empty:
        clean = pd.DataFrame(columns=EVENT_COLUMNS)
    else:
        clean = all_events.copy()
        clean["start_date_dt"] = pd.to_datetime(clean["start_date"], errors="coerce")
        clean = clean.sort_values(["start_date_dt", "event_type", "event_name"]).drop(
            columns=["start_date_dt"]
        )
        # Drop exact duplicates while preserving curated/manual rows first when present.
        clean["_priority"] = np.where(clean["source_catalog"].astype(str).str.lower().eq("manual"), 0, 1)
        clean = clean.sort_values("_priority")
        clean = clean.drop_duplicates(
            subset=["event_name", "event_type", "start_date", "end_date"], keep="first"
        )
        clean = clean.drop(columns=["_priority"])
        clean["event_id"] = _deduplicate_ids(clean["event_id"])
        clean = clean[EVENT_COLUMNS].reset_index(drop=True)

    validate_event_catalog(clean)
    write_table(clean, output_path, index=False)
    LOGGER.info("Wrote clean event catalog with %s events: %s", len(clean), output_path)
    return clean


def validate_event_catalog(df: pd.DataFrame) -> None:
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Event catalog is missing required columns: {missing}")
    if df.empty:
        return
    if df["event_id"].isna().any():
        raise ValueError("event_id contains missing values")
    if df["event_id"].duplicated().any():
        duplicated = df.loc[df["event_id"].duplicated(), "event_id"].tolist()
        raise ValueError(f"event_id must be unique. Duplicated: {duplicated[:5]}")
    bad_types = set(df["event_type"].dropna()) - {"tropical_cyclone", "urban_flooding"}
    if bad_types:
        raise ValueError(f"Unexpected event_type values: {sorted(bad_types)}")
    start = pd.to_datetime(df["start_date"], errors="coerce")
    end = pd.to_datetime(df["end_date"], errors="coerce")
    if start.isna().any():
        raise ValueError("start_date contains invalid dates")
    if (end < start).any():
        raise ValueError("end_date cannot be earlier than start_date")


@dataclass(frozen=True)
class EventCatalogPaths:
    raw_events_dir: Path
    clean_output_path: Path
    template_path: Path
