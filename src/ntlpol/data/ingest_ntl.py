from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from ntlpol.io_utils import ensure_parent, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.ingest_ntl")

DAILY_NTL_COLUMNS = [
    "grid_id",
    "date",
    "year_month",
    "raw_radiance",
    "cleaned_radiance",
    "valid_obs_count",
    "cloud_or_quality_bad_count",
    "coverage_ratio",
    "source",
]

MONTHLY_NTL_COLUMNS = [
    "grid_id",
    "year_month",
    "raw_radiance",
    "cleaned_radiance",
    "valid_obs_count",
    "cloud_or_quality_bad_count",
    "coverage_ratio",
    "source",
]

NTL_TEMPLATE_COLUMNS = DAILY_NTL_COLUMNS

ALIASES: dict[str, tuple[str, ...]] = {
    "grid_id": ("grid_id", "grid id", "cell_id", "pixel_id", "id"),
    "date": ("date", "datetime", "time", "observation_date", "system:time_start"),
    "year_month": ("year_month", "year month", "month", "ym", "calendar_year_month"),
    "raw_radiance": (
        "raw_radiance",
        "raw radiance",
        "avg_rad",
        "radiance",
        "ntl",
        "vnp46a2_radiance",
        "DNB_BRDF_Corrected_NTL",
        "Gap_Filled_DNB_BRDF_Corrected_NTL",
    ),
    "cleaned_radiance": (
        "cleaned_radiance",
        "cleaned radiance",
        "clean_rad",
        "radiance_clean",
        "masked_radiance",
    ),
    "valid_obs_count": ("valid_obs_count", "valid observation count", "valid_count", "cf_cvg"),
    "cloud_or_quality_bad_count": (
        "cloud_or_quality_bad_count",
        "bad_count",
        "invalid_obs_count",
    ),
    "coverage_ratio": ("coverage_ratio", "coverage", "valid_ratio", "valid_fraction"),
    "source": ("source", "dataset"),
}


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9:_]+", " ", str(value).strip().lower()).strip()


def _lookup(columns: Iterable[str]) -> dict[str, str]:
    normalized = {_norm(c): c for c in columns}
    result: dict[str, str] = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            if _norm(alias) in normalized:
                result[target] = normalized[_norm(alias)]
                break
    return result


def _to_datetime_series(raw: pd.DataFrame, lookup: dict[str, str]) -> pd.Series:
    if lookup.get("date"):
        parsed = pd.to_datetime(raw[lookup["date"]], errors="coerce")
        # Earth Engine system:time_start can arrive as milliseconds.
        if parsed.isna().mean() > 0.8:
            numeric = pd.to_numeric(raw[lookup["date"]], errors="coerce")
            parsed = pd.to_datetime(numeric, unit="ms", errors="coerce")
        return parsed
    if lookup.get("year_month"):
        return pd.to_datetime(raw[lookup["year_month"]].astype(str).str.slice(0, 7), errors="coerce")
    raise ValueError("NTL data must contain either date or year_month column")


def normalize_ntl_frame(raw: pd.DataFrame, *, source: str = "VNP46A2") -> pd.DataFrame:
    """Normalize raw daily or monthly NTL CSVs into daily-standard rows."""
    if raw.empty:
        return pd.DataFrame(columns=DAILY_NTL_COLUMNS)
    lookup = _lookup(raw.columns)
    if "grid_id" not in lookup:
        raise ValueError("NTL CSV must contain a grid_id/cell_id/pixel_id column")
    if "raw_radiance" not in lookup and "cleaned_radiance" not in lookup:
        raise ValueError("NTL CSV must contain raw_radiance, cleaned_radiance, avg_rad, or radiance")

    out = pd.DataFrame()
    out["grid_id"] = raw[lookup["grid_id"]].astype(str)
    dt = _to_datetime_series(raw, lookup)
    out["date"] = dt.dt.strftime("%Y-%m-%d")
    out["year_month"] = dt.dt.to_period("M").astype(str)

    raw_col = lookup.get("raw_radiance") or lookup.get("cleaned_radiance")
    clean_col = lookup.get("cleaned_radiance") or raw_col
    out["raw_radiance"] = pd.to_numeric(raw[raw_col], errors="coerce")
    out["cleaned_radiance"] = pd.to_numeric(raw[clean_col], errors="coerce")

    if lookup.get("valid_obs_count"):
        out["valid_obs_count"] = pd.to_numeric(raw[lookup["valid_obs_count"]], errors="coerce")
    else:
        out["valid_obs_count"] = np.where(out["cleaned_radiance"].notna(), 1, 0)

    if lookup.get("cloud_or_quality_bad_count"):
        out["cloud_or_quality_bad_count"] = pd.to_numeric(
            raw[lookup["cloud_or_quality_bad_count"]], errors="coerce"
        )
    else:
        out["cloud_or_quality_bad_count"] = np.where(out["cleaned_radiance"].isna(), 1, 0)

    if lookup.get("coverage_ratio"):
        out["coverage_ratio"] = pd.to_numeric(raw[lookup["coverage_ratio"]], errors="coerce")
    else:
        denom = out["valid_obs_count"].fillna(0) + out["cloud_or_quality_bad_count"].fillna(0)
        out["coverage_ratio"] = np.where(denom > 0, out["valid_obs_count"] / denom, np.nan)

    out["source"] = raw[lookup["source"]].astype(str) if lookup.get("source") else source
    out = out.dropna(subset=["grid_id", "date", "year_month"]).copy()
    out = out[DAILY_NTL_COLUMNS]
    return out


def aggregate_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Derive project monthly NTL from daily-standard rows.

    The raw source remains daily VNP46A2. Monthly aggregation is a derivative
    product used for the 12-month pre / 24-month post sequence design.
    """
    if df.empty:
        return pd.DataFrame(columns=MONTHLY_NTL_COLUMNS)
    grouped = df.groupby(["grid_id", "year_month"], as_index=False).agg(
        raw_radiance=("raw_radiance", "median"),
        cleaned_radiance=("cleaned_radiance", "median"),
        valid_obs_count=("valid_obs_count", "sum"),
        cloud_or_quality_bad_count=("cloud_or_quality_bad_count", "sum"),
        coverage_ratio=("coverage_ratio", "mean"),
        source=("source", lambda x: ";".join(sorted(set(map(str, x))))),
    )
    return grouped[MONTHLY_NTL_COLUMNS]


def read_raw_ntl_csvs(raw_ntl_dir: str | Path) -> list[Path]:
    raw_ntl_dir = Path(raw_ntl_dir)
    if not raw_ntl_dir.exists():
        return []
    preferred = sorted(raw_ntl_dir.glob("grid_daily_ntl*.csv"))
    if preferred:
        return [p for p in preferred if "template" not in p.name.lower()]
    preferred = sorted(raw_ntl_dir.glob("grid_monthly_ntl*.csv"))
    if preferred:
        return [p for p in preferred if "template" not in p.name.lower()]
    return [p for p in sorted(raw_ntl_dir.glob("*.csv")) if "template" not in p.name.lower()]


def _read_normalized_file(path: Path, *, source: str, chunksize: int) -> pd.DataFrame:
    """Read one raw CSV in chunks, normalize it, and return daily-standard rows.

    This intentionally processes one monthly partition at a time. It avoids the
    previous all-file concat pattern, which can require several GB for all-India
    daily VNP46A2 exports.
    """
    parts: list[pd.DataFrame] = []
    total_rows = 0
    try:
        reader = pd.read_csv(path, chunksize=chunksize, low_memory=False)
        for i, chunk in enumerate(reader, start=1):
            total_rows += len(chunk)
            normalized = normalize_ntl_frame(chunk, source=source)
            if not normalized.empty:
                parts.append(normalized)
            if i == 1 or i % 5 == 0:
                LOGGER.info(
                    "  %s: read %s rows so far, normalized chunks kept=%s",
                    path.name,
                    f"{total_rows:,}",
                    len(parts),
                )
    except EmptyDataError:
        LOGGER.warning("Skipping empty CSV: %s", path)
        return pd.DataFrame(columns=DAILY_NTL_COLUMNS)

    if not parts:
        return pd.DataFrame(columns=DAILY_NTL_COLUMNS)

    daily = pd.concat(parts, ignore_index=True)
    before = len(daily)
    daily = daily.drop_duplicates(subset=["grid_id", "date"], keep="last")
    if before != len(daily):
        LOGGER.info("  %s: dropped %s duplicate daily rows", path.name, f"{before - len(daily):,}")
    return daily


def ingest_ntl(
    *,
    raw_ntl_dir: str | Path,
    output_path: str | Path,
    daily_output_path: str | Path | None = None,
    template_path: str | Path | None = None,
    source: str = "VNP46A2",
    chunksize: int = 500_000,
    write_daily_table: bool = False,
) -> pd.DataFrame:
    """Ingest raw VNP46A2 CSVs using a memory-safe per-file pipeline.

    For all-India 5 km grids, a daily table can easily exceed 100M rows. The
    modeling pipeline only needs the derived monthly sequence, so the default is
    to aggregate each raw daily CSV to monthly rows immediately and avoid writing
    the massive daily table. Set write_daily_table=True only for small test runs.
    """
    raw_ntl_dir = Path(raw_ntl_dir)
    raw_ntl_dir.mkdir(parents=True, exist_ok=True)
    files = read_raw_ntl_csvs(raw_ntl_dir)
    if not files:
        template = pd.DataFrame(columns=NTL_TEMPLATE_COLUMNS)
        if template_path is not None:
            ensure_parent(template_path)
            template.to_csv(template_path, index=False)
            LOGGER.warning("No raw NTL CSV found. Wrote daily-standard template: %s", template_path)
        daily = pd.DataFrame(columns=DAILY_NTL_COLUMNS)
        monthly = pd.DataFrame(columns=MONTHLY_NTL_COLUMNS)
        if daily_output_path is not None and write_daily_table:
            write_table(daily, daily_output_path, index=False)
        write_table(monthly, output_path, index=False)
        return monthly

    LOGGER.info("Found %s raw NTL CSV file(s). Processing one file at a time.", len(files))
    monthly_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []

    for file_idx, path in enumerate(files, start=1):
        LOGGER.info("[%s/%s] Reading NTL CSV: %s", file_idx, len(files), path)
        daily_file = _read_normalized_file(path, source=source, chunksize=chunksize)
        if daily_file.empty:
            LOGGER.warning("  %s produced no normalized rows. Skipping.", path.name)
            continue
        monthly_file = aggregate_to_monthly(daily_file)
        LOGGER.info(
            "  %s: daily rows=%s -> monthly rows=%s",
            path.name,
            f"{len(daily_file):,}",
            f"{len(monthly_file):,}",
        )
        monthly_parts.append(monthly_file)
        if write_daily_table:
            daily_parts.append(daily_file)
        # Release the large per-file table before reading the next CSV.
        del daily_file

    if monthly_parts:
        monthly = pd.concat(monthly_parts, ignore_index=True)
        before = len(monthly)
        monthly = monthly.drop_duplicates(subset=["grid_id", "year_month"], keep="last")
        if before != len(monthly):
            LOGGER.warning(
                "Dropped %s duplicate monthly rows. This usually means duplicate Drive exports were present.",
                f"{before - len(monthly):,}",
            )
    else:
        monthly = pd.DataFrame(columns=MONTHLY_NTL_COLUMNS)

    if write_daily_table and daily_output_path is not None:
        daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame(columns=DAILY_NTL_COLUMNS)
        daily = daily.drop_duplicates(subset=["grid_id", "date"], keep="last")
        write_table(daily, daily_output_path, index=False)
        LOGGER.info("Wrote daily NTL table with %s rows: %s", f"{len(daily):,}", daily_output_path)
    elif daily_output_path is not None:
        LOGGER.info(
            "Skipped writing full daily NTL table to avoid huge memory/disk use. "
            "Monthly modeling table will be written instead."
        )

    write_table(monthly, output_path, index=False)
    LOGGER.info("Wrote derived monthly NTL table with %s rows: %s", f"{len(monthly):,}", output_path)
    return monthly
