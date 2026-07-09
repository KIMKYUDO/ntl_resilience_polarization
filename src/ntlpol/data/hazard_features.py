from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ntlpol.io_utils import ensure_parent, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.hazard_features")

HAZARD_FEATURE_COLUMNS = [
    "event_id",
    "grid_id",
    "cyclone_distance_to_track_km",
    "cyclone_max_wind_near_grid",
    "rainfall_accum_event_mm",
    "rainfall_accum_pre_event_3d_mm",
    "rainfall_accum_post_event_7d_mm",
    "flood_exposure_ratio",
    "flood_duration_days",
    "flooded_pixel_ratio",
]

ALIASES = {
    "event_id": ("event_id", "event id", "disaster_id"),
    "grid_id": ("grid_id", "grid id", "cell_id", "pixel_id"),
    "cyclone_distance_to_track_km": (
        "cyclone_distance_to_track_km",
        "distance_to_track_km",
        "track_distance_km",
    ),
    "cyclone_max_wind_near_grid": (
        "cyclone_max_wind_near_grid",
        "max_wind_near_grid",
        "wind_speed",
        "max_wind",
    ),
    "rainfall_accum_event_mm": (
        "rainfall_accum_event_mm",
        "event_rainfall_mm",
        "rainfall_mm",
        "gpm_event_mm",
    ),
    "rainfall_accum_pre_event_3d_mm": (
        "rainfall_accum_pre_event_3d_mm",
        "pre_event_3d_rainfall_mm",
    ),
    "rainfall_accum_post_event_7d_mm": (
        "rainfall_accum_post_event_7d_mm",
        "post_event_7d_rainfall_mm",
    ),
    "flood_exposure_ratio": (
        "flood_exposure_ratio",
        "flood_ratio",
        "flood_extent_ratio",
    ),
    "flood_duration_days": ("flood_duration_days", "duration_days"),
    "flooded_pixel_ratio": ("flooded_pixel_ratio", "flooded_pixels_ratio", "flooded_fraction"),
}


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _lookup(columns: Iterable[str]) -> dict[str, str]:
    normalized = {_norm(c): c for c in columns}
    out: dict[str, str] = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            if _norm(alias) in normalized:
                out[target] = normalized[_norm(alias)]
                break
    return out


def normalize_hazard_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=HAZARD_FEATURE_COLUMNS)
    lookup = _lookup(raw.columns)
    if "event_id" not in lookup or "grid_id" not in lookup:
        raise ValueError("Hazard feature CSV must contain event_id and grid_id columns")
    out = pd.DataFrame()
    out["event_id"] = raw[lookup["event_id"]].astype(str)
    out["grid_id"] = raw[lookup["grid_id"]].astype(str)
    for col in HAZARD_FEATURE_COLUMNS:
        if col in {"event_id", "grid_id"}:
            continue
        if col in lookup:
            out[col] = pd.to_numeric(raw[lookup[col]], errors="coerce")
        else:
            out[col] = np.nan
    return out[HAZARD_FEATURE_COLUMNS]


def _candidate_csvs(root: Path) -> list[Path]:
    candidates = [
        root / "data/raw/events/grid_event_hazard_features.csv",
        root / "data/raw/events/grid_event_cyclone_features.csv",
        root / "data/raw/rainfall/gpm_imerg/grid_event_rainfall_features.csv",
        root / "data/raw/rainfall/gpm_imerg/grid_event_gpm_rainfall.csv",
        root / "data/raw/flood/sentinel1_flood/grid_event_flood_features.csv",
        root / "data/raw/flood/global_flood_database/grid_event_flood_features.csv",
    ]
    return [p for p in candidates if p.exists()]


def build_hazard_features(
    *,
    root: str | Path,
    output_path: str | Path,
    template_path: str | Path | None = None,
) -> pd.DataFrame:
    root = Path(root)
    files = _candidate_csvs(root)
    if not files:
        template = pd.DataFrame(columns=HAZARD_FEATURE_COLUMNS)
        if template_path is not None:
            ensure_parent(template_path)
            template.to_csv(template_path, index=False)
            LOGGER.warning("No raw hazard feature CSV found. Wrote template: %s", template_path)
        write_table(template, output_path, index=False)
        return template

    merged: pd.DataFrame | None = None
    for path in files:
        LOGGER.info("Reading hazard feature file: %s", path)
        frame = normalize_hazard_frame(pd.read_csv(path))
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["event_id", "grid_id"], how="outer", suffixes=("", "_new"))
            for col in HAZARD_FEATURE_COLUMNS:
                if col in {"event_id", "grid_id"}:
                    continue
                new_col = f"{col}_new"
                if new_col in merged.columns:
                    merged[col] = merged[col].combine_first(merged[new_col])
                    merged = merged.drop(columns=new_col)
    assert merged is not None
    merged = merged.groupby(["event_id", "grid_id"], as_index=False).first()
    merged = merged[HAZARD_FEATURE_COLUMNS]
    write_table(merged, output_path, index=False)
    LOGGER.info("Wrote hazard features with %s rows: %s", len(merged), output_path)
    return merged
