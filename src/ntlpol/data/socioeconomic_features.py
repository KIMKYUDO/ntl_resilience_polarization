from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ntlpol.io_utils import ensure_parent, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.socioeconomic_features")

STATIC_FEATURE_COLUMNS = [
    "grid_id",
    "population_density",
    "built_up_ratio",
    "road_density",
    "distance_to_major_road_km",
    "distance_to_city_center_km",
    "electrification_proxy",
    "high_income_proxy",
    "elevation_m",
    "slope_deg",
    "distance_to_coast_km",
    "distance_to_river_km",
]

ALIASES = {
    "grid_id": ("grid_id", "grid id", "cell_id", "pixel_id"),
    "population_density": ("population_density", "pop_density", "population_per_km2"),
    "built_up_ratio": ("built_up_ratio", "builtup_ratio", "built_ratio", "urban_ratio"),
    "road_density": ("road_density", "roads_density", "road_km_per_km2"),
    "distance_to_major_road_km": ("distance_to_major_road_km", "dist_major_road_km"),
    "distance_to_city_center_km": ("distance_to_city_center_km", "dist_city_center_km"),
    "electrification_proxy": ("electrification_proxy", "power_grid_proxy", "electricity_proxy"),
    "high_income_proxy": ("high_income_proxy", "wealth_proxy", "income_proxy"),
    "elevation_m": ("elevation_m", "elevation", "dem_m"),
    "slope_deg": ("slope_deg", "slope", "slope_degree"),
    "distance_to_coast_km": ("distance_to_coast_km", "dist_coast_km"),
    "distance_to_river_km": ("distance_to_river_km", "dist_river_km"),
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


def normalize_static_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=STATIC_FEATURE_COLUMNS)
    lookup = _lookup(raw.columns)
    if "grid_id" not in lookup:
        raise ValueError("Static feature CSV must contain grid_id column")
    out = pd.DataFrame({"grid_id": raw[lookup["grid_id"]].astype(str)})
    for col in STATIC_FEATURE_COLUMNS:
        if col == "grid_id":
            continue
        out[col] = pd.to_numeric(raw[lookup[col]], errors="coerce") if col in lookup else np.nan
    return out[STATIC_FEATURE_COLUMNS]


def _candidate_csvs(root: Path) -> list[Path]:
    candidates = [
        root / "data/raw/socioeconomic/grid_static_features.csv",
        root / "data/raw/socioeconomic/population/grid_population_features.csv",
        root / "data/raw/socioeconomic/builtup/grid_builtup_features.csv",
        root / "data/raw/socioeconomic/roads/grid_road_features.csv",
        root / "data/raw/socioeconomic/electrification/grid_power_features.csv",
    ]
    return [p for p in candidates if p.exists()]


def build_socioeconomic_features(
    *,
    root: str | Path,
    output_path: str | Path,
    template_path: str | Path | None = None,
) -> pd.DataFrame:
    root = Path(root)
    files = _candidate_csvs(root)
    if not files:
        template = pd.DataFrame(columns=STATIC_FEATURE_COLUMNS)
        if template_path is not None:
            ensure_parent(template_path)
            template.to_csv(template_path, index=False)
            LOGGER.warning("No raw static feature CSV found. Wrote template: %s", template_path)
        write_table(template, output_path, index=False)
        return template

    merged: pd.DataFrame | None = None
    for path in files:
        LOGGER.info("Reading static feature file: %s", path)
        frame = normalize_static_feature_frame(pd.read_csv(path))
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="grid_id", how="outer", suffixes=("", "_new"))
            for col in STATIC_FEATURE_COLUMNS:
                if col == "grid_id":
                    continue
                new_col = f"{col}_new"
                if new_col in merged.columns:
                    merged[col] = merged[col].combine_first(merged[new_col])
                    merged = merged.drop(columns=new_col)
    assert merged is not None
    merged = merged.groupby("grid_id", as_index=False).first()[STATIC_FEATURE_COLUMNS]
    write_table(merged, output_path, index=False)
    LOGGER.info("Wrote static features with %s rows: %s", len(merged), output_path)
    return merged
