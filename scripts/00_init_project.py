from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

# Allow running before editable install:
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.config import load_config  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


SCHEMA = {
    "event_catalog_clean": {
        "event_id": "string",
        "event_name": "string",
        "event_type": "string",
        "country": "string",
        "state_names": "string_or_list",
        "start_date": "date",
        "end_date": "date",
        "event_year": "int",
        "event_month": "int",
        "source_catalog": "string",
        "ibtracs_sid": "string_or_null",
        "primary_hazard": "string",
        "notes": "string",
    },
    "grid_cells": {
        "grid_id": "string",
        "geometry": "polygon",
        "centroid_lon": "float",
        "centroid_lat": "float",
        "area_km2": "float",
        "district_id": "string",
        "district_name": "string",
        "state_name": "string",
        "is_land": "bool",
        "is_valid_sample_area": "bool",
    },
    "grid_daily_ntl": {
        "grid_id": "string",
        "date": "YYYY-MM-DD",
        "year_month": "YYYY-MM",
        "raw_radiance": "float",
        "cleaned_radiance": "float",
        "valid_obs_count": "int",
        "cloud_or_quality_bad_count": "int",
        "coverage_ratio": "float",
        "source": "string",
    },
    "grid_monthly_ntl": {
        "grid_id": "string",
        "year_month": "YYYY-MM",
        "raw_radiance": "float",
        "cleaned_radiance": "float",
        "valid_obs_count": "int",
        "cloud_or_quality_bad_count": "int",
        "coverage_ratio": "float",
        "source": "string",
    },
    "grid_event_ntl_sequence_full": {
        "event_id": "string",
        "grid_id": "string",
        "relative_month": "int_-12_to_24",
        "calendar_year_month": "YYYY-MM",
        "raw_radiance": "float",
        "cleaned_radiance": "float",
        "baseline_radiance": "float",
        "baseline_normalized_radiance": "float",
        "radiance_anomaly": "float",
        "recovery_ratio": "float",
        "valid_obs_count": "int",
        "coverage_ratio": "float",
    },
    "grid_event_targets": {
        "event_id": "string",
        "grid_id": "string",
        "recovery_score_12m": "float",
        "recovery_score_24m": "float",
        "recovery_delay_percentile": "float_0_to_1",
        "y_delayed_slowest_20pct": "int_0_or_1",
        "y_delayed_slowest_30pct": "int_0_or_1",
        "y_no_recovery_12m": "int_0_or_1",
        "y_no_recovery_24m": "int_0_or_1",
        "capped_t50_months": "float",
        "is_t50_censored": "int_0_or_1",
        "resilience_index": "float",
        "recovery_speed": "float",
        "valid_target": "bool",
    },
}


FEATURE_COLUMNS = {
    "sequence_channels": [
        "raw_radiance",
        "cleaned_radiance",
        "baseline_normalized_radiance",
        "radiance_anomaly",
        "recovery_ratio",
        "valid_obs_count",
    ],
    "static_tabular_features": [
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
    ],
    "event_hazard_features": [
        "cyclone_distance_to_track_km",
        "cyclone_max_wind_near_grid",
        "rainfall_accum_event_mm",
        "rainfall_accum_pre_event_3d_mm",
        "rainfall_accum_post_event_7d_mm",
        "flood_exposure_ratio",
        "flood_duration_days",
        "flooded_pixel_ratio",
    ],
    "event_context_features": [
        "event_year",
        "event_month",
        "event_type_encoded",
        "is_coastal_event",
    ],
    "target_columns": [
        "y_delayed_slowest_20pct",
        "recovery_delay_percentile",
        "y_no_recovery_12m",
        "y_no_recovery_24m",
    ],
}


SEQUENCE_WINDOWS = {
    "full_measurement": {
        "relative_month_start": -12,
        "relative_month_end": 24,
        "length": 37,
        "use": "target_construction_and_descriptive_measurement_only",
    },
    "early3": {
        "relative_month_start": -12,
        "relative_month_end": 3,
        "length": 16,
        "use": "early_prediction_input",
    },
    "early6": {
        "relative_month_start": -12,
        "relative_month_end": 6,
        "length": 19,
        "use": "early_prediction_input",
    },
}


README = """\
# Polarization of Post-Disaster Resilience Analysis Using NTL Time-Series Data

This project quantifies post-disaster recovery polarization in India using
NTL time-series data at event_id × grid_id level.

Main interpretation:
This is not a framework for claiming that AI precisely predicts exact recovery
months. It is a framework for quantifying recovery polarization and identifying
long-term delayed recovery risk areas using early NTL recovery patterns and
spatial/socioeconomic/hazard exposure features.
"""


def write_json_if_missing(path: Path, obj: dict, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text_if_missing(path: Path, text: str, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize the NTL polarization project.")
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root directory. Defaults to repository root.",
    )
    parser.add_argument(
        "--overwrite-metadata",
        action="store_true",
        help="Overwrite generated metadata JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    logger = setup_logger("ntlpol.init")

    logger.info("Project root: %s", paths.root)
    paths.ensure_dirs(keep=True)

    for rel_pkg in [
        "src/ntlpol",
        "src/ntlpol/data",
        "src/ntlpol/extractors",
        "src/ntlpol/targets",
        "src/ntlpol/models",
        "src/ntlpol/evaluation",
        "src/ntlpol/visualization",
    ]:
        write_text_if_missing(paths.root / rel_pkg / "__init__.py", "")

    write_text_if_missing(paths.root / "README.md", README)
    write_text_if_missing(
        paths.root / ".gitignore",
        dedent(
            """\
            __pycache__/
            *.py[cod]
            .venv/
            .env
            .ipynb_checkpoints/
            data/raw/**
            data/interim/**
            data/processed/*.npy
            outputs/models/**
            outputs/final_bundle/**
            !**/.gitkeep
            """
        ),
    )

    write_json_if_missing(
        paths.metadata / "schema.json",
        SCHEMA,
        overwrite=args.overwrite_metadata,
    )
    write_json_if_missing(
        paths.metadata / "feature_columns.json",
        FEATURE_COLUMNS,
        overwrite=args.overwrite_metadata,
    )
    write_json_if_missing(
        paths.metadata / "sequence_windows.json",
        SEQUENCE_WINDOWS,
        overwrite=args.overwrite_metadata,
    )

    cfg_path = paths.configs / "base.yaml"
    if cfg_path.exists():
        cfg = load_config(cfg_path)
        cfg.validate_phase1()
        logger.info("Validated config: %s", cfg_path)
    else:
        logger.warning("configs/base.yaml not found. Create it before running later phases.")

    logger.info("Created/verified project directories and Phase 1 metadata.")
    logger.info("Next: python scripts/01_build_event_catalog.py")


if __name__ == "__main__":
    main()
