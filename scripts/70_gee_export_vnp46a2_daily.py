from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.config import load_config  # noqa: E402
from ntlpol.extractors.gee_common import initialize_ee, load_feature_collection, month_starts  # noqa: E402
from ntlpol.extractors.gee_ntl import export_vnp46a2_grid_daily_month_to_drive  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit Google Earth Engine Drive exports for grid-level DAILY VNP46A2 NTL. "
            "One CSV task is submitted per calendar month to keep task size manageable."
        )
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gee.yaml")
    parser.add_argument("--ee-project", type=str, default="global-terrain-501005-d6")
    parser.add_argument("--grid-asset", type=str, default="projects/global-terrain-501005-d6/assets/india_grid_5km", help="Earth Engine FeatureCollection asset ID for grid cells.")
    parser.add_argument(
        "--grid-geojson",
        type=Path,
        default=None,
        help="Small local GeoJSON fallback. Use only for tests; national grids should be EE assets.",
    )
    parser.add_argument("--start", type=str, required=True, help="Start month, YYYY-MM")
    parser.add_argument("--end", type=str, required=True, help="End month, YYYY-MM, inclusive")
    parser.add_argument("--drive-folder", type=str, default=None)
    parser.add_argument("--file-prefix", type=str, default="grid_daily_ntl")
    parser.add_argument("--scale-m", type=int, default=None)
    parser.add_argument("--tile-scale", type=int, default=None)
    parser.add_argument("--id-property", type=str, default=None)
    parser.add_argument("--raw-band", action="store_true", help="Use DNB_BRDF_Corrected_NTL instead of the gap-filled band.")
    parser.add_argument(
        "--max-months",
        type=int,
        default=None,
        help="Optional safety cap for testing; e.g., --max-months 1 submits only the first monthly task.",
    )
    return parser.parse_args()


def _parse_ym(value: str) -> tuple[int, int]:
    year, month = value.split("-")
    return int(year), int(month)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, validate=False)
    logger = setup_logger("ntlpol.script.gee_vnp46a2_daily")
    args.ee_project = args.ee_project or cfg.get("earth_engine.project")
    args.grid_asset = args.grid_asset or cfg.get("earth_engine.grid_asset")
    args.drive_folder = args.drive_folder or cfg.get("earth_engine.drive_folder", "ntl_resilience_exports")
    args.scale_m = args.scale_m or int(cfg.get("exports.ntl.scale_m", 500))
    args.tile_scale = args.tile_scale or int(cfg.get("exports.ntl.tile_scale", 4))
    args.id_property = args.id_property or cfg.get("earth_engine.grid_id_property", "grid_id")

    start_year, start_month = _parse_ym(args.start)
    end_year, end_month = _parse_ym(args.end)
    months = month_starts(start_year, start_month, end_year, end_month)
    if args.max_months is not None:
        months = months[: args.max_months]

    ee = initialize_ee(project=args.ee_project)
    grid_fc = load_feature_collection(
        ee=ee,
        asset_id=args.grid_asset,
        geojson_path=args.grid_geojson,
        id_property=args.id_property,
    )

    tasks = []
    for year, month in months:
        ym = f"{year:04d}_{month:02d}"
        task = export_vnp46a2_grid_daily_month_to_drive(
            ee,
            grid_fc=grid_fc,
            year=year,
            month=month,
            drive_folder=args.drive_folder,
            file_name_prefix=f"{args.file_prefix}_{ym}",
            description=f"export_vnp46a2_daily_{ym}",
            scale_m=args.scale_m,
            tile_scale=args.tile_scale,
            id_property=args.id_property,
            dataset_id=cfg.get("gee.datasets.vnp46a2", "NASA/VIIRS/002/VNP46A2"),
            use_gap_filled=not args.raw_band,
        )
        tasks.append((ym, task.id))
        logger.info("Submitted daily VNP46A2 task for %s: %s", ym, task.id)

    logger.info("Submitted %s monthly partition task(s).", len(tasks))
    logger.info(
        "After Drive CSVs are ready, place them under "
        "data/raw/ntl/vnp46a2_daily_or_monthly_ingest/ with names like grid_daily_ntl_YYYY_MM.csv."
    )
    logger.info("Then run: python scripts/03_ingest_ntl.py")


if __name__ == "__main__":
    main()
