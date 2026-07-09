from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.config import load_config  # noqa: E402
from ntlpol.extractors.gee_common import initialize_ee, load_feature_collection  # noqa: E402
from ntlpol.extractors.gee_ntl import export_vnp46a2_grid_monthly_to_drive  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OPTIONAL: Submit a Google Earth Engine Drive export for pre-aggregated monthly VNP46A2 NTL. Primary project workflow uses scripts/70_gee_export_vnp46a2_daily.py."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gee.yaml")
    parser.add_argument("--ee-project", type=str, default="global-terrain-501005-d6")
    parser.add_argument("--grid-asset", type=str, default="projects/global-terrain-501005-d6/assets/india_grid_5km", help="Earth Engine FeatureCollection asset ID for grid cells.")
    parser.add_argument(
        "--grid-geojson",
        type=Path,
        default=None,
        help="Small local GeoJSON fallback. Use only for testing; national grids should be EE assets.",
    )
    parser.add_argument("--start", type=str, required=True, help="Start month, YYYY-MM")
    parser.add_argument("--end", type=str, required=True, help="End month, YYYY-MM, inclusive")
    parser.add_argument("--drive-folder", type=str, default=None)
    parser.add_argument("--file-prefix", type=str, default="grid_monthly_ntl")
    parser.add_argument("--scale-m", type=int, default=None)
    parser.add_argument("--tile-scale", type=int, default=None)
    parser.add_argument("--id-property", type=str, default=None)
    parser.add_argument("--raw-band", action="store_true", help="Use DNB_BRDF_Corrected_NTL instead of gap-filled band.")
    return parser.parse_args()


def _parse_ym(value: str) -> tuple[int, int]:
    year, month = value.split("-")
    return int(year), int(month)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, validate=False)
    logger = setup_logger("ntlpol.script.gee_vnp46a2")
    start_year, start_month = _parse_ym(args.start)
    end_year, end_month = _parse_ym(args.end)

    ee = initialize_ee(project=args.ee_project)
    grid_fc = load_feature_collection(
        ee=ee,
        asset_id=args.grid_asset,
        geojson_path=args.grid_geojson,
        id_property=args.id_property,
    )
    task = export_vnp46a2_grid_monthly_to_drive(
        ee,
        grid_fc=grid_fc,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        drive_folder=args.drive_folder,
        file_name_prefix=args.file_prefix,
        description=f"export_vnp46a2_{args.start}_to_{args.end}",
        scale_m=args.scale_m,
        tile_scale=args.tile_scale,
        id_property=args.id_property,
        dataset_id=cfg.get("gee.datasets.vnp46a2", "NASA/VIIRS/002/VNP46A2"),
        use_gap_filled=not args.raw_band,
    )
    logger.info("Submitted GEE task: %s", task.id)
    logger.warning("This is an optional fallback. Primary workflow: scripts/70_gee_export_vnp46a2_daily.py")
    logger.info("After the Drive CSV is ready, save it as data/raw/ntl/vnp46a2_daily_or_monthly_ingest/grid_monthly_ntl.csv")


if __name__ == "__main__":
    main()
