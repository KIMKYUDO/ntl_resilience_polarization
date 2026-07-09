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
from ntlpol.extractors.gee_socioeconomic import export_ghsl_grid_static_to_drive  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a GEE Drive export for GHSL population/built-up grid features.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gee.yaml")
    parser.add_argument("--ee-project", type=str, default="global-terrain-501005-d6")
    parser.add_argument("--grid-asset", type=str, default="projects/global-terrain-501005-d6/assets/india_grid_5km")
    parser.add_argument("--grid-geojson", type=Path, default=None)
    parser.add_argument("--drive-folder", type=str, default=None)
    parser.add_argument("--file-prefix", type=str, default="grid_static_ghsl_features")
    parser.add_argument("--epoch-year", type=int, default=2020)
    parser.add_argument("--scale-m", type=int, default=None)
    parser.add_argument("--tile-scale", type=int, default=None)
    parser.add_argument("--id-property", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, validate=False)
    logger = setup_logger("ntlpol.script.gee_ghsl")
    args.ee_project = args.ee_project or cfg.get("earth_engine.project")
    args.grid_asset = args.grid_asset or cfg.get("earth_engine.grid_asset")
    args.drive_folder = args.drive_folder or cfg.get("earth_engine.drive_folder", "ntl_resilience_exports")
    args.scale_m = args.scale_m or int(cfg.get("exports.ghsl_static.scale_m", 100))
    args.tile_scale = args.tile_scale or int(cfg.get("exports.ghsl_static.tile_scale", 4))
    args.id_property = args.id_property or cfg.get("earth_engine.grid_id_property", "grid_id")

    ee = initialize_ee(project=args.ee_project)
    grid_fc = load_feature_collection(
        ee=ee,
        asset_id=args.grid_asset,
        geojson_path=args.grid_geojson,
        id_property=args.id_property,
    )
    task = export_ghsl_grid_static_to_drive(
        ee,
        grid_fc=grid_fc,
        drive_folder=args.drive_folder,
        epoch_year=args.epoch_year,
        file_name_prefix=args.file_prefix,
        scale_m=args.scale_m,
        tile_scale=args.tile_scale,
        id_property=args.id_property,
        pop_dataset_id=cfg.get("gee.datasets.ghsl_pop", "JRC/GHSL/P2023A/GHS_POP"),
        built_dataset_id=cfg.get("gee.datasets.ghsl_built", "JRC/GHSL/P2023A/GHS_BUILT_S"),
    )
    logger.info("Submitted GEE task: %s", task.id)
    logger.info("After the Drive CSV is ready, save it as data/raw/socioeconomic/grid_static_features.csv")


if __name__ == "__main__":
    main()
