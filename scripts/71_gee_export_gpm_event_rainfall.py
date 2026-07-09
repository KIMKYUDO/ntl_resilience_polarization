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
from ntlpol.extractors.gee_rainfall import export_gpm_grid_event_rainfall_to_drive  # noqa: E402
from ntlpol.io_utils import read_table  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a GEE Drive export for GPM event rainfall features.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gee.yaml")
    parser.add_argument("--ee-project", type=str, default="global-terrain-501005-d6")
    parser.add_argument("--grid-asset", type=str, default="projects/global-terrain-501005-d6/assets/india_grid_5km")
    parser.add_argument("--grid-geojson", type=Path, default=None)
    parser.add_argument("--event-catalog", type=Path, default=None)
    parser.add_argument("--drive-folder", type=str, default=None)
    parser.add_argument("--file-prefix", type=str, default="grid_event_gpm_rainfall")
    parser.add_argument("--scale-m", type=int, default=None)
    parser.add_argument("--tile-scale", type=int, default=None)
    parser.add_argument("--id-property", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    cfg = load_config(args.config, validate=False)
    logger = setup_logger("ntlpol.script.gee_gpm")
    args.ee_project = args.ee_project or cfg.get("earth_engine.project")
    args.grid_asset = args.grid_asset or cfg.get("earth_engine.grid_asset")
    args.drive_folder = args.drive_folder or cfg.get("earth_engine.drive_folder", "ntl_resilience_exports")
    args.scale_m = args.scale_m or int(cfg.get("exports.rainfall.scale_m", 11132))
    args.tile_scale = args.tile_scale or int(cfg.get("exports.rainfall.tile_scale", 4))
    args.id_property = args.id_property or cfg.get("earth_engine.grid_id_property", "grid_id")

    event_catalog = args.event_catalog or (paths.interim / "events/event_catalog_clean.parquet")
    events_df = read_table(event_catalog)
    if events_df.empty:
        raise ValueError(f"Event catalog is empty: {event_catalog}")
    required = {"event_id", "start_date", "end_date"}
    if missing := required - set(events_df.columns):
        raise ValueError(f"Event catalog missing columns: {sorted(missing)}")
    events = events_df[list(required)].dropna().to_dict("records")

    ee = initialize_ee(project=args.ee_project)
    grid_fc = load_feature_collection(
        ee=ee,
        asset_id=args.grid_asset,
        geojson_path=args.grid_geojson,
        id_property=args.id_property,
    )
    task = export_gpm_grid_event_rainfall_to_drive(
        ee,
        grid_fc=grid_fc,
        events=events,
        drive_folder=args.drive_folder,
        file_name_prefix=args.file_prefix,
        scale_m=args.scale_m,
        tile_scale=args.tile_scale,
        id_property=args.id_property,
        dataset_id=cfg.get("gee.datasets.gpm_imerg", "NASA/GPM_L3/IMERG_V07"),
    )
    logger.info("Submitted GEE task: %s", task.id)
    logger.info("After the Drive CSV is ready, save or merge it into data/raw/events/grid_event_hazard_features.csv")


if __name__ == "__main__":
    main()
