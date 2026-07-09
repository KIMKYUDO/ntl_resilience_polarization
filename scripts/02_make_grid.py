from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.config import load_config  # noqa: E402
from ntlpol.data.make_grid import build_grid  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create India 1km/5km grid cells for modeling.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument("--resolution-km", type=float, default=None)
    parser.add_argument(
        "--allow-bbox-fallback",
        action="store_true",
        help="Use rough India bounding box if no boundary file exists. Development only.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Abort if estimated candidate grid cells exceed this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    cfg = load_config(args.config)
    logger = setup_logger("ntlpol.script.make_grid")

    resolution_km = args.resolution_km or float(cfg.require("spatial.grid_resolution_km"))
    suffix = f"{int(resolution_km)}km" if float(resolution_km).is_integer() else f"{resolution_km}km"

    grid = build_grid(
        districts_dir=paths.raw / "boundaries/india_admin_districts",
        states_dir=paths.raw / "boundaries/india_states",
        output_grid_path=paths.interim / f"grid/india_grid_{suffix}.parquet",
        output_lookup_path=paths.interim / "grid/grid_district_lookup.parquet",
        resolution_km=resolution_km,
        crs_geographic=cfg.get("spatial.crs_geographic", "EPSG:4326"),
        crs_projected=cfg.get("spatial.crs_projected", "EPSG:7755"),
        allow_bbox_fallback=args.allow_bbox_fallback,
        max_cells=args.max_cells,
    )

    logger.info("Grid rows: %s", len(grid))
    if not grid["is_valid_sample_area"].any():
        logger.warning(
            "Grid was generated from placeholder bbox. Provide India boundary files before final analysis."
        )
    logger.info("Next: python scripts/03_ingest_ntl.py")


if __name__ == "__main__":
    main()
