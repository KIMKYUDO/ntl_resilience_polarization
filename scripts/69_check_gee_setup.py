from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.config import load_config  # noqa: E402
from ntlpol.extractors.gee_common import initialize_ee  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check Earth Engine auth/project/grid asset before submitting export tasks.")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/gee.yaml")
    p.add_argument("--ee-project", type=str, default=None)
    p.add_argument("--grid-asset", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, validate=False)
    project = args.ee_project or cfg.get("earth_engine.project") or "global-terrain-501005-d6"
    grid_asset = args.grid_asset or cfg.get("earth_engine.grid_asset")
    logger = setup_logger("ntlpol.script.check_gee_setup")

    logger.info("Checking Earth Engine project: %s", project)
    ee = initialize_ee(project=project)
    logger.info("Earth Engine initialized successfully.")

    if not grid_asset:
        raise ValueError("No grid asset configured. Set earth_engine.grid_asset in configs/gee.yaml or pass --grid-asset.")

    logger.info("Checking grid FeatureCollection asset: %s", grid_asset)
    fc = ee.FeatureCollection(grid_asset)
    count = fc.size().getInfo()
    first = fc.first().toDictionary().getInfo()
    logger.info("Grid asset is readable. Feature count: %s", count)
    logger.info("First feature properties: %s", first)
    if "grid_id" not in first:
        raise ValueError("Grid asset does not contain a 'grid_id' property. Export scripts require grid_id.")
    logger.info("GEE setup check passed.")


if __name__ == "__main__":
    main()
