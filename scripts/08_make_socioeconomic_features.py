from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.data.socioeconomic_features import build_socioeconomic_features  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge grid static socioeconomic/geospatial features.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    logger = setup_logger("ntlpol.script.socioeconomic_features")
    df = build_socioeconomic_features(
        root=paths.root,
        output_path=paths.interim / "features/grid_static_features.parquet",
        template_path=paths.raw / "socioeconomic/grid_static_features_template.csv",
    )
    logger.info("Static feature rows: %s", len(df))
    logger.info("Next: python scripts/09_build_modeling_dataset.py")


if __name__ == "__main__":
    main()
