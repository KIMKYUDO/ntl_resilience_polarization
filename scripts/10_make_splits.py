from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.data.leakage_checks import write_splits  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Leave-One-Event/Year split JSON files.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    logger = setup_logger("ntlpol.script.splits")
    out = write_splits(
        modeling_index_path=paths.processed / "modeling_index.parquet",
        output_dir=paths.processed / "splits",
    )
    logger.info("Split outputs: %s", out)
    logger.info("Next: python scripts/11_check_leakage.py")


if __name__ == "__main__":
    main()
