from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.models.ensemble import ensemble_predictions  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble LightGBM and sequence model predictions.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--members", nargs="*", default=["lgbm", "tcn_early3", "transformer_early3"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    ensemble_predictions(
        prediction_dir=paths.outputs / "predictions",
        output_dir=paths.outputs,
        members=args.members,
    )


if __name__ == "__main__":
    main()
