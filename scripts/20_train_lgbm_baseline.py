from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.config import load_config  # noqa: E402
from ntlpol.models.train_lgbm import train_lgbm_leave_one_event  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM tabular Leave-One-Event baseline.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    cfg = load_config(args.config)
    train_lgbm_leave_one_event(
        x_tab_path=paths.processed / "X_tab.parquet",
        y_path=paths.processed / "y_multitask.parquet",
        split_path=paths.processed / "splits/leave_one_event_out.json",
        output_dir=paths.outputs,
        seed=int(cfg.get("training.random_seed", 42)),
    )


if __name__ == "__main__":
    main()
