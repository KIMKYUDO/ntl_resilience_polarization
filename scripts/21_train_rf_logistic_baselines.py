from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.config import load_config  # noqa: E402
from ntlpol.models.train_sklearn_baselines import train_sklearn_baseline  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Logistic/RF tabular baselines.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    cfg = load_config(args.config)
    seed = int(cfg.get("training.random_seed", 42))
    for kind in ["logistic_ridge", "random_forest"]:
        train_sklearn_baseline(
            x_tab_path=paths.processed / "X_tab.parquet",
            y_path=paths.processed / "y_multitask.parquet",
            split_path=paths.processed / "splits/leave_one_event_out.json",
            output_dir=paths.outputs,
            kind=kind,
            seed=seed,
        )


if __name__ == "__main__":
    main()
