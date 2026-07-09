from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.config import load_config  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402
from ntlpol.targets.make_targets import make_targets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create event-relative multi-task recovery targets.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    cfg = load_config(args.config)
    logger = setup_logger("ntlpol.script.make_targets")
    targets = make_targets(
        recovery_metrics_path=paths.interim / "targets/grid_event_recovery_metrics.parquet",
        output_path=paths.interim / "targets/grid_event_targets.parquet",
        slow_quantile_main=float(cfg.get("targets.delayed_recovery.slow_quantile", 0.8)),
        slow_quantile_secondary=float(cfg.get("targets.delayed_recovery_secondary.slow_quantile", 0.7)),
    )
    logger.info("Target rows: %s", len(targets))
    logger.info("Next: python scripts/07_make_hazard_features.py")


if __name__ == "__main__":
    main()
