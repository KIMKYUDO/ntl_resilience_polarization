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
from ntlpol.targets.recovery_metrics import compute_recovery_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute post-disaster recovery metrics from NTL sequences.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    cfg = load_config(args.config)
    logger = setup_logger("ntlpol.script.recovery_metrics")
    metrics = compute_recovery_metrics(
        sequence_path=paths.interim / "ntl/grid_event_ntl_sequence_full.parquet",
        output_path=paths.interim / "targets/grid_event_recovery_metrics.parquet",
        recovery_threshold=float(cfg.get("targets.no_recovery.threshold_recovery_ratio", 0.9)),
        t50_cap_months=int(cfg.get("targets.censored_t50.cap_months", 24)),
        baseline_min_valid_months=int(cfg.get("ntl.baseline.min_valid_months", 8)),
    )
    logger.info("Recovery metric rows: %s", len(metrics))
    logger.info("Next: python scripts/06_make_targets.py")


if __name__ == "__main__":
    main()
