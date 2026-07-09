from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.config import load_config  # noqa: E402
from ntlpol.data.build_ntl_sequence import build_grid_event_ntl_sequences  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event_id × grid_id NTL sequence table.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    cfg = load_config(args.config)
    logger = setup_logger("ntlpol.script.build_sequence")
    seq = build_grid_event_ntl_sequences(
        event_catalog_path=paths.interim / "events/event_catalog_clean.parquet",
        monthly_ntl_path=paths.interim / "ntl/grid_monthly_ntl.parquet",
        output_path=paths.interim / "ntl/grid_event_ntl_sequence_full.parquet",
        pre_months=int(cfg.require("time_window.pre_months")),
        event_months=int(cfg.get("time_window.event_months", 1)),
        post_months=int(cfg.require("time_window.post_months_for_measurement")),
        baseline_min_valid_months=int(cfg.get("ntl.baseline.min_valid_months", 8)),
    )
    logger.info("Sequence rows: %s", len(seq))
    logger.info("Next: python scripts/05_make_recovery_metrics.py")


if __name__ == "__main__":
    main()
