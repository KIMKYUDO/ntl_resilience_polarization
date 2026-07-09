from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.data.leakage_checks import run_leakage_checks  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage checks before modeling.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    logger = setup_logger("ntlpol.script.leakage")
    report = run_leakage_checks(
        modeling_index_path=paths.processed / "modeling_index.parquet",
        x_tab_path=paths.processed / "X_tab.parquet",
        loo_split_path=paths.processed / "splits/leave_one_event_out.json",
        output_path=paths.outputs / "metrics/leakage_check_report.json",
    )
    logger.info("Leakage report: %s", report)
    if not report["passed"]:
        raise SystemExit(1)
    logger.info("Next: python scripts/20_train_lgbm_baseline.py")


if __name__ == "__main__":
    main()
