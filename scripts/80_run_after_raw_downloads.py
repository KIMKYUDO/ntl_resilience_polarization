from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE = [
    "scripts/03_ingest_ntl.py",
    "scripts/04_build_grid_event_ntl_sequences.py",
    "scripts/05_make_recovery_metrics.py",
    "scripts/06_make_targets.py",
    "scripts/07_make_hazard_features.py",
    "scripts/08_make_socioeconomic_features.py",
    "scripts/09_build_modeling_dataset.py",
    "scripts/10_make_splits.py",
    "scripts/11_check_leakage.py",
    "scripts/20_train_lgbm_baseline.py",
    "scripts/21_train_rf_logistic_baselines.py",
    "scripts/30_train_tcn_multitask.py",
    "scripts/31_train_gru_multitask.py",
    "scripts/32_train_transformer_multitask.py",
    "scripts/40_ensemble.py",
    "scripts/50_evaluate_all.py",
    "scripts/60_make_outputs.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modeling pipeline after raw event/grid/NTL/feature CSVs are available.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--skip-deep", action="store_true", help="Run only data pipeline + baseline ML + outputs.")
    parser.add_argument("--stop-on-error", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scripts = PIPELINE.copy()
    if args.skip_deep:
        scripts = [s for s in scripts if not any(x in s for x in ["30_train", "31_train", "32_train", "40_ensemble"])]
    for rel in scripts:
        cmd = [sys.executable, str(args.root / rel), "--root", str(args.root)]
        print(f"\n[RUN] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=args.root)
        if result.returncode != 0:
            print(f"[ERROR] Failed: {rel} (exit={result.returncode})")
            if args.stop_on_error:
                raise SystemExit(result.returncode)
    print("\n[DONE] Pipeline finished.")


if __name__ == "__main__":
    main()
