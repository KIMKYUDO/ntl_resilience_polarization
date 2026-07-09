from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.paths import ProjectPaths  # noqa: E402
from ntlpol.visualization.maps import make_risk_map_ready_table  # noqa: E402
from ntlpol.visualization.trajectories import plot_event_mean_trajectories  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final result bundle tables and figures.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def _copy_if_exists(src: Path, dst_dir: Path) -> None:
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
    else:
        alt = src.with_suffix(".csv")
        if alt.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(alt, dst_dir / alt.name)


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    bundle = paths.outputs / "final_bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    # Risk map-ready output is the main policy-facing artifact.
    try:
        make_risk_map_ready_table(
            predictions_path=paths.outputs / "predictions/ensemble_predictions.parquet",
            modeling_index_path=paths.processed / "modeling_index.parquet",
            x_tab_path=paths.processed / "X_tab.parquet",
            output_path=paths.outputs / "maps/slow_recovery_risk_map_ready.parquet",
        )
    except FileNotFoundError:
        pass

    try:
        plot_event_mean_trajectories(
            sequence_path=paths.interim / "ntl/grid_event_ntl_sequence_full.parquet",
            output_dir=paths.outputs / "figures/trajectories",
        )
    except FileNotFoundError:
        pass

    for src in [
        paths.outputs / "metrics/performance_summary.parquet",
        paths.outputs / "predictions/ensemble_predictions.parquet",
        paths.outputs / "maps/slow_recovery_risk_map_ready.parquet",
        paths.outputs / "metrics/leakage_check_report.json",
    ]:
        _copy_if_exists(src, bundle)


if __name__ == "__main__":
    main()
