from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.config import load_config  # noqa: E402
from ntlpol.data.build_modeling_dataset import build_modeling_dataset  # noqa: E402
from ntlpol.io_utils import read_json  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sequence tensors, tabular matrix, and y vectors.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    cfg = load_config(args.config)
    logger = setup_logger("ntlpol.script.modeling_dataset")

    feature_meta = read_json(paths.metadata / "feature_columns.json")
    out = build_modeling_dataset(
        event_catalog_path=paths.interim / "events/event_catalog_clean.parquet",
        sequence_path=paths.interim / "ntl/grid_event_ntl_sequence_full.parquet",
        targets_path=paths.interim / "targets/grid_event_targets.parquet",
        static_features_path=paths.interim / "features/grid_static_features.parquet",
        hazard_features_path=paths.interim / "features/grid_event_hazard_features.parquet",
        output_dir=paths.processed,
        channels=feature_meta["sequence_channels"],
        static_cols=feature_meta["static_tabular_features"],
        hazard_cols=feature_meta["event_hazard_features"],
        event_context_cols=feature_meta["event_context_features"],
        early_post_months=cfg.get("time_window.early_input_post_months", [3, 6]),
        pre_months=int(cfg.get("time_window.pre_months", 12)),
    )
    logger.info("Modeling dataset outputs: %s", json.dumps({k: str(v) for k, v in out.items()}, indent=2))
    logger.info("Next: python scripts/10_make_splits.py")


if __name__ == "__main__":
    main()
