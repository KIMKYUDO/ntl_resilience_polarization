from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.config import load_config  # noqa: E402
from ntlpol.data.ingest_ntl import ingest_ntl  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory-safe ingest of grid-level DAILY VNP46A2 CSVs and derivation of monthly modeling table."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument("--chunksize", type=int, default=500_000, help="CSV rows per chunk while reading each raw file.")
    parser.add_argument(
        "--write-daily-table",
        action="store_true",
        help="Also write data/interim/ntl/grid_daily_ntl.parquet. Not recommended for all-India daily exports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    cfg = load_config(args.config)
    logger = setup_logger("ntlpol.script.ingest_ntl")
    output = paths.interim / "ntl/grid_monthly_ntl.parquet"
    daily_output = paths.interim / "ntl/grid_daily_ntl.parquet"
    template = paths.raw / "ntl/vnp46a2_daily_or_monthly_ingest/grid_daily_ntl_template.csv"
    df = ingest_ntl(
        raw_ntl_dir=paths.raw / "ntl/vnp46a2_daily_or_monthly_ingest",
        output_path=output,
        daily_output_path=daily_output,
        template_path=template,
        source=cfg.get("ntl.source", "VNP46A2"),
        chunksize=args.chunksize,
        write_daily_table=args.write_daily_table,
    )
    logger.info("Derived monthly NTL rows: %s", f"{len(df):,}")
    logger.info("Next: python scripts/04_build_grid_event_ntl_sequences.py")


if __name__ == "__main__":
    main()
