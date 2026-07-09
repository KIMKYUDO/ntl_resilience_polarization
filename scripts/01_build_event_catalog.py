from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.config import load_config  # noqa: E402
from ntlpol.data.event_catalog import build_event_catalog  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cleaned India tropical cyclone / urban flooding event catalog."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument(
        "--raw-events-dir",
        type=Path,
        default=None,
        help="Directory containing manual_event_catalog.csv, emdat_india_events.csv, etc.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    cfg = load_config(args.config)
    logger = setup_logger("ntlpol.script.event_catalog")

    include_types = cfg.require("events.include_types")
    country = cfg.get("events.country_filter", "India")
    raw_events_dir = args.raw_events_dir or (paths.raw / "events")
    output_path = args.output or (paths.interim / "events/event_catalog_clean.parquet")
    template_path = raw_events_dir / "manual_event_catalog_template.csv"

    df = build_event_catalog(
        raw_events_dir=raw_events_dir,
        output_path=output_path,
        template_path=template_path,
        country_filter=country,
        include_types=include_types,
    )

    logger.info("Event catalog rows: %s", len(df))
    if df.empty:
        logger.info("Fill %s, save as manual_event_catalog.csv, then rerun.", template_path)
    else:
        logger.info("Next: python scripts/02_make_grid.py")


if __name__ == "__main__":
    main()
