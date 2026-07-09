from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.data.cyclone_hazard import compute_cyclone_hazard_features, find_grid_centroid_file  # noqa: E402
from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cyclone distance/wind exposure features from IBTrACS CSV.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--event-catalog", type=Path, default=None)
    parser.add_argument("--ibtracs-csv", type=Path, default=None)
    parser.add_argument("--grid", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--wind-radius-km", type=float, default=100.0)
    parser.add_argument("--chunk-size", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    logger = setup_logger("ntlpol.script.cyclone_hazard")
    event_catalog = args.event_catalog or (paths.interim / "events/event_catalog_clean.parquet")
    ibtracs_csv = args.ibtracs_csv or (paths.raw / "events/ibtracs_tracks.csv")
    if not ibtracs_csv.exists():
        template = paths.raw / "events/ibtracs_tracks_template.csv"
        template.parent.mkdir(parents=True, exist_ok=True)
        template.write_text("sid,season,name,iso_time,lat,lon,usa_wind\n", encoding="utf-8")
        raise FileNotFoundError(
            f"IBTrACS CSV not found: {ibtracs_csv}. Wrote template: {template}"
        )
    grid = args.grid or find_grid_centroid_file(paths.root)
    if grid is None:
        raise FileNotFoundError("No grid file found under data/interim/grid. Run scripts/02_make_grid.py first.")
    output = args.output or (paths.raw / "events/grid_event_cyclone_features.csv")
    df = compute_cyclone_hazard_features(
        event_catalog_path=event_catalog,
        ibtracs_csv_path=ibtracs_csv,
        grid_path=grid,
        output_path=output,
        wind_radius_km=args.wind_radius_km,
        chunk_size=args.chunk_size,
    )
    logger.info("Cyclone hazard rows: %s", len(df))
    logger.info("Next: python scripts/07_make_hazard_features.py")


if __name__ == "__main__":
    main()
