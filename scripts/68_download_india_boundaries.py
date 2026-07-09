from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ntlpol.logging_utils import setup_logger  # noqa: E402
from ntlpol.paths import ProjectPaths  # noqa: E402

GEOB_API = "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/"
DEFAULT_FILENAME = "geoboundaries_india_adm2.geojson"


def _download_text(url: str, timeout: int = 120) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ntl-resilience-polarization/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        return response.read().decode("utf-8")


def _download_binary(url: str, output_path: Path, timeout: int = 300) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "ntl-resilience-polarization/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
        output_path.write_bytes(response.read())


def download_geoboundaries_india_adm2(output_dir: Path, overwrite: bool = False) -> Path:
    logger = setup_logger("ntlpol.script.download_india_boundaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / DEFAULT_FILENAME
    metadata_path = output_dir / "geoboundaries_india_adm2_metadata.json"

    if output_path.exists() and not overwrite:
        logger.info("Boundary already exists: %s", output_path)
        return output_path

    logger.info("Fetching geoBoundaries metadata: %s", GEOB_API)
    metadata = json.loads(_download_text(GEOB_API))

    download_url = metadata.get("gjDownloadURL") or metadata.get("simplifiedGeometryGeoJSON")
    if not download_url:
        raise RuntimeError("geoBoundaries API response did not include gjDownloadURL.")

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Downloading India ADM2 GeoJSON from geoBoundaries.")
    _download_binary(download_url, output_path)

    logger.info("Saved boundary: %s", output_path)
    logger.info("Saved metadata: %s", metadata_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download geoBoundaries India ADM2 boundary to data/raw/boundaries/india_admin_districts."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_root(args.root)
    output_dir = paths.raw / "boundaries/india_admin_districts"
    path = download_geoboundaries_india_adm2(output_dir=output_dir, overwrite=args.overwrite)

    print(f"[DONE] Downloaded/verified: {path}")
    print("[NEXT] python scripts/02_make_grid.py --resolution-km 5 --max-cells 500000")


if __name__ == "__main__":
    main()
