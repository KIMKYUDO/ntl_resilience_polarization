from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EarthEngineUnavailable(RuntimeError):
    """Raised when the earthengine-api package is not installed or authenticated."""


def import_ee() -> Any:
    try:
        import ee  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise EarthEngineUnavailable(
            "earthengine-api is required for GEE exports. Install with: pip install earthengine-api"
        ) from exc
    return ee


def initialize_ee(*, project: str | None = None) -> Any:
    """Initialize Earth Engine and return the imported ee module.

    Parameters
    ----------
    project:
        Optional Google Cloud project ID. Passing it is recommended for newer
        Earth Engine accounts.
    """
    ee = import_ee()
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:  # pragma: no cover - requires EE auth
        raise EarthEngineUnavailable(
            "Could not initialize Earth Engine. Run `earthengine authenticate` first, "
            "then retry with --ee-project YOUR_PROJECT_ID if needed."
        ) from exc
    return ee


def load_feature_collection(
    *,
    ee: Any,
    asset_id: str | None = None,
    geojson_path: str | Path | None = None,
    id_property: str = "grid_id",
    max_local_features: int = 5000,
) -> Any:
    """Load a grid FeatureCollection from an EE asset or a small local GeoJSON.

    Large 1 km/5 km national grids should be uploaded to Earth Engine as an
    asset and passed with --grid-asset. The local GeoJSON path is intentionally
    capped to avoid accidentally serializing hundreds of thousands of features
    into an EE client request.
    """
    if asset_id:
        return ee.FeatureCollection(asset_id)

    if geojson_path is None:
        raise ValueError("Either asset_id or geojson_path must be provided")

    path = Path(geojson_path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    if len(features) > max_local_features:
        raise ValueError(
            f"Local GeoJSON has {len(features):,} features, above max_local_features={max_local_features:,}. "
            "Upload the grid to Earth Engine and use --grid-asset instead."
        )

    ee_features = []
    for i, feat in enumerate(features):
        geom = feat.get("geometry")
        props = feat.get("properties", {}) or {}
        if id_property not in props:
            props[id_property] = f"LOCAL_GRID_{i:06d}"
        ee_features.append(ee.Feature(ee.Geometry(geom), props))
    return ee.FeatureCollection(ee_features)


def submit_drive_table_export(
    *,
    ee: Any,
    collection: Any,
    description: str,
    folder: str,
    file_name_prefix: str,
    selectors: list[str] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "collection": collection,
        "description": description,
        "folder": folder,
        "fileNamePrefix": file_name_prefix,
        "fileFormat": "CSV",
    }
    if selectors:
        kwargs["selectors"] = selectors
    task = ee.batch.Export.table.toDrive(**kwargs)
    task.start()
    return task


def month_starts(start_year: int, start_month: int, end_year: int, end_month: int) -> list[tuple[int, int]]:
    """Return inclusive year-month starts."""
    out: list[tuple[int, int]] = []
    y, m = int(start_year), int(start_month)
    end_key = int(end_year) * 12 + int(end_month)
    while y * 12 + m <= end_key:
        out.append((y, m))
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out
