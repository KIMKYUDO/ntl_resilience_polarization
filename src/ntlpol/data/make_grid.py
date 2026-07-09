from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None
try:
    from shapely.validation import make_valid as shapely_make_valid
except Exception:  # pragma: no cover - old shapely fallback
    shapely_make_valid = None

try:
    from shapely.errors import GEOSException
except Exception:  # pragma: no cover
    GEOSException = Exception

from ntlpol.io_utils import write_geodataframe, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.make_grid")

GRID_COLUMNS = [
    "grid_id",
    "centroid_lon",
    "centroid_lat",
    "area_km2",
    "district_id",
    "district_name",
    "state_name",
    "is_land",
    "is_valid_sample_area",
    "geometry",
]

LOOKUP_COLUMNS = ["grid_id", "district_id", "district_name", "state_name"]

# India bounding box used only as a development fallback when no boundary file exists.
INDIA_BBOX_WGS84 = (68.0, 6.0, 98.0, 38.5)


def _polygonal_parts(geom):
    """Return polygonal parts from a repaired geometry.

    make_valid can return GeometryCollection or line artifacts. For grid clipping
    and admin joins we only want polygon / multipolygon components.
    """
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        parts = []
        for part in geom.geoms:
            parts.extend(_polygonal_parts(part))
        return parts
    return []


def _repair_geometry(geom):
    """Repair invalid polygon geometry robustly for Shapely/GEOS operations."""
    if geom is None or geom.is_empty:
        return None

    repaired = geom
    if not repaired.is_valid:
        if shapely_make_valid is not None:
            try:
                repaired = shapely_make_valid(repaired)
            except Exception:
                repaired = geom
        if not repaired.is_valid:
            try:
                repaired = repaired.buffer(0)
            except Exception:
                return None

    parts = _polygonal_parts(repaired)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    try:
        return unary_union(parts)
    except Exception:
        return MultiPolygon(parts)


def clean_boundary_geometries(gdf: gpd.GeoDataFrame, *, label: str = "boundary") -> gpd.GeoDataFrame:
    """Drop empty geometries and repair invalid admin boundary polygons.

    Some public admin-boundary datasets contain self-intersections or small
    topology conflicts. Those invalid geometries can make unary_union fail with
    errors such as: "TopologyException: side location conflict". This function
    fixes them before grid generation.
    """
    if gdf.empty:
        raise ValueError(f"{label} GeoDataFrame is empty")

    out = gdf.copy()
    before = len(out)
    invalid_before = int((~out.geometry.is_valid).sum())

    out = out[out.geometry.notna()].copy()
    out["geometry"] = out.geometry.apply(_repair_geometry)
    out = out[out.geometry.notna()].copy()
    out = out[~out.geometry.is_empty].copy()

    # Explode multipolygons so spatial joins and overlay-like operations are safer.
    try:
        out = out.explode(index_parts=False, ignore_index=True)
    except TypeError:  # older GeoPandas
        out = out.explode().reset_index(drop=True)

    invalid_after = int((~out.geometry.is_valid).sum())
    if invalid_before or before != len(out) or invalid_after:
        LOGGER.info(
            "Cleaned %s geometries: rows %s -> %s, invalid %s -> %s",
            label,
            before,
            len(out),
            invalid_before,
            invalid_after,
        )

    if out.empty:
        raise ValueError(f"No valid polygon geometries remain after cleaning {label}")
    return out


def find_vector_file(directory: str | Path) -> Path | None:
    directory = Path(directory)
    if directory.is_file():
        return directory
    if not directory.exists():
        return None
    suffix_priority = [".gpkg", ".shp", ".geojson", ".json"]
    files = [p for p in directory.rglob("*") if p.suffix.lower() in suffix_priority]
    if not files:
        return None
    files.sort(key=lambda p: suffix_priority.index(p.suffix.lower()))
    return files[0]


def read_boundary(path: str | Path, *, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Boundary file is empty: {path}")
    if gdf.crs is None:
        LOGGER.warning("Boundary CRS is missing. Assuming %s for %s", target_crs, path)
        gdf = gdf.set_crs(target_crs)
    return clean_boundary_geometries(gdf.to_crs(target_crs), label=f"boundary file {Path(path).name}")


def _make_fallback_boundary(crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    LOGGER.warning(
        "No India boundary file found. Using rough India bounding box fallback. "
        "Use only for pipeline testing, not final analysis."
    )
    geom = box(*INDIA_BBOX_WGS84)
    return gpd.GeoDataFrame(
        {"country": ["India_bbox_placeholder"], "is_placeholder_boundary": [True]},
        geometry=[geom],
        crs=crs,
    )


def _resolve_boundary(
    *,
    districts_dir: str | Path,
    states_dir: str | Path | None = None,
    crs_geographic: str = "EPSG:4326",
    allow_bbox_fallback: bool = False,
) -> tuple[gpd.GeoDataFrame, bool]:
    district_path = find_vector_file(districts_dir)
    if district_path is not None:
        LOGGER.info("Using district boundary: %s", district_path)
        return read_boundary(district_path, target_crs=crs_geographic), False

    if states_dir is not None:
        state_path = find_vector_file(states_dir)
        if state_path is not None:
            LOGGER.info("Using state boundary as grid clipping boundary: %s", state_path)
            return read_boundary(state_path, target_crs=crs_geographic), False

    if allow_bbox_fallback:
        return _make_fallback_boundary(crs=crs_geographic), True

    raise FileNotFoundError(
        "No India district/state boundary file found. Put a .shp/.gpkg/.geojson under "
        f"{districts_dir} or pass allow_bbox_fallback=True for development only."
    )


def _grid_id(row_idx: int, col_idx: int, resolution_km: float) -> str:
    res = int(resolution_km) if float(resolution_km).is_integer() else str(resolution_km).replace(".", "p")
    return f"G{res}KM_R{row_idx:05d}_C{col_idx:05d}"


def make_projected_grid(
    boundary: gpd.GeoDataFrame,
    *,
    resolution_km: float,
    crs_projected: str,
    min_intersection_ratio: float = 0.05,
    max_cells: int | None = None,
) -> gpd.GeoDataFrame:
    """Create square grid cells over India using a fast centroid-in-boundary test.

    Earlier versions checked `cell.intersection(india_union).area` for every
    candidate cell. That is accurate but extremely slow for a complex ADM2
    boundary. For modeling and GEE zonal extraction, a stable square-cell sample
    unit is more important than clipping every edge cell. Therefore we keep cells
    whose centroid falls inside the cleaned India boundary. This avoids expensive
    repeated polygon intersections and is usually much faster for 5 km grids.

    `min_intersection_ratio` is retained for API compatibility but is not used in
    this fast centroid-based path.
    """
    if resolution_km <= 0:
        raise ValueError("resolution_km must be positive")
    cell_size_m = resolution_km * 1000.0

    boundary_proj = clean_boundary_geometries(
        boundary.to_crs(crs_projected),
        label="projected boundary",
    )

    minx, miny, maxx, maxy = boundary_proj.total_bounds
    n_cols = int(math.ceil((maxx - minx) / cell_size_m))
    n_rows = int(math.ceil((maxy - miny) / cell_size_m))
    estimated = n_cols * n_rows

    LOGGER.info(
        "Generating candidate %skm grid: %s rows x %s cols = about %s cells",
        resolution_km,
        n_rows,
        n_cols,
        f"{estimated:,}",
    )

    if max_cells is not None and estimated > max_cells:
        raise ValueError(
            f"Grid would create about {estimated:,} candidate cells, above max_cells={max_cells:,}. "
            "Use a coarser resolution, increase max_cells, or provide a clipped boundary."
        )

    records: list[dict[str, object]] = []
    geoms: list[Polygon] = []
    LOGGER.info("Step 1/5: creating candidate square polygons")
    row_iter = range(n_rows)
    if tqdm is not None:
        row_iter = tqdm(row_iter, desc="candidate grid rows", unit="row")

    log_every = max(1, n_rows // 10)
    for r in row_iter:
        if tqdm is None and (r == 0 or (r + 1) % log_every == 0 or r + 1 == n_rows):
            LOGGER.info("Candidate grid progress: %s/%s rows (%.1f%%)", r + 1, n_rows, 100.0 * (r + 1) / n_rows)
        y1 = miny + r * cell_size_m
        y2 = y1 + cell_size_m
        for c in range(n_cols):
            x1 = minx + c * cell_size_m
            x2 = x1 + cell_size_m
            records.append({"grid_id": _grid_id(r, c, resolution_km)})
            geoms.append(box(x1, y1, x2, y2))

    LOGGER.info("Step 2/5: converting %s candidates to GeoDataFrame", f"{len(records):,}")
    candidates = gpd.GeoDataFrame(records, geometry=geoms, crs=crs_projected)
    LOGGER.info("Built %s candidate grid cells", f"{len(candidates):,}")

    LOGGER.info("Step 3/5: computing candidate centroids")
    points = candidates[["grid_id", "geometry"]].copy()
    points["geometry"] = points.geometry.centroid

    try:
        LOGGER.info("Step 4/5: spatial join, keeping centroids inside India boundary")
        joined = gpd.sjoin(
            points,
            boundary_proj[["geometry"]],
            how="inner",
            predicate="within",
        )
        keep_ids = joined["grid_id"].drop_duplicates()
        LOGGER.info("Spatial join matched %s unique grid centroids", f"{len(keep_ids):,}")
        grid = candidates[candidates["grid_id"].isin(set(keep_ids))].copy()
    except Exception as exc:  # pragma: no cover - backend dependent
        LOGGER.warning(
            "Fast centroid spatial join failed: %s. Falling back to prepared union intersects check.",
            exc,
        )
        geoms_clean = boundary_proj.geometry.dropna().tolist()
        geom_union = unary_union(geoms_clean)
        mask = points.geometry.apply(lambda g: bool(geom_union.contains(g) or geom_union.touches(g)))
        grid = candidates.loc[mask.to_numpy()].copy()

    if grid.empty:
        raise ValueError("No grid cells selected by the boundary centroid filter")

    LOGGER.info("Selected %s grid cells inside boundary", f"{len(grid):,}")

    LOGGER.info("Step 5/5: computing grid attributes and WGS84 centroids")
    grid["area_km2"] = grid.geometry.area / 1_000_000.0
    centroids_wgs84 = grid.geometry.centroid.to_crs("EPSG:4326")
    grid["centroid_lon"] = centroids_wgs84.x
    grid["centroid_lat"] = centroids_wgs84.y
    grid["is_land"] = True
    grid["is_valid_sample_area"] = True
    return grid

def _pick_col(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {c.lower().replace("_", " "): c for c in columns}
    for cand in candidates:
        key = cand.lower().replace("_", " ")
        if key in normalized:
            return normalized[key]
    return None


def attach_admin_metadata(
    grid: gpd.GeoDataFrame,
    admin_gdf: gpd.GeoDataFrame,
    *,
    crs_projected: str,
) -> gpd.GeoDataFrame:
    """Attach district/state metadata by centroid spatial join.

    If spatial index support is unavailable, GeoPandas still attempts the join;
    if it fails, the function falls back to blank metadata.
    """
    grid = grid.copy()
    try:
        admin = clean_boundary_geometries(
            admin_gdf.copy().to_crs(crs_projected),
            label="admin boundary",
        )
    except ValueError:
        return _blank_admin(grid)
    if admin.empty:
        return _blank_admin(grid)

    district_id_col = _pick_col(admin.columns, ["district_id", "dtcode", "dist_id", "id"])
    district_name_col = _pick_col(
        admin.columns,
        ["district_name", "district", "distname", "dtname", "name_2", "name"],
    )
    state_name_col = _pick_col(
        admin.columns,
        ["state_name", "state", "stname", "name_1", "admin1"],
    )

    attrs = []
    if district_id_col:
        attrs.append(district_id_col)
    if district_name_col:
        attrs.append(district_name_col)
    if state_name_col:
        attrs.append(state_name_col)
    if not attrs:
        return _blank_admin(grid)

    LOGGER.info("Attaching admin metadata: centroid spatial join to district boundary")
    points = grid[["grid_id", "geometry"]].copy()
    points["geometry"] = points.geometry.centroid
    try:
        LOGGER.info("Admin join input: %s grid cells, %s admin polygons", f"{len(points):,}", f"{len(admin):,}")
        joined = gpd.sjoin(
            points,
            admin[attrs + ["geometry"]],
            how="left",
            predicate="within",
        )
    except Exception as exc:  # pragma: no cover - backend dependent
        LOGGER.warning("Admin spatial join failed: %s. Using blank admin metadata.", exc)
        return _blank_admin(grid)

    LOGGER.info("Admin spatial join completed: %s matched rows", f"{len(joined):,}")
    joined = joined.drop_duplicates("grid_id").set_index("grid_id")
    grid["district_id"] = grid["grid_id"].map(
        joined[district_id_col].astype(str) if district_id_col else pd.Series(dtype=str)
    )
    grid["district_name"] = grid["grid_id"].map(
        joined[district_name_col].astype(str) if district_name_col else pd.Series(dtype=str)
    )
    grid["state_name"] = grid["grid_id"].map(
        joined[state_name_col].astype(str) if state_name_col else pd.Series(dtype=str)
    )
    for col in ["district_id", "district_name", "state_name"]:
        grid[col] = grid[col].fillna("")
    return grid


def _blank_admin(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid = grid.copy()
    grid["district_id"] = ""
    grid["district_name"] = ""
    grid["state_name"] = ""
    return grid


def build_grid(
    *,
    districts_dir: str | Path,
    states_dir: str | Path | None,
    output_grid_path: str | Path,
    output_lookup_path: str | Path,
    resolution_km: float = 5,
    crs_geographic: str = "EPSG:4326",
    crs_projected: str = "EPSG:7755",
    allow_bbox_fallback: bool = False,
    min_intersection_ratio: float = 0.05,
    max_cells: int | None = None,
) -> gpd.GeoDataFrame:
    boundary, used_placeholder = _resolve_boundary(
        districts_dir=districts_dir,
        states_dir=states_dir,
        crs_geographic=crs_geographic,
        allow_bbox_fallback=allow_bbox_fallback,
    )

    grid = make_projected_grid(
        boundary,
        resolution_km=resolution_km,
        crs_projected=crs_projected,
        min_intersection_ratio=min_intersection_ratio,
        max_cells=max_cells,
    )
    if used_placeholder:
        grid = _blank_admin(grid)
        grid["is_valid_sample_area"] = False
    else:
        grid = attach_admin_metadata(grid, boundary, crs_projected=crs_projected)

    grid = grid[GRID_COLUMNS]
    LOGGER.info("Writing grid output to %s", output_grid_path)
    write_geodataframe(grid, output_grid_path, index=False)
    lookup = pd.DataFrame(grid.drop(columns="geometry"))[LOOKUP_COLUMNS]
    LOGGER.info("Writing grid-district lookup to %s", output_lookup_path)
    write_table(lookup, output_lookup_path, index=False)
    LOGGER.info("Wrote %s grid cells to %s", f"{len(grid):,}", output_grid_path)
    return grid


@dataclass(frozen=True)
class GridBuildPaths:
    districts_dir: Path
    states_dir: Path
    output_grid_path: Path
    output_lookup_path: Path
