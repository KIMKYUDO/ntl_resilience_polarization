from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ntlpol.io_utils import read_table, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.cyclone_hazard")

IBTRACS_ALIASES: dict[str, tuple[str, ...]] = {
    "sid": ("sid", "SID", "storm_id", "ibtracs_sid", "serial_num"),
    "season": ("season", "SEASON", "year"),
    "name": ("name", "NAME", "storm_name", "cyclone_name"),
    "iso_time": ("iso_time", "ISO_TIME", "time", "datetime", "date"),
    "lat": ("lat", "LAT", "latitude"),
    "lon": ("lon", "LON", "longitude"),
    "wind": ("usa_wind", "USA_WIND", "wmo_wind", "WMO_WIND", "wind", "max_wind"),
}

GRID_ALIASES: dict[str, tuple[str, ...]] = {
    "grid_id": ("grid_id", "cell_id", "pixel_id"),
    "centroid_lon": ("centroid_lon", "lon", "longitude", "x"),
    "centroid_lat": ("centroid_lat", "lat", "latitude", "y"),
}


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _lookup(columns: Iterable[str], aliases: dict[str, tuple[str, ...]]) -> dict[str, str]:
    normalized = {_norm(c): c for c in columns}
    out: dict[str, str] = {}
    for target, opts in aliases.items():
        for opt in opts:
            key = _norm(opt)
            if key in normalized:
                out[target] = normalized[key]
                break
    return out


def normalize_ibtracs(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["sid", "season", "name", "iso_time", "lat", "lon", "wind"])
    lookup = _lookup(raw.columns, IBTRACS_ALIASES)
    required = {"sid", "lat", "lon"}
    if missing := required - set(lookup):
        raise ValueError(f"IBTrACS CSV missing columns: {sorted(missing)}")
    out = pd.DataFrame()
    out["sid"] = raw[lookup["sid"]].astype(str).str.strip()
    out["season"] = pd.to_numeric(raw[lookup["season"]], errors="coerce") if "season" in lookup else np.nan
    out["name"] = raw[lookup["name"]].astype(str).str.strip() if "name" in lookup else ""
    out["iso_time"] = pd.to_datetime(raw[lookup["iso_time"]], errors="coerce") if "iso_time" in lookup else pd.NaT
    out["lat"] = pd.to_numeric(raw[lookup["lat"]], errors="coerce")
    out["lon"] = pd.to_numeric(raw[lookup["lon"]], errors="coerce")
    out["wind"] = pd.to_numeric(raw[lookup["wind"]], errors="coerce") if "wind" in lookup else np.nan
    out = out.dropna(subset=["lat", "lon"])
    out = out[(out["lat"].between(-90, 90)) & (out["lon"].between(-180, 360))].copy()
    out.loc[out["lon"] > 180, "lon"] -= 360
    return out.reset_index(drop=True)


def _read_grid_centroids(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".csv", ".pkl", ".pickle"}:
        df = read_table(path)
    else:
        try:
            import geopandas as gpd
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("geopandas is required to read grid GeoJSON/Shapefile inputs") from exc
        gdf = gpd.read_file(path)
        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        if "centroid_lon" not in df.columns or "centroid_lat" not in df.columns:
            cent = gdf.to_crs("EPSG:4326").geometry.centroid
            df["centroid_lon"] = cent.x
            df["centroid_lat"] = cent.y
    lookup = _lookup(df.columns, GRID_ALIASES)
    required = {"grid_id", "centroid_lon", "centroid_lat"}
    if missing := required - set(lookup):
        raise ValueError(f"Grid centroid table missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "grid_id": df[lookup["grid_id"]].astype(str),
            "centroid_lon": pd.to_numeric(df[lookup["centroid_lon"]], errors="coerce"),
            "centroid_lat": pd.to_numeric(df[lookup["centroid_lat"]], errors="coerce"),
        }
    ).dropna()
    return out.reset_index(drop=True)


def find_grid_centroid_file(root: str | Path) -> Path | None:
    root = Path(root)
    candidates = [
        root / "data/interim/grid/india_grid_5km.parquet",
        root / "data/interim/grid/india_grid_5km.csv",
        root / "data/interim/grid/india_grid_5km.geojson",
        root / "data/interim/grid/india_grid_1km.parquet",
        root / "data/interim/grid/india_grid_1km.csv",
        root / "data/interim/grid/india_grid_1km.geojson",
    ]
    for path in candidates:
        if path.exists():
            return path
    grid_dir = root / "data/interim/grid"
    for suffix in ("*.parquet", "*.csv", "*.geojson", "*.gpkg", "*.shp"):
        found = sorted(grid_dir.glob(suffix))
        if found:
            return found[0]
    return None


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    r = 6371.0088
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _track_for_event(event: pd.Series, tracks: pd.DataFrame) -> pd.DataFrame:
    sid = str(event.get("ibtracs_sid", "") or "").strip()
    if sid and sid.lower() not in {"nan", "none", ""}:
        matched = tracks[tracks["sid"].astype(str).str.strip().eq(sid)]
        if not matched.empty:
            return matched
    year = int(event.get("event_year")) if pd.notna(event.get("event_year")) else None
    name = _slug(event.get("event_name", ""))
    cand = tracks.copy()
    if year is not None and "season" in cand.columns:
        cand = cand[pd.to_numeric(cand["season"], errors="coerce").eq(year)]
    if name:
        by_name = cand[cand["name"].map(_slug).eq(name)]
        if not by_name.empty:
            return by_name
    return pd.DataFrame(columns=tracks.columns)


def compute_cyclone_hazard_features(
    *,
    event_catalog_path: str | Path,
    ibtracs_csv_path: str | Path,
    grid_path: str | Path,
    output_path: str | Path,
    wind_radius_km: float = 100.0,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    events = read_table(event_catalog_path)
    tracks = normalize_ibtracs(pd.read_csv(ibtracs_csv_path))
    grids = _read_grid_centroids(grid_path)

    if events.empty or tracks.empty or grids.empty:
        out = pd.DataFrame(
            columns=["event_id", "grid_id", "cyclone_distance_to_track_km", "cyclone_max_wind_near_grid"]
        )
        write_table(out, output_path, index=False)
        return out

    events = events[events.get("event_type", "").astype(str).eq("tropical_cyclone")].copy()
    records: list[pd.DataFrame] = []
    grid_lon = grids["centroid_lon"].to_numpy(dtype=float)
    grid_lat = grids["centroid_lat"].to_numpy(dtype=float)

    for _, event in events.iterrows():
        event_id = str(event["event_id"])
        track = _track_for_event(event, tracks)
        if track.empty:
            LOGGER.warning("No IBTrACS track matched event_id=%s; leaving cyclone hazard missing.", event_id)
            continue
        t_lon = track["lon"].to_numpy(dtype=float)
        t_lat = track["lat"].to_numpy(dtype=float)
        t_wind = track["wind"].to_numpy(dtype=float)
        event_frames: list[pd.DataFrame] = []
        for start in range(0, len(grids), chunk_size):
            end = min(start + chunk_size, len(grids))
            lon = grid_lon[start:end][:, None]
            lat = grid_lat[start:end][:, None]
            dist = haversine_km(lon, lat, t_lon[None, :], t_lat[None, :])
            min_dist = np.nanmin(dist, axis=1)
            if np.isfinite(t_wind).any():
                within = dist <= float(wind_radius_km)
                wind_matrix = np.where(within, t_wind[None, :], np.nan)
                has_within = np.isfinite(wind_matrix).any(axis=1)
                max_wind = np.full(end - start, np.nan)
                if has_within.any():
                    max_wind[has_within] = np.nanmax(wind_matrix[has_within], axis=1)
                # If no point is inside radius, use nearest-track wind as weaker proxy.
                nearest_idx = np.nanargmin(dist, axis=1)
                nearest_wind = t_wind[nearest_idx]
                max_wind = np.where(np.isfinite(max_wind), max_wind, nearest_wind)
            else:
                max_wind = np.full(end - start, np.nan)
            event_frames.append(
                pd.DataFrame(
                    {
                        "event_id": event_id,
                        "grid_id": grids.iloc[start:end]["grid_id"].to_numpy(),
                        "cyclone_distance_to_track_km": min_dist,
                        "cyclone_max_wind_near_grid": max_wind,
                    }
                )
            )
        records.append(pd.concat(event_frames, ignore_index=True))

    if records:
        out = pd.concat(records, ignore_index=True)
    else:
        out = pd.DataFrame(
            columns=["event_id", "grid_id", "cyclone_distance_to_track_km", "cyclone_max_wind_near_grid"]
        )
    write_table(out, output_path, index=False)
    LOGGER.info("Wrote cyclone hazard features with %s rows: %s", len(out), output_path)
    return out
