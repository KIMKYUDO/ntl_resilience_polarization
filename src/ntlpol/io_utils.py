from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


PARQUET_FALLBACK_SUFFIX = ".csv"
GEO_FALLBACK_SUFFIX = ".geojson"


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    path = ensure_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_table(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Write a tabular dataframe.

    The project standard is Parquet. In lightweight environments without pyarrow
    or fastparquet, this falls back to a CSV next to the requested path so that
    the pipeline remains executable during development.
    """
    path = ensure_parent(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=index)
            return path
        except Exception as exc:  # pragma: no cover - depends on optional engines
            fallback = path.with_suffix(PARQUET_FALLBACK_SUFFIX)
            df.to_csv(fallback, index=index)
            print(
                f"[WARN] Could not write parquet ({exc}). "
                f"Wrote CSV fallback: {fallback}"
            )
            return fallback

    if suffix == ".csv":
        df.to_csv(path, index=index)
        return path

    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
        return path

    raise ValueError(f"Unsupported table output suffix: {path.suffix}")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        csv_fallback = path.with_suffix(PARQUET_FALLBACK_SUFFIX)
        if csv_fallback.exists():
            path = csv_fallback
        else:
            raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table input suffix: {path.suffix}")


def write_geodataframe(gdf: Any, path: str | Path, *, index: bool = False) -> Path:
    """Write a GeoDataFrame, preferring the requested format with safe fallback.

    GeoParquet needs optional parquet engines. If unavailable, GeoJSON is used.
    """
    path = ensure_parent(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        try:
            gdf.to_parquet(path, index=index)
            return path
        except Exception as exc:  # pragma: no cover - optional dependency branch
            fallback = path.with_suffix(GEO_FALLBACK_SUFFIX)
            gdf.to_file(fallback, driver="GeoJSON")
            print(
                f"[WARN] Could not write GeoParquet ({exc}). "
                f"Wrote GeoJSON fallback: {fallback}"
            )
            return fallback

    if suffix in {".geojson", ".json"}:
        gdf.to_file(path, driver="GeoJSON")
        return path

    if suffix in {".gpkg"}:
        gdf.to_file(path, driver="GPKG")
        return path

    raise ValueError(f"Unsupported geodata output suffix: {path.suffix}")
