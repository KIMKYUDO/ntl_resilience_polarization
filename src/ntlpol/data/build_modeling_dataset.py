from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ntlpol.io_utils import read_table, write_json, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.modeling_dataset")

TARGET_COLUMNS = [
    "y_delayed_slowest_20pct",
    "recovery_delay_percentile",
    "y_no_recovery_12m",
    "y_no_recovery_24m",
]


def _event_type_encoded(series: pd.Series) -> pd.Series:
    mapping = {"tropical_cyclone": 1, "urban_flooding": 2}
    return series.map(mapping).fillna(0).astype(int)


def _build_index(targets: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    idx = targets[["event_id", "grid_id", "valid_target"]].copy()
    idx = idx[idx["valid_target"].astype(bool)].drop_duplicates(["event_id", "grid_id"])
    event_cols = ["event_id", "event_year", "event_month", "event_type"]
    idx = idx.merge(events[event_cols], on="event_id", how="left")
    idx["event_type_encoded"] = _event_type_encoded(idx["event_type"])
    idx = idx.reset_index(drop=True)
    idx.insert(0, "sample_id", np.arange(len(idx), dtype=int))
    return idx


def _make_sequence_tensor(
    seq: pd.DataFrame,
    index: pd.DataFrame,
    *,
    channels: Sequence[str],
    relative_month_start: int,
    relative_month_end: int,
) -> np.ndarray:
    months = list(range(relative_month_start, relative_month_end + 1))
    n = len(index)
    tensor = np.full((n, len(months), len(channels)), np.nan, dtype=np.float32)
    if n == 0:
        return tensor
    sample_lookup = {
        (row.event_id, row.grid_id): int(row.sample_id)
        for row in index[["sample_id", "event_id", "grid_id"]].itertuples(index=False)
    }
    seq = seq[seq["relative_month"].isin(months)].copy()
    month_lookup = {m: i for i, m in enumerate(months)}
    for row in seq.itertuples(index=False):
        key = (getattr(row, "event_id"), getattr(row, "grid_id"))
        sample_idx = sample_lookup.get(key)
        if sample_idx is None:
            continue
        t = month_lookup.get(int(getattr(row, "relative_month")))
        if t is None:
            continue
        for c_idx, channel in enumerate(channels):
            value = getattr(row, channel, np.nan) if hasattr(row, channel) else np.nan
            try:
                tensor[sample_idx, t, c_idx] = float(value)
            except Exception:
                tensor[sample_idx, t, c_idx] = np.nan
    return tensor


def _make_tabular(
    index: pd.DataFrame,
    static_features: pd.DataFrame,
    hazard_features: pd.DataFrame,
    *,
    static_cols: Sequence[str],
    hazard_cols: Sequence[str],
    event_context_cols: Sequence[str],
) -> pd.DataFrame:
    tab = index[["sample_id", "event_id", "grid_id", "event_year", "event_month", "event_type_encoded"]].copy()
    if not static_features.empty:
        tab = tab.merge(static_features, on="grid_id", how="left")
    else:
        for col in static_cols:
            tab[col] = np.nan
    if not hazard_features.empty:
        tab = tab.merge(hazard_features, on=["event_id", "grid_id"], how="left")
    else:
        for col in hazard_cols:
            tab[col] = np.nan
    if "is_coastal_event" in event_context_cols and "is_coastal_event" not in tab.columns:
        tab["is_coastal_event"] = np.nan
    feature_cols = list(static_cols) + list(hazard_cols) + list(event_context_cols)
    for col in feature_cols:
        if col not in tab.columns:
            tab[col] = np.nan
    return tab[["sample_id", "event_id", "grid_id"] + feature_cols]


def build_modeling_dataset(
    *,
    event_catalog_path: str | Path,
    sequence_path: str | Path,
    targets_path: str | Path,
    static_features_path: str | Path,
    hazard_features_path: str | Path,
    output_dir: str | Path,
    channels: Sequence[str],
    static_cols: Sequence[str],
    hazard_cols: Sequence[str],
    event_context_cols: Sequence[str],
    early_post_months: Sequence[int] = (3, 6),
    pre_months: int = 12,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = read_table(event_catalog_path)
    seq = read_table(sequence_path)
    targets = read_table(targets_path)
    static_features = read_table(static_features_path)
    hazard_features = read_table(hazard_features_path)

    if events.empty or seq.empty or targets.empty:
        index = pd.DataFrame(
            columns=[
                "sample_id",
                "event_id",
                "grid_id",
                "valid_target",
                "event_year",
                "event_month",
                "event_type",
                "event_type_encoded",
            ]
        )
    else:
        index = _build_index(targets, events)

    paths: dict[str, Path] = {}
    index_path = output_dir / "modeling_index.parquet"
    paths["modeling_index"] = write_table(index, index_path, index=False)

    y = index[["sample_id", "event_id", "grid_id"]].merge(
        targets[["event_id", "grid_id"] + TARGET_COLUMNS], on=["event_id", "grid_id"], how="left"
    ) if not index.empty else pd.DataFrame(columns=["sample_id", "event_id", "grid_id"] + TARGET_COLUMNS)
    paths["y_multitask"] = write_table(y, output_dir / "y_multitask.parquet", index=False)

    tab = _make_tabular(
        index,
        static_features,
        hazard_features,
        static_cols=static_cols,
        hazard_cols=hazard_cols,
        event_context_cols=event_context_cols,
    )
    paths["X_tab"] = write_table(tab, output_dir / "X_tab.parquet", index=False)

    for post in early_post_months:
        tensor = _make_sequence_tensor(
            seq,
            index,
            channels=channels,
            relative_month_start=-pre_months,
            relative_month_end=int(post),
        )
        arr_path = output_dir / f"X_seq_early{post}.npy"
        np.save(arr_path, tensor)
        paths[f"X_seq_early{post}"] = arr_path
        LOGGER.info("Wrote %s with shape %s", arr_path, tensor.shape)

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    paths["modeling_columns"] = write_json(
        metadata_dir / "modeling_columns.json",
        {
            "sequence_channels": list(channels),
            "static_tabular_features": list(static_cols),
            "event_hazard_features": list(hazard_cols),
            "event_context_features": list(event_context_cols),
            "target_columns": TARGET_COLUMNS,
        },
    )
    LOGGER.info("Built modeling dataset with %s valid samples", len(index))
    return paths
