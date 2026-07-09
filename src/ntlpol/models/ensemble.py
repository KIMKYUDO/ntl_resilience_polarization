from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ntlpol.evaluation.metrics import evaluate_prediction_frame
from ntlpol.io_utils import read_table, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.ensemble")

PRED_COLS = [
    "pred_delayed_prob",
    "pred_recovery_percentile",
    "pred_no_recovery_12m_prob",
    "pred_no_recovery_24m_prob",
]

LABEL_COLS = [
    "y_delayed_slowest_20pct",
    "recovery_delay_percentile",
    "y_no_recovery_12m",
    "y_no_recovery_24m",
]


def _read_prediction_if_exists(path: Path, name: str) -> pd.DataFrame:
    try:
        df = read_table(path)
    except FileNotFoundError:
        LOGGER.warning("Prediction file not found: %s", path)
        return pd.DataFrame()
    if df.empty:
        return df
    keep = ["sample_id", "event_id", "grid_id"] + LABEL_COLS + PRED_COLS
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep].copy()
    df = df.rename(columns={c: f"{name}_{c}" for c in PRED_COLS})
    return df


def ensemble_predictions(
    *,
    prediction_dir: str | Path,
    output_dir: str | Path,
    members: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_dir = Path(prediction_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    members = members or ["lgbm", "tcn_early3", "transformer_early3"]

    frames = []
    for member in members:
        frames.append(_read_prediction_if_exists(prediction_dir / f"{member}_predictions.parquet", member))
    frames = [f for f in frames if not f.empty]
    if not frames:
        pred = pd.DataFrame()
        metrics = pd.DataFrame()
        write_table(pred, output_dir / "predictions/ensemble_predictions.parquet", index=False)
        write_table(metrics, output_dir / "metrics/ensemble_metrics.parquet", index=False)
        LOGGER.warning("No non-empty member predictions. Wrote empty ensemble outputs.")
        return pred, metrics

    base_cols = ["sample_id", "event_id", "grid_id"] + LABEL_COLS
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=base_cols, how="outer")

    pred = merged[base_cols].copy()
    for col in PRED_COLS:
        member_cols = [f"{m}_{col}" for m in members if f"{m}_{col}" in merged.columns]
        pred[col] = merged[member_cols].mean(axis=1, skipna=True) if member_cols else np.nan

    metrics = pd.DataFrame([{ "model": "ensemble", **evaluate_prediction_frame(pred)}])
    write_table(pred, output_dir / "predictions/ensemble_predictions.parquet", index=False)
    write_table(metrics, output_dir / "metrics/ensemble_metrics.parquet", index=False)
    LOGGER.info("Wrote ensemble predictions for %s samples", len(pred))
    return pred, metrics
