from __future__ import annotations

from pathlib import Path

import pandas as pd

from ntlpol.evaluation.metrics import evaluate_prediction_frame
from ntlpol.io_utils import read_table, write_table


def evaluate_by_event(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "event_id" not in predictions.columns:
        return pd.DataFrame()
    rows = []
    for event_id, group in predictions.groupby("event_id"):
        rows.append({"event_id": event_id, **evaluate_prediction_frame(group)})
    return pd.DataFrame(rows)


def evaluate_prediction_file(*, prediction_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    pred = read_table(prediction_path)
    event_metrics = evaluate_by_event(pred)
    write_table(event_metrics, output_path, index=False)
    return event_metrics
