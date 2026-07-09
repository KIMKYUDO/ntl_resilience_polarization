from __future__ import annotations

import pandas as pd

from ntlpol.evaluation.metrics import top_k_recall


def event_topk_table(
    predictions: pd.DataFrame,
    *,
    k_fracs: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for event_id, g in predictions.groupby("event_id"):
        row = {"event_id": event_id}
        for k in k_fracs:
            row[f"top{int(k * 100)}_recall"] = top_k_recall(
                g["y_delayed_slowest_20pct"], g["pred_delayed_prob"], k_frac=k
            )
        rows.append(row)
    return pd.DataFrame(rows)
