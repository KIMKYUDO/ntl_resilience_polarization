from __future__ import annotations

import numpy as np
import pandas as pd


def gini(values: pd.Series | np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.any(x < 0):
        x = x - np.min(x)
    if np.allclose(x.sum(), 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def event_level_polarization(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive event-level recovery inequality metrics."""
    if metrics.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "n_valid_grids",
                "delay_score_gini",
                "p90_p10_delay_gap",
                "median_recovery_score_24m",
                "no_recovery_24m_rate",
            ]
        )
    valid = metrics[metrics["valid_target"].astype(bool)].copy()
    rows = []
    for event_id, g in valid.groupby("event_id"):
        delay = g["recovery_delay_score"]
        rows.append(
            {
                "event_id": event_id,
                "n_valid_grids": len(g),
                "delay_score_gini": gini(delay),
                "p90_p10_delay_gap": float(delay.quantile(0.9) - delay.quantile(0.1)),
                "median_recovery_score_24m": float(g["recovery_score_24m"].median()),
                "no_recovery_24m_rate": float(g["y_no_recovery_24m"].mean()),
            }
        )
    return pd.DataFrame(rows)
