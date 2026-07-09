from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ntlpol.io_utils import read_table, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.make_targets")

TARGET_COLUMNS = [
    "event_id",
    "grid_id",
    "recovery_score_12m",
    "recovery_score_24m",
    "recovery_delay_percentile",
    "y_delayed_slowest_20pct",
    "y_delayed_slowest_30pct",
    "y_no_recovery_12m",
    "y_no_recovery_24m",
    "capped_t50_months",
    "is_t50_censored",
    "resilience_index",
    "recovery_speed",
    "valid_target",
]


def add_event_relative_targets(
    metrics: pd.DataFrame,
    *,
    slow_quantile_main: float = 0.8,
    slow_quantile_secondary: float = 0.7,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    df = metrics.copy()
    if "valid_target" not in df.columns:
        df["valid_target"] = True
    df["recovery_delay_percentile"] = np.nan

    valid = df["valid_target"].astype(bool) & df["recovery_delay_score"].notna()
    df.loc[valid, "recovery_delay_percentile"] = (
        df.loc[valid]
        .groupby("event_id")["recovery_delay_score"]
        .rank(method="average", pct=True)
    )

    df["y_delayed_slowest_20pct"] = (
        df["recovery_delay_percentile"] >= slow_quantile_main
    ).astype("Int64")
    df["y_delayed_slowest_30pct"] = (
        df["recovery_delay_percentile"] >= slow_quantile_secondary
    ).astype("Int64")
    df.loc[~valid, ["y_delayed_slowest_20pct", "y_delayed_slowest_30pct"]] = pd.NA
    return df[TARGET_COLUMNS]


def make_targets(
    *,
    recovery_metrics_path: str | Path,
    output_path: str | Path,
    slow_quantile_main: float = 0.8,
    slow_quantile_secondary: float = 0.7,
) -> pd.DataFrame:
    metrics = read_table(recovery_metrics_path)
    targets = add_event_relative_targets(
        metrics,
        slow_quantile_main=slow_quantile_main,
        slow_quantile_secondary=slow_quantile_secondary,
    )
    write_table(targets, output_path, index=False)
    LOGGER.info("Wrote multi-task targets with %s rows: %s", len(targets), output_path)
    return targets
