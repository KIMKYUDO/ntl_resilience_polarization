from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ntlpol.io_utils import read_table, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.build_ntl_sequence")

SEQUENCE_COLUMNS = [
    "event_id",
    "grid_id",
    "relative_month",
    "calendar_year_month",
    "raw_radiance",
    "cleaned_radiance",
    "baseline_radiance",
    "baseline_valid_months",
    "baseline_normalized_radiance",
    "radiance_anomaly",
    "recovery_ratio",
    "valid_obs_count",
    "coverage_ratio",
]


def add_months(year_month: str, delta: int) -> str:
    return str(pd.Period(year_month, freq="M") + delta)


def build_event_grid_month_index(
    events: pd.DataFrame,
    grid_ids: pd.Series,
    *,
    pre_months: int,
    event_months: int,
    post_months: int,
) -> pd.DataFrame:
    if events.empty or grid_ids.empty:
        return pd.DataFrame(columns=["event_id", "grid_id", "relative_month", "calendar_year_month"])
    rel_months = list(range(-pre_months, post_months + 1))
    grid_ids = grid_ids.dropna().astype(str).drop_duplicates().sort_values().reset_index(drop=True)

    event_frames = []
    for _, ev in events.iterrows():
        event_ym = f"{int(ev['event_year']):04d}-{int(ev['event_month']):02d}"
        rel_df = pd.DataFrame(
            {
                "relative_month": rel_months,
                "calendar_year_month": [add_months(event_ym, r) for r in rel_months],
            }
        )
        grid_df = pd.DataFrame({"grid_id": grid_ids})
        grid_df["_key"] = 1
        rel_df["_key"] = 1
        idx = grid_df.merge(rel_df, on="_key").drop(columns="_key")
        idx.insert(0, "event_id", ev["event_id"])
        event_frames.append(idx)
    return pd.concat(event_frames, ignore_index=True)


def normalize_sequence(seq: pd.DataFrame, *, baseline_min_valid_months: int = 8) -> pd.DataFrame:
    if seq.empty:
        return pd.DataFrame(columns=SEQUENCE_COLUMNS)
    seq = seq.copy()
    baseline_mask = seq["relative_month"].between(-12, -1)
    baseline_stats = (
        seq.loc[baseline_mask]
        .groupby(["event_id", "grid_id"])
        .agg(
            baseline_radiance=("cleaned_radiance", "median"),
            baseline_valid_months=("cleaned_radiance", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    seq = seq.merge(baseline_stats, on=["event_id", "grid_id"], how="left")
    valid_baseline = (seq["baseline_radiance"] > 0) & (
        seq["baseline_valid_months"] >= baseline_min_valid_months
    )
    seq["baseline_normalized_radiance"] = np.where(
        valid_baseline, seq["cleaned_radiance"] / seq["baseline_radiance"], np.nan
    )
    seq["radiance_anomaly"] = np.where(
        valid_baseline, seq["cleaned_radiance"] - seq["baseline_radiance"], np.nan
    )
    # Recovery ratio is intentionally brightness-relative, not fitted T50-based.
    # Threshold targets later use this as observed recovery-to-baseline ratio.
    seq["recovery_ratio"] = seq["baseline_normalized_radiance"]
    return seq[SEQUENCE_COLUMNS]


def build_grid_event_ntl_sequences(
    *,
    event_catalog_path: str | Path,
    monthly_ntl_path: str | Path,
    output_path: str | Path,
    pre_months: int = 12,
    event_months: int = 1,
    post_months: int = 24,
    baseline_min_valid_months: int = 8,
) -> pd.DataFrame:
    events = read_table(event_catalog_path)
    ntl = read_table(monthly_ntl_path)
    if events.empty or ntl.empty:
        seq = pd.DataFrame(columns=SEQUENCE_COLUMNS)
        write_table(seq, output_path, index=False)
        LOGGER.warning("Event catalog or NTL table is empty. Wrote empty sequence table.")
        return seq

    required_events = {"event_id", "event_year", "event_month"}
    required_ntl = {"grid_id", "year_month", "raw_radiance", "cleaned_radiance"}
    if missing := required_events - set(events.columns):
        raise ValueError(f"Event catalog missing columns: {sorted(missing)}")
    if missing := required_ntl - set(ntl.columns):
        raise ValueError(f"Monthly NTL missing columns: {sorted(missing)}")

    idx = build_event_grid_month_index(
        events,
        ntl["grid_id"],
        pre_months=pre_months,
        event_months=event_months,
        post_months=post_months,
    )
    seq = idx.merge(
        ntl,
        left_on=["grid_id", "calendar_year_month"],
        right_on=["grid_id", "year_month"],
        how="left",
    ).drop(columns=["year_month", "source"], errors="ignore")
    if "valid_obs_count" not in seq.columns:
        seq["valid_obs_count"] = np.nan
    if "coverage_ratio" not in seq.columns:
        seq["coverage_ratio"] = np.nan
    seq = normalize_sequence(seq, baseline_min_valid_months=baseline_min_valid_months)
    write_table(seq, output_path, index=False)
    LOGGER.info("Wrote full NTL sequence with %s rows: %s", len(seq), output_path)
    return seq
