from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ntlpol.io_utils import read_table, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.recovery_metrics")

RECOVERY_METRIC_COLUMNS = [
    "event_id",
    "grid_id",
    "baseline_radiance",
    "baseline_valid_months",
    "post_valid_months_24m",
    "event_min_recovery_ratio_0_3m",
    "recovery_score_12m",
    "recovery_score_24m",
    "y_no_recovery_12m",
    "y_no_recovery_24m",
    "capped_t50_months",
    "is_t50_censored",
    "resilience_index",
    "recovery_speed",
    "recovery_delay_score",
    "valid_target",
]

KEY_COLUMNS = ["event_id", "grid_id"]
NEEDED_SEQUENCE_COLUMNS = [
    "event_id",
    "grid_id",
    "relative_month",
    "recovery_ratio",
    "baseline_radiance",
    "baseline_valid_months",
]


def _first_month_at_or_above(group: pd.DataFrame, threshold: float, *, max_month: int) -> float:
    """Slow reference helper retained for compatibility/tests."""
    post = group[group["relative_month"].between(1, max_month)].sort_values("relative_month")
    hit = post.loc[post["recovery_ratio"] >= threshold, "relative_month"]
    if hit.empty:
        return float(max_month)
    return float(hit.iloc[0])


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slow reference helper retained for compatibility/tests."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def compute_recovery_metrics_for_group(
    group: pd.DataFrame,
    *,
    recovery_threshold: float = 0.9,
    t50_cap_months: int = 24,
    baseline_min_valid_months: int = 8,
) -> dict[str, object]:
    """Original per-group implementation retained for small debugging/tests.

    The production compute_recovery_metrics() below uses a vectorized wide-table
    implementation because the expanded India grid has about 891k grid-event
    groups and the old Python group loop is too slow.
    """
    group = group.sort_values("relative_month")
    event_id = group["event_id"].iloc[0]
    grid_id = group["grid_id"].iloc[0]
    baseline = group["baseline_radiance"].dropna().median()
    baseline_valid = group["baseline_valid_months"].dropna().median()

    post_24 = group[group["relative_month"].between(1, 24)]
    post_valid_24 = int(post_24["recovery_ratio"].notna().sum())
    early = group[group["relative_month"].between(0, 3)]
    event_min = early["recovery_ratio"].min(skipna=True)

    score_12 = group.loc[group["relative_month"].between(10, 12), "recovery_ratio"].median()
    score_24 = group.loc[group["relative_month"].between(22, 24), "recovery_ratio"].median()
    resilience = post_24["recovery_ratio"].clip(lower=0, upper=1.5).mean()

    early_post = group[group["relative_month"].between(0, 6)]
    speed = _slope(
        early_post["relative_month"].to_numpy(dtype=float),
        early_post["recovery_ratio"].to_numpy(dtype=float),
    )

    t50_threshold = np.nan
    if np.isfinite(event_min) and np.isfinite(baseline) and baseline > 0:
        t50_threshold = event_min + 0.5 * (1.0 - event_min)
    if not np.isfinite(t50_threshold):
        t50_threshold = 0.5

    first_t50 = _first_month_at_or_above(group, t50_threshold, max_month=t50_cap_months)
    reached_90_12 = group.loc[
        group["relative_month"].between(1, 12), "recovery_ratio"
    ].ge(recovery_threshold).any()
    reached_90_24 = group.loc[
        group["relative_month"].between(1, 24), "recovery_ratio"
    ].ge(recovery_threshold).any()

    no_recovery_12 = int(not bool(reached_90_12))
    no_recovery_24 = int(not bool(reached_90_24))
    is_t50_censored = int(
        not group.loc[group["relative_month"].between(1, t50_cap_months), "recovery_ratio"]
        .ge(t50_threshold)
        .any()
    )

    valid_target = bool(
        np.isfinite(baseline)
        and baseline > 0
        and np.isfinite(baseline_valid)
        and baseline_valid >= baseline_min_valid_months
        and post_valid_24 >= 12
    )

    delay_score = (
        float(first_t50)
        + 6.0 * no_recovery_24
        + 3.0 * no_recovery_12
        + 4.0 * max(0.0, 1.0 - float(score_24)) if np.isfinite(score_24) else float(first_t50) + 10.0
    )

    return {
        "event_id": event_id,
        "grid_id": grid_id,
        "baseline_radiance": float(baseline) if np.isfinite(baseline) else np.nan,
        "baseline_valid_months": int(baseline_valid) if np.isfinite(baseline_valid) else 0,
        "post_valid_months_24m": post_valid_24,
        "event_min_recovery_ratio_0_3m": float(event_min) if np.isfinite(event_min) else np.nan,
        "recovery_score_12m": float(score_12) if np.isfinite(score_12) else np.nan,
        "recovery_score_24m": float(score_24) if np.isfinite(score_24) else np.nan,
        "y_no_recovery_12m": no_recovery_12,
        "y_no_recovery_24m": no_recovery_24,
        "capped_t50_months": float(first_t50),
        "is_t50_censored": is_t50_censored,
        "resilience_index": float(resilience) if np.isfinite(resilience) else np.nan,
        "recovery_speed": speed,
        "recovery_delay_score": delay_score,
        "valid_target": valid_target,
    }


def _read_sequence_minimal(sequence_path: str | Path) -> pd.DataFrame:
    path = Path(sequence_path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        seq = pd.read_parquet(path, columns=NEEDED_SEQUENCE_COLUMNS)
    else:
        seq = read_table(path)
        seq = seq[NEEDED_SEQUENCE_COLUMNS].copy()

    missing = set(NEEDED_SEQUENCE_COLUMNS) - set(seq.columns)
    if missing:
        raise ValueError(f"Sequence table missing columns: {sorted(missing)}")

    seq = seq[NEEDED_SEQUENCE_COLUMNS].copy()

    # Use compact dtypes. This avoids the very slow pyarrow ChunkedArray boolean
    # filtering path that appeared in the old per-group implementation.
    seq["event_id"] = seq["event_id"].astype("category")
    seq["grid_id"] = seq["grid_id"].astype("category")
    seq["relative_month"] = pd.to_numeric(seq["relative_month"], errors="coerce").astype("int16")
    for col in ["recovery_ratio", "baseline_radiance", "baseline_valid_months"]:
        seq[col] = pd.to_numeric(seq[col], errors="coerce").astype("float32")
    return seq


def _existing_month_columns(wide: pd.DataFrame, start: int, end: int) -> list[int]:
    cols = []
    available = set(wide.columns)
    for m in range(start, end + 1):
        if m in available:
            cols.append(m)
    return cols


def _median_cols(wide: pd.DataFrame, cols: list[int]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=wide.index, dtype="float32")
    return wide[cols].median(axis=1, skipna=True)


def _mean_clip_cols(wide: pd.DataFrame, cols: list[int], lower: float, upper: float) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=wide.index, dtype="float32")
    return wide[cols].clip(lower=lower, upper=upper).mean(axis=1, skipna=True)


def _count_valid_cols(wide: pd.DataFrame, cols: list[int]) -> pd.Series:
    if not cols:
        return pd.Series(0, index=wide.index, dtype="int16")
    return wide[cols].notna().sum(axis=1).astype("int16")


def _any_ge_cols(wide: pd.DataFrame, cols: list[int], threshold: float) -> np.ndarray:
    if not cols:
        return np.zeros(len(wide), dtype=bool)
    arr = wide[cols].to_numpy(dtype="float32", copy=False)
    return np.nan_to_num(arr >= threshold, nan=False).any(axis=1)


def _vectorized_speed(wide: pd.DataFrame, cols: list[int]) -> np.ndarray:
    if not cols:
        return np.full(len(wide), np.nan, dtype="float32")

    y = wide[cols].to_numpy(dtype="float64", copy=True)
    x = np.asarray(cols, dtype="float64")[None, :]
    mask = np.isfinite(y)

    n = mask.sum(axis=1).astype("float64")
    y0 = np.where(mask, y, 0.0)
    x_full = np.broadcast_to(x, y.shape)
    x0 = np.where(mask, x_full, 0.0)

    sum_x = x0.sum(axis=1)
    sum_y = y0.sum(axis=1)
    sum_xy = (x0 * y0).sum(axis=1)
    sum_x2 = (x0 * x0).sum(axis=1)

    denom = n * sum_x2 - sum_x * sum_x
    slope = (n * sum_xy - sum_x * sum_y) / denom
    slope[(n < 3) | ~np.isfinite(denom) | (denom == 0)] = np.nan
    return slope.astype("float32")


def _vectorized_first_t50(
    wide: pd.DataFrame,
    cols: list[int],
    threshold: np.ndarray,
    *,
    cap_months: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not cols:
        first = np.full(len(wide), float(cap_months), dtype="float32")
        censored = np.ones(len(wide), dtype="int8")
        return first, censored

    post = wide[cols].to_numpy(dtype="float32", copy=False)
    thr = threshold.astype("float32")[:, None]
    hit = np.isfinite(post) & (post >= thr)
    has_hit = hit.any(axis=1)
    months = np.asarray(cols, dtype="float32")
    first = np.where(has_hit, months[hit.argmax(axis=1)], float(cap_months)).astype("float32")
    censored = (~has_hit).astype("int8")
    return first, censored


def compute_recovery_metrics(
    *,
    sequence_path: str | Path,
    output_path: str | Path,
    recovery_threshold: float = 0.9,
    t50_cap_months: int = 24,
    baseline_min_valid_months: int = 8,
) -> pd.DataFrame:
    seq = _read_sequence_minimal(sequence_path)
    if seq.empty:
        metrics = pd.DataFrame(columns=RECOVERY_METRIC_COLUMNS)
        write_table(metrics, output_path, index=False)
        LOGGER.warning("Sequence table is empty. Wrote empty recovery metrics.")
        return metrics

    LOGGER.info("Loaded sequence table: %s rows", len(seq))
    LOGGER.info("Building vectorized recovery-ratio wide table...")

    try:
        rr_wide = seq.pivot(index=KEY_COLUMNS, columns="relative_month", values="recovery_ratio")
    except ValueError:
        LOGGER.warning("Duplicate event/grid/month rows found. Falling back to pivot_table(first).")
        rr_wide = seq.pivot_table(
            index=KEY_COLUMNS,
            columns="relative_month",
            values="recovery_ratio",
            aggfunc="first",
            observed=True,
            sort=False,
        )

    # Normalize month column names to plain int where possible.
    rr_wide.columns = [int(c) for c in rr_wide.columns]
    rr_wide = rr_wide.sort_index(axis=1)

    LOGGER.info("Wide table shape: %s grid-event rows x %s relative-month columns", rr_wide.shape[0], rr_wide.shape[1])

    LOGGER.info("Aggregating baseline fields...")
    base = seq.groupby(KEY_COLUMNS, observed=True, sort=False).agg(
        baseline_radiance=("baseline_radiance", "median"),
        baseline_valid_months=("baseline_valid_months", "median"),
    )
    base = base.reindex(rr_wide.index)

    m_0_3 = _existing_month_columns(rr_wide, 0, 3)
    m_0_6 = _existing_month_columns(rr_wide, 0, 6)
    m_1_12 = _existing_month_columns(rr_wide, 1, 12)
    m_1_24 = _existing_month_columns(rr_wide, 1, 24)
    m_10_12 = _existing_month_columns(rr_wide, 10, 12)
    m_22_24 = _existing_month_columns(rr_wide, 22, 24)
    m_t50 = _existing_month_columns(rr_wide, 1, t50_cap_months)

    LOGGER.info("Computing vectorized recovery metrics...")
    metrics = base.copy()
    metrics["post_valid_months_24m"] = _count_valid_cols(rr_wide, m_1_24)
    metrics["event_min_recovery_ratio_0_3m"] = _median_cols(rr_wide, m_0_3) if False else rr_wide[m_0_3].min(axis=1, skipna=True) if m_0_3 else np.nan
    metrics["recovery_score_12m"] = _median_cols(rr_wide, m_10_12)
    metrics["recovery_score_24m"] = _median_cols(rr_wide, m_22_24)
    metrics["resilience_index"] = _mean_clip_cols(rr_wide, m_1_24, lower=0.0, upper=1.5)
    metrics["recovery_speed"] = _vectorized_speed(rr_wide, m_0_6)

    baseline_arr = metrics["baseline_radiance"].to_numpy(dtype="float64")
    event_min_arr = metrics["event_min_recovery_ratio_0_3m"].to_numpy(dtype="float64")
    threshold_arr = event_min_arr + 0.5 * (1.0 - event_min_arr)
    threshold_arr[~(np.isfinite(event_min_arr) & np.isfinite(baseline_arr) & (baseline_arr > 0))] = 0.5

    first_t50, is_t50_censored = _vectorized_first_t50(
        rr_wide,
        m_t50,
        threshold_arr,
        cap_months=t50_cap_months,
    )
    metrics["capped_t50_months"] = first_t50
    metrics["is_t50_censored"] = is_t50_censored

    reached_90_12 = _any_ge_cols(rr_wide, m_1_12, recovery_threshold)
    reached_90_24 = _any_ge_cols(rr_wide, m_1_24, recovery_threshold)
    metrics["y_no_recovery_12m"] = (~reached_90_12).astype("int8")
    metrics["y_no_recovery_24m"] = (~reached_90_24).astype("int8")

    baseline_valid_arr = metrics["baseline_valid_months"].to_numpy(dtype="float64")
    post_valid_arr = metrics["post_valid_months_24m"].to_numpy(dtype="float64")
    valid_target = (
        np.isfinite(baseline_arr)
        & (baseline_arr > 0)
        & np.isfinite(baseline_valid_arr)
        & (baseline_valid_arr >= baseline_min_valid_months)
        & (post_valid_arr >= 12)
    )
    metrics["valid_target"] = valid_target

    score24 = metrics["recovery_score_24m"].to_numpy(dtype="float64")
    first_t50_arr = metrics["capped_t50_months"].to_numpy(dtype="float64")
    no12 = metrics["y_no_recovery_12m"].to_numpy(dtype="float64")
    no24 = metrics["y_no_recovery_24m"].to_numpy(dtype="float64")
    delay_if_score24 = first_t50_arr + 6.0 * no24 + 3.0 * no12 + 4.0 * np.maximum(0.0, 1.0 - score24)
    delay_if_missing = first_t50_arr + 10.0
    metrics["recovery_delay_score"] = np.where(np.isfinite(score24), delay_if_score24, delay_if_missing)

    metrics = metrics.reset_index()
    # category columns can survive reset_index; convert to plain strings for stable IO.
    metrics["event_id"] = metrics["event_id"].astype(str)
    metrics["grid_id"] = metrics["grid_id"].astype(str)

    # Match the previous output semantics: baseline_valid_months was int when finite.
    metrics["baseline_valid_months"] = metrics["baseline_valid_months"].fillna(0).round().astype("int16")
    metrics["post_valid_months_24m"] = metrics["post_valid_months_24m"].fillna(0).astype("int16")
    metrics["y_no_recovery_12m"] = metrics["y_no_recovery_12m"].astype("int8")
    metrics["y_no_recovery_24m"] = metrics["y_no_recovery_24m"].astype("int8")
    metrics["is_t50_censored"] = metrics["is_t50_censored"].astype("int8")
    metrics["valid_target"] = metrics["valid_target"].astype(bool)

    metrics = metrics.reindex(columns=RECOVERY_METRIC_COLUMNS)
    write_table(metrics, output_path, index=False)
    LOGGER.info("Wrote recovery metrics: %s rows -> %s", len(metrics), output_path)
    return metrics
