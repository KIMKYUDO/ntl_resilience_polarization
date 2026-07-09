from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(np.asarray(arr, dtype=float))
    return mask


def safe_binary_metrics(y_true, y_prob, *, threshold: float = 0.5, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = _finite_mask(y_true, y_prob)
    y_true = y_true[mask].astype(int)
    y_prob = y_prob[mask]
    if y_true.size == 0:
        return {f"{prefix}n": 0}
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        f"{prefix}n": float(y_true.size),
        f"{prefix}positive_rate": float(y_true.mean()),
        f"{prefix}precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if len(np.unique(y_true)) > 1:
        out[f"{prefix}auroc"] = float(roc_auc_score(y_true, y_prob))
        out[f"{prefix}auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out[f"{prefix}auroc"] = np.nan
        out[f"{prefix}auprc"] = np.nan
    return out


def top_k_recall(y_true, risk_score, *, k_frac: float = 0.2) -> float:
    y_true = np.asarray(y_true, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)
    mask = _finite_mask(y_true, risk_score)
    y_true = y_true[mask].astype(int)
    risk_score = risk_score[mask]
    if y_true.size == 0 or y_true.sum() == 0:
        return np.nan
    k = max(1, int(np.ceil(k_frac * len(y_true))))
    top_idx = np.argsort(-risk_score)[:k]
    return float(y_true[top_idx].sum() / y_true.sum())


def safe_regression_metrics(y_true, y_pred, *, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = _finite_mask(y_true, y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {f"{prefix}n": 0}
    rho = spearmanr(y_true, y_pred).correlation if y_true.size >= 3 else np.nan
    return {
        f"{prefix}n": float(y_true.size),
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        f"{prefix}spearman": float(rho) if np.isfinite(rho) else np.nan,
    }


def evaluate_prediction_frame(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if {"y_delayed_slowest_20pct", "pred_delayed_prob"}.issubset(df.columns):
        out.update(
            safe_binary_metrics(
                df["y_delayed_slowest_20pct"], df["pred_delayed_prob"], prefix="delayed_"
            )
        )
        out["delayed_top20_recall"] = top_k_recall(
            df["y_delayed_slowest_20pct"], df["pred_delayed_prob"], k_frac=0.2
        )
        out["delayed_top30_recall"] = top_k_recall(
            df["y_delayed_slowest_20pct"], df["pred_delayed_prob"], k_frac=0.3
        )
    if {"recovery_delay_percentile", "pred_recovery_percentile"}.issubset(df.columns):
        out.update(
            safe_regression_metrics(
                df["recovery_delay_percentile"],
                df["pred_recovery_percentile"],
                prefix="percentile_",
            )
        )
    if {"y_no_recovery_12m", "pred_no_recovery_12m_prob"}.issubset(df.columns):
        out.update(
            safe_binary_metrics(
                df["y_no_recovery_12m"], df["pred_no_recovery_12m_prob"], prefix="no12_"
            )
        )
    if {"y_no_recovery_24m", "pred_no_recovery_24m_prob"}.issubset(df.columns):
        out.update(
            safe_binary_metrics(
                df["y_no_recovery_24m"], df["pred_no_recovery_24m_prob"], prefix="no24_"
            )
        )
    return out
