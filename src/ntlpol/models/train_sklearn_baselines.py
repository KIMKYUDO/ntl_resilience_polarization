from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ntlpol.evaluation.metrics import evaluate_prediction_frame
from ntlpol.io_utils import read_json, read_table, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.train_sklearn")
TARGETS = [
    "y_delayed_slowest_20pct",
    "recovery_delay_percentile",
    "y_no_recovery_12m",
    "y_no_recovery_24m",
]


def _feature_columns(x_tab: pd.DataFrame) -> list[str]:
    return [c for c in x_tab.columns if c not in {"sample_id", "event_id", "grid_id"}]


def _models(kind: str, seed: int):
    if kind == "logistic_ridge":
        clf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        reg = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        )
    elif kind == "random_forest":
        clf = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
        reg = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=5,
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown baseline kind: {kind}")
    return clf, reg


def _fit_predict_binary(model, x_train, y_train, x_test):
    if len(np.unique(y_train.dropna().astype(int))) < 2:
        return np.full(len(x_test), float(y_train.mean()) if len(y_train) else np.nan)
    model.fit(x_train, y_train.astype(int))
    return model.predict_proba(x_test)[:, 1]


def train_sklearn_baseline(
    *,
    x_tab_path: str | Path,
    y_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    kind: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_tab = read_table(x_tab_path)
    y = read_table(y_path)
    splits = read_json(split_path)
    if x_tab.empty or y.empty or not splits:
        pred = pd.DataFrame()
        metrics = pd.DataFrame()
        write_table(pred, output_dir / f"predictions/{kind}_predictions.parquet", index=False)
        write_table(metrics, output_dir / f"metrics/{kind}_event_metrics.parquet", index=False)
        return pred, metrics

    data = x_tab.merge(y, on=["sample_id", "event_id", "grid_id"], how="inner")
    feature_cols = _feature_columns(x_tab)
    preds = []
    rows = []
    for fold, split in splits.items():
        train = data[data["sample_id"].isin(split.get("train", []))]
        test = data[data["sample_id"].isin(split.get("test", []))]
        if train.empty or test.empty:
            continue
        clf1, reg = _models(kind, seed)
        clf12, _ = _models(kind, seed + 12)
        clf24, _ = _models(kind, seed + 24)
        pred = test[["sample_id", "event_id", "grid_id"] + TARGETS].copy()
        pred["pred_delayed_prob"] = _fit_predict_binary(
            clf1, train[feature_cols], train["y_delayed_slowest_20pct"], test[feature_cols]
        )
        valid_reg = train["recovery_delay_percentile"].notna()
        if valid_reg.sum() >= 2:
            reg.fit(train.loc[valid_reg, feature_cols], train.loc[valid_reg, "recovery_delay_percentile"])
            pred["pred_recovery_percentile"] = reg.predict(test[feature_cols]).clip(0, 1)
        else:
            pred["pred_recovery_percentile"] = np.nan
        pred["pred_no_recovery_12m_prob"] = _fit_predict_binary(
            clf12, train[feature_cols], train["y_no_recovery_12m"], test[feature_cols]
        )
        pred["pred_no_recovery_24m_prob"] = _fit_predict_binary(
            clf24, train[feature_cols], train["y_no_recovery_24m"], test[feature_cols]
        )
        pred["fold"] = fold
        preds.append(pred)
        rows.append({"fold": fold, **evaluate_prediction_frame(pred)})

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    metrics = pd.DataFrame(rows)
    write_table(pred_df, output_dir / f"predictions/{kind}_predictions.parquet", index=False)
    write_table(metrics, output_dir / f"metrics/{kind}_event_metrics.parquet", index=False)
    LOGGER.info("Wrote %s predictions for %s samples", kind, len(pred_df))
    return pred_df, metrics
