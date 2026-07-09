from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ntlpol.evaluation.metrics import evaluate_prediction_frame
from ntlpol.io_utils import read_json, read_table, write_table
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.train_lgbm")

TARGETS = [
    "y_delayed_slowest_20pct",
    "recovery_delay_percentile",
    "y_no_recovery_12m",
    "y_no_recovery_24m",
]


def _feature_columns(x_tab: pd.DataFrame) -> list[str]:
    return [c for c in x_tab.columns if c not in {"sample_id", "event_id", "grid_id"}]


def _make_classifier(seed: int) -> Pipeline:
    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        class_weight="balanced",
        verbosity=-1,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def _make_regressor(seed: int) -> Pipeline:
    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbosity=-1,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def _fit_predict_binary(model: Pipeline, x_train, y_train, x_test) -> np.ndarray:
    if len(np.unique(y_train.dropna().astype(int))) < 2:
        return np.full(len(x_test), float(y_train.mean()) if len(y_train) else np.nan)
    model.fit(x_train, y_train.astype(int))
    return model.predict_proba(x_test)[:, 1]


def train_lgbm_leave_one_event(
    *,
    x_tab_path: str | Path,
    y_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models/lgbm"
    model_dir.mkdir(parents=True, exist_ok=True)

    x_tab = read_table(x_tab_path)
    y = read_table(y_path)
    splits = read_json(split_path)
    if x_tab.empty or y.empty or not splits:
        pred = pd.DataFrame(
            columns=[
                "sample_id",
                "event_id",
                "grid_id",
                *TARGETS,
                "pred_delayed_prob",
                "pred_recovery_percentile",
                "pred_no_recovery_12m_prob",
                "pred_no_recovery_24m_prob",
                "fold",
            ]
        )
        metrics = pd.DataFrame(columns=["fold"])
        write_table(pred, output_dir / "predictions/lgbm_predictions.parquet", index=False)
        write_table(metrics, output_dir / "metrics/lgbm_event_metrics.parquet", index=False)
        LOGGER.warning("Empty modeling data or splits. Wrote empty LGBM outputs.")
        return pred, metrics

    data = x_tab.merge(y, on=["sample_id", "event_id", "grid_id"], how="inner")
    feature_cols = _feature_columns(x_tab)
    preds = []
    metrics_rows = []
    importances = []

    for fold, split in splits.items():
        train_ids = set(split.get("train", []))
        test_ids = set(split.get("test", []))
        train = data[data["sample_id"].isin(train_ids)].copy()
        test = data[data["sample_id"].isin(test_ids)].copy()
        if train.empty or test.empty:
            continue
        x_train = train[feature_cols]
        x_test = test[feature_cols]

        delayed_model = _make_classifier(seed)
        no12_model = _make_classifier(seed + 12)
        no24_model = _make_classifier(seed + 24)
        pct_model = _make_regressor(seed + 1)

        pred = test[["sample_id", "event_id", "grid_id"] + TARGETS].copy()
        pred["pred_delayed_prob"] = _fit_predict_binary(
            delayed_model, x_train, train["y_delayed_slowest_20pct"], x_test
        )
        valid_reg = train["recovery_delay_percentile"].notna()
        if valid_reg.sum() >= 2:
            pct_model.fit(x_train.loc[valid_reg], train.loc[valid_reg, "recovery_delay_percentile"])
            pred["pred_recovery_percentile"] = pct_model.predict(x_test).clip(0, 1)
        else:
            pred["pred_recovery_percentile"] = np.nan
        pred["pred_no_recovery_12m_prob"] = _fit_predict_binary(
            no12_model, x_train, train["y_no_recovery_12m"], x_test
        )
        pred["pred_no_recovery_24m_prob"] = _fit_predict_binary(
            no24_model, x_train, train["y_no_recovery_24m"], x_test
        )
        pred["fold"] = fold
        preds.append(pred)

        row = {"fold": fold, **evaluate_prediction_frame(pred)}
        metrics_rows.append(row)

        # Save fold models; useful for later inference and audit.
        joblib.dump(
            {
                "feature_cols": feature_cols,
                "delayed_model": delayed_model,
                "percentile_model": pct_model,
                "no12_model": no12_model,
                "no24_model": no24_model,
            },
            model_dir / f"lgbm_{fold}.joblib",
        )
        for name, model in [
            ("delayed", delayed_model),
            ("percentile", pct_model),
            ("no12", no12_model),
            ("no24", no24_model),
        ]:
            fitted = model.named_steps.get("model")
            if hasattr(fitted, "feature_importances_"):
                for col, imp in zip(feature_cols, fitted.feature_importances_):
                    importances.append({"fold": fold, "head": name, "feature": col, "importance": imp})

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    metrics_df = pd.DataFrame(metrics_rows)
    imp_df = pd.DataFrame(importances)
    write_table(pred_df, output_dir / "predictions/lgbm_predictions.parquet", index=False)
    write_table(metrics_df, output_dir / "metrics/lgbm_event_metrics.parquet", index=False)
    write_table(imp_df, output_dir / "tables/lgbm_feature_importance.parquet", index=False)
    LOGGER.info("Wrote LGBM predictions for %s samples", len(pred_df))
    return pred_df, metrics_df
