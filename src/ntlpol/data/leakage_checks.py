from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ntlpol.io_utils import read_json, read_table, write_json
from ntlpol.logging_utils import setup_logger

LOGGER = setup_logger("ntlpol.leakage_checks")

TARGET_LEAKAGE_KEYWORDS = [
    "target",
    "recovery_score",
    "recovery_delay",
    "delayed_slowest",
    "no_recovery",
    "t50",
    "resilience_index",
]


def make_leave_one_event_out_splits(index: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    splits: dict[str, dict[str, list[int]]] = {}
    for event_id in sorted(index["event_id"].dropna().unique()):
        test = index.loc[index["event_id"] == event_id, "sample_id"].astype(int).tolist()
        train = index.loc[index["event_id"] != event_id, "sample_id"].astype(int).tolist()
        splits[str(event_id)] = {"train": train, "test": test}
    return splits


def make_leave_one_year_out_splits(index: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    if "event_year" not in index.columns:
        return {}
    splits: dict[str, dict[str, list[int]]] = {}
    for year in sorted(index["event_year"].dropna().unique()):
        test = index.loc[index["event_year"] == year, "sample_id"].astype(int).tolist()
        train = index.loc[index["event_year"] != year, "sample_id"].astype(int).tolist()
        splits[str(int(year))] = {"train": train, "test": test}
    return splits


def write_splits(*, modeling_index_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = read_table(modeling_index_path)
    paths = {
        "leave_one_event_out": write_json(
            output_dir / "leave_one_event_out.json", make_leave_one_event_out_splits(index)
        ),
        "leave_one_year_out": write_json(
            output_dir / "leave_one_year_out.json", make_leave_one_year_out_splits(index)
        ),
    }
    LOGGER.info("Wrote split files to %s", output_dir)
    return paths


def check_no_target_columns_in_features(x_tab: pd.DataFrame) -> list[str]:
    feature_cols = [c for c in x_tab.columns if c not in {"sample_id", "event_id", "grid_id"}]
    bad = [c for c in feature_cols if any(k in c.lower() for k in TARGET_LEAKAGE_KEYWORDS)]
    return bad


def check_leave_one_event_splits(index: pd.DataFrame, splits: dict) -> list[str]:
    errors: list[str] = []
    sample_to_event = index.set_index("sample_id")["event_id"].to_dict()
    for split_name, split in splits.items():
        train_events = {sample_to_event.get(i) for i in split.get("train", [])}
        test_events = {sample_to_event.get(i) for i in split.get("test", [])}
        overlap = train_events & test_events
        overlap.discard(None)
        if overlap:
            errors.append(f"{split_name}: train/test event overlap {sorted(overlap)}")
    return errors


def run_leakage_checks(
    *,
    modeling_index_path: str | Path,
    x_tab_path: str | Path,
    loo_split_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    index = read_table(modeling_index_path)
    x_tab = read_table(x_tab_path)
    splits = read_json(loo_split_path) if Path(loo_split_path).exists() else {}

    errors: list[str] = []
    duplicate_count = int(index.duplicated(["event_id", "grid_id"]).sum()) if not index.empty else 0
    if duplicate_count:
        errors.append(f"Duplicate event_id × grid_id rows in modeling_index: {duplicate_count}")

    bad_feature_cols = check_no_target_columns_in_features(x_tab)
    if bad_feature_cols:
        errors.append(f"Potential target leakage feature columns: {bad_feature_cols}")

    errors.extend(check_leave_one_event_splits(index, splits))
    report = {
        "passed": len(errors) == 0,
        "n_modeling_samples": int(len(index)),
        "duplicate_event_grid_rows": duplicate_count,
        "errors": errors,
    }
    write_json(output_path, report)
    if report["passed"]:
        LOGGER.info("Leakage checks passed.")
    else:
        LOGGER.error("Leakage checks failed: %s", errors)
    return report
