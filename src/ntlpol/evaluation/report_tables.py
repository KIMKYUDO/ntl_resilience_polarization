from __future__ import annotations

from pathlib import Path

import pandas as pd

from ntlpol.evaluation.metrics import evaluate_prediction_frame
from ntlpol.io_utils import read_table, write_table


def collect_model_metrics(*, prediction_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    prediction_dir = Path(prediction_dir)
    rows = []
    for path in sorted(prediction_dir.glob("*_predictions.*")):
        if path.suffix.lower() not in {".csv", ".parquet"}:
            continue
        model = path.name.replace("_predictions", "").split(".")[0]
        try:
            pred = read_table(path)
        except Exception:
            continue
        row = {"model": model, **evaluate_prediction_frame(pred)} if not pred.empty else {"model": model}
        rows.append(row)
    table = pd.DataFrame(rows)
    write_table(table, output_path, index=False)
    return table
