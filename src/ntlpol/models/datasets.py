from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ntlpol.io_utils import read_table

TARGET_COLUMNS = [
    "y_delayed_slowest_20pct",
    "recovery_delay_percentile",
    "y_no_recovery_12m",
    "y_no_recovery_24m",
]


class NTLSequenceTabularDataset(Dataset):
    def __init__(
        self,
        *,
        x_seq: np.ndarray,
        x_tab: pd.DataFrame,
        y: pd.DataFrame,
        sample_ids: list[int] | np.ndarray,
        feature_cols: list[str] | None = None,
    ) -> None:
        self.x_seq = x_seq.astype(np.float32)
        self.x_tab_df = x_tab.copy()
        self.y_df = y.copy()
        self.sample_ids = np.asarray(sample_ids, dtype=int)
        self.feature_cols = feature_cols or [
            c for c in x_tab.columns if c not in {"sample_id", "event_id", "grid_id"}
        ]
        self.x_tab_df = self.x_tab_df.set_index("sample_id")
        self.y_df = self.y_df.set_index("sample_id")

        # Fit simple median imputation on the provided subset only.
        subset_tab = self.x_tab_df.loc[self.sample_ids, self.feature_cols].astype(float)
        self.tab_medians = subset_tab.median(axis=0).fillna(0.0)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_id = int(self.sample_ids[idx])
        seq = self.x_seq[sample_id]
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        tab = self.x_tab_df.loc[sample_id, self.feature_cols].astype(float).fillna(self.tab_medians)
        target = self.y_df.loc[sample_id, TARGET_COLUMNS].astype(float)
        return {
            "sample_id": torch.tensor(sample_id, dtype=torch.long),
            "x_seq": torch.tensor(seq, dtype=torch.float32),
            "x_tab": torch.tensor(tab.to_numpy(dtype=np.float32), dtype=torch.float32),
            "y_delayed": torch.tensor(target["y_delayed_slowest_20pct"], dtype=torch.float32),
            "y_percentile": torch.tensor(target["recovery_delay_percentile"], dtype=torch.float32),
            "y_no12": torch.tensor(target["y_no_recovery_12m"], dtype=torch.float32),
            "y_no24": torch.tensor(target["y_no_recovery_24m"], dtype=torch.float32),
        }


def load_modeling_arrays(
    *,
    processed_dir: str | Path,
    early_post_months: int = 3,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    x_seq = np.load(processed_dir / f"X_seq_early{early_post_months}.npy")
    x_tab = read_table(processed_dir / "X_tab.parquet")
    y = read_table(processed_dir / "y_multitask.parquet")
    index = read_table(processed_dir / "modeling_index.parquet")
    return x_seq, x_tab, y, index
