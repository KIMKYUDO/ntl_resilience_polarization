from __future__ import annotations

from pathlib import Path

import pandas as pd

from ntlpol.io_utils import read_table, write_table


def make_risk_map_ready_table(
    *,
    predictions_path: str | Path,
    modeling_index_path: str | Path,
    x_tab_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    pred = read_table(predictions_path)
    index = read_table(modeling_index_path)
    x_tab = read_table(x_tab_path)
    if pred.empty:
        out = pd.DataFrame()
    else:
        cols = [c for c in ["sample_id", "event_id", "grid_id", "event_year", "event_month"] if c in index.columns]
        out = pred.merge(index[cols], on=["sample_id", "event_id", "grid_id"], how="left")
        # Include lon/lat if upstream static features add them later.
        lonlat_cols = [c for c in ["sample_id", "centroid_lon", "centroid_lat"] if c in x_tab.columns]
        if len(lonlat_cols) > 1:
            out = out.merge(x_tab[lonlat_cols], on="sample_id", how="left")
    write_table(out, output_path, index=False)
    return out
