from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ntlpol.io_utils import read_table


def plot_lgbm_feature_importance(*, importance_path: str | Path, output_path: str | Path, top_n: int = 30) -> Path | None:
    imp = read_table(importance_path)
    if imp.empty:
        return None
    summary = imp.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(top_n)
    fig = plt.figure(figsize=(8, max(4, 0.25 * len(summary))))
    plt.barh(summary["feature"][::-1], summary["importance"][::-1])
    plt.xlabel("Mean importance")
    plt.title("LightGBM feature importance")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
