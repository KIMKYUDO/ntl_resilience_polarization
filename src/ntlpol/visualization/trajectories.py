from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ntlpol.io_utils import read_table


def plot_event_mean_trajectories(*, sequence_path: str | Path, output_dir: str | Path) -> list[Path]:
    seq = read_table(sequence_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if seq.empty:
        return []
    paths = []
    for event_id, g in seq.groupby("event_id"):
        mean = g.groupby("relative_month", as_index=False)["recovery_ratio"].mean()
        fig = plt.figure()
        plt.plot(mean["relative_month"], mean["recovery_ratio"])
        plt.axvline(0, linestyle="--")
        plt.xlabel("Relative month")
        plt.ylabel("Mean recovery ratio")
        plt.title(f"NTL recovery trajectory: {event_id}")
        path = output_dir / f"trajectory_{event_id}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
