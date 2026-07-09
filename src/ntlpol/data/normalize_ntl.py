from __future__ import annotations

import pandas as pd

from ntlpol.data.build_ntl_sequence import normalize_sequence


def add_baseline_normalization(seq: pd.DataFrame, *, baseline_min_valid_months: int = 8) -> pd.DataFrame:
    """Public wrapper kept as a separate module for the Phase 2 roadmap."""
    return normalize_sequence(seq, baseline_min_valid_months=baseline_min_valid_months)
