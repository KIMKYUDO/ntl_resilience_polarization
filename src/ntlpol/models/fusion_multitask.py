from __future__ import annotations

import torch
from torch import nn

from ntlpol.models.gru import GRUEncoder
from ntlpol.models.tcn import TCNEncoder
from ntlpol.models.transformer import TransformerSequenceEncoder


class MultiTaskSequenceFusionModel(nn.Module):
    def __init__(
        self,
        *,
        seq_channels: int,
        tab_features: int,
        encoder_type: str = "tcn",
        seq_hidden: int = 64,
        tab_hidden: int = 64,
        fusion_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if encoder_type == "tcn":
            self.seq_encoder = TCNEncoder(seq_channels, hidden_channels=seq_hidden, dropout=dropout)
        elif encoder_type == "gru":
            self.seq_encoder = GRUEncoder(seq_channels, hidden_size=seq_hidden, dropout=dropout)
        elif encoder_type == "transformer":
            self.seq_encoder = TransformerSequenceEncoder(seq_channels, d_model=seq_hidden, dropout=dropout)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.tab_encoder = nn.Sequential(
            nn.Linear(tab_features, tab_hidden),
            nn.BatchNorm1d(tab_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tab_hidden, tab_hidden),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(seq_hidden + tab_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
        )
        self.head_delayed = nn.Linear(fusion_hidden, 1)
        self.head_percentile = nn.Sequential(nn.Linear(fusion_hidden, 1), nn.Sigmoid())
        self.head_no12 = nn.Linear(fusion_hidden, 1)
        self.head_no24 = nn.Linear(fusion_hidden, 1)

    def forward(self, x_seq: torch.Tensor, x_tab: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_h = self.seq_encoder(x_seq)
        tab_h = self.tab_encoder(x_tab)
        h = self.fusion(torch.cat([seq_h, tab_h], dim=1))
        return {
            "delayed_logit": self.head_delayed(h).squeeze(-1),
            "percentile": self.head_percentile(h).squeeze(-1),
            "no_recovery_12m_logit": self.head_no12(h).squeeze(-1),
            "no_recovery_24m_logit": self.head_no24(h).squeeze(-1),
        }
