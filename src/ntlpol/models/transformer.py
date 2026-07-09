from __future__ import annotations

import torch
from torch import nn


class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_channels, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        b, t, _ = x_seq.shape
        x = self.input_proj(x_seq) + self.pos_embedding[:, :t, :]
        h = self.encoder(x)
        return h[:, -1, :]
