from __future__ import annotations

import torch
from torch import nn


class GRUEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x_seq)
        return h_n[-1]
