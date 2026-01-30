from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolylineEncoder(nn.Module):
    '''
    PointNet-style polyline encoder:
      points: (B, M, P, F) -> emb: (B, M, D)
    '''
    def __init__(self, in_dim: int, d_model: int, hidden_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.point_mlp = MLP(in_dim, hidden_dim, d_model, num_layers=3, dropout=dropout)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # points: (B, M, P, F)
        B, M, P, Fdim = points.shape
        x = self.point_mlp(points)  # (B,M,P,D)
        x = x.max(dim=2).values  # max pool over points -> (B,M,D)
        x = self.out_ln(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return x


class StateDropout(nn.Module):
    '''
    Drop subsets of the kinematic state during training to discourage shortcut dependencies.
    (Inspired by the "State Dropout Encoder" mentioned in RAP.)
    '''
    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., F)
        if not self.training or self.drop_prob <= 0:
            return x
        # Drop entire feature dims with probability drop_prob (shared across batch/time)
        Fdim = x.shape[-1]
        keep = torch.rand((Fdim,), device=x.device) > self.drop_prob
        keep = keep.to(x.dtype)
        return x * keep


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # key_padding_mask: (B, S) with True meaning "pad" in PyTorch MHA API.
        # Our masks are usually True=valid; we will convert outside.
        h, _ = self.attn(x, key, value, key_padding_mask=key_padding_mask)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))
        return x
