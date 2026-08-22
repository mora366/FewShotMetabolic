from __future__ import annotations

import torch
from torch import nn


class HierarchicalTransformerLayer(nn.Module):
    def __init__(
        self, hidden_dim: int, heads: int, feedforward_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.feedforward_norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden_dim = inputs.shape
        query = (
            self.query(inputs)
            .reshape(batch, tokens, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.key(inputs)
            .reshape(batch, tokens, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.value(inputs)
            .reshape(batch, tokens, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        attended = (
            torch.matmul(weights, value)
            .transpose(1, 2)
            .reshape(batch, tokens, hidden_dim)
        )
        attended = self.output(attended)
        hidden = self.attention_norm(inputs + attended)
        return self.feedforward_norm(hidden + self.feedforward(hidden))


class HierarchicalTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            HierarchicalTransformerLayer(hidden_dim, heads, feedforward_dim, dropout)
            for _ in range(layers)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden
