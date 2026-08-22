from __future__ import annotations

import torch
from torch import nn

from .lora import LowRankLinear


class PathwayAdapter(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, rank: int, alpha: float
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.projection = LowRankLinear(hidden_dim, hidden_dim, rank, alpha)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        encoded = torch.nn.functional.gelu(self.encoder(inputs))
        frozen = self.projection.frozen_output(encoded)
        adapted = frozen + self.projection.adapter_output(encoded)
        mixed = gate * adapted + (1.0 - gate) * frozen
        return self.normalization(mixed)


class PathwayGate(nn.Module):
    def __init__(
        self, phenotype_count: int, embedding_dim: int, pathway_count: int
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(phenotype_count, embedding_dim)
        self.gates = nn.ModuleList(
            nn.Linear(embedding_dim + 1, 1) for _ in range(pathway_count)
        )

    def forward(
        self,
        phenotype: torch.Tensor,
        pathway_inputs: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        embedding = self.embeddings(phenotype)
        activations = []
        for layer, values in zip(self.gates, pathway_inputs, strict=True):
            mean = values.mean(dim=1, keepdim=True)
            activations.append(
                torch.sigmoid(layer(torch.cat((embedding, mean), dim=1)))
            )
        return torch.stack(activations, dim=1)
