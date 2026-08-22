from __future__ import annotations

import math

import torch
from torch import nn


class LowRankLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0 or rank > min(in_features, out_features):
            raise ValueError("rank exceeds linear dimensions")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features), requires_grad=False)
            if bias
            else None
        )
        self.adapter_a = nn.Parameter(torch.empty(rank, in_features))
        self.adapter_b = nn.Parameter(torch.zeros(out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.adapter_a, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_b)
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def frozen_output(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(inputs, self.weight, self.bias)

    def adapter_output(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = torch.nn.functional.linear(inputs, self.adapter_a)
        return self.alpha * torch.nn.functional.linear(projected, self.adapter_b)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.frozen_output(inputs) + self.adapter_output(inputs)

    def merged_weight(self) -> torch.Tensor:
        return self.weight + self.alpha * torch.matmul(self.adapter_b, self.adapter_a)

    def adapter_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.adapter_a, self.adapter_b
