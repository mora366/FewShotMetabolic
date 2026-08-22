from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class Phenotype(IntEnum):
    MHO = 0
    MUO = 1
    MONW = 2
    SO = 3
    MHNO = 4
    MUNO = 5


@dataclass(frozen=True)
class MetabolicBatch:
    features: torch.Tensor
    phenotype: torch.Tensor
    classification_target: torch.Tensor
    regression_target: torch.Tensor

    def validate(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must have shape batch by feature")
        size = self.features.shape[0]
        if self.phenotype.shape != (size,):
            raise ValueError("phenotype shape does not match batch")
        if self.classification_target.shape != (size,):
            raise ValueError("classification target shape does not match batch")
        if self.regression_target.shape != (size,):
            raise ValueError("regression target shape does not match batch")

    def to(self, device: torch.device | str) -> MetabolicBatch:
        return MetabolicBatch(
            features=self.features.to(device),
            phenotype=self.phenotype.to(device),
            classification_target=self.classification_target.to(device),
            regression_target=self.regression_target.to(device),
        )


@dataclass(frozen=True)
class ModelOutput:
    classification_logits: torch.Tensor
    glycemic_mean: torch.Tensor
    glycemic_log_variance: torch.Tensor
    pathway_representations: tuple[torch.Tensor, ...]
    gate_activations: torch.Tensor


@dataclass(frozen=True)
class Episode:
    support: MetabolicBatch
    query: MetabolicBatch
    phenotype: Phenotype


@dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    prediction: torch.Tensor
    pathway: torch.Tensor
    adapter: torch.Tensor
