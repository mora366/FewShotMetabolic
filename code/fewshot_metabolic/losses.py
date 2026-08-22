from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .schema import LossOutput, MetabolicBatch, ModelOutput


def classification_prediction_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("classification logits must have two columns")
    if target.shape != (logits.shape[0],):
        raise ValueError("classification target shape mismatch")
    return torch.nn.functional.cross_entropy(logits, target)


def regression_prediction_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("regression prediction shape mismatch")
    return torch.mean(torch.square(target - prediction))


def gaussian_regression_loss(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if mean.shape != target.shape or log_variance.shape != target.shape:
        raise ValueError("uncertainty prediction shape mismatch")
    precision = torch.exp(-log_variance)
    return 0.5 * torch.mean(log_variance + precision * torch.square(target - mean))


def pathway_consistency_loss(
    representations: tuple[torch.Tensor, ...],
    margin: float,
) -> torch.Tensor:
    if len(representations) < 2:
        raise ValueError("pathway consistency requires multiple pathways")
    total = representations[0].new_zeros(())
    comparisons = 0
    for first_index, first in enumerate(representations):
        for second_index, second in enumerate(representations):
            if first_index == second_index:
                continue
            similarity = torch.nn.functional.cosine_similarity(first, second, dim=-1)
            total = total + torch.relu(similarity - margin).mean()
            comparisons += 1
    return total / comparisons


def adapter_regularization(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    selected = tuple(parameters)
    if not selected:
        raise ValueError("adapter regularization received no parameters")
    total = selected[0].new_zeros(())
    for parameter in selected:
        total = total + torch.sum(torch.square(parameter))
    return total


class FewShotMetabolicLoss(nn.Module):
    def __init__(
        self,
        pathway_weight: float,
        adapter_weight: float,
        pathway_margin: float,
        regression_weight: float = 1.0,
        use_predictive_variance: bool = False,
    ) -> None:
        super().__init__()
        self.pathway_weight = pathway_weight
        self.adapter_weight = adapter_weight
        self.pathway_margin = pathway_margin
        self.regression_weight = regression_weight
        self.use_predictive_variance = use_predictive_variance

    def forward(
        self,
        output: ModelOutput,
        batch: MetabolicBatch,
        adapter_parameters: Iterable[torch.Tensor],
    ) -> LossOutput:
        classification = classification_prediction_loss(
            output.classification_logits,
            batch.classification_target,
        )
        if self.use_predictive_variance:
            regression = gaussian_regression_loss(
                output.glycemic_mean,
                output.glycemic_log_variance,
                batch.regression_target,
            )
        else:
            regression = regression_prediction_loss(
                output.glycemic_mean,
                batch.regression_target,
            )
        prediction = classification + self.regression_weight * regression
        pathway = pathway_consistency_loss(
            output.pathway_representations, self.pathway_margin
        )
        adapter = adapter_regularization(adapter_parameters)
        total = (
            prediction + self.pathway_weight * pathway + self.adapter_weight * adapter
        )
        return LossOutput(
            total=total, prediction=prediction, pathway=pathway, adapter=adapter
        )


def focal_classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    cross_entropy = torch.nn.functional.cross_entropy(logits, target, reduction="none")
    probability = torch.exp(-cross_entropy)
    values = torch.pow(1.0 - probability, gamma) * cross_entropy
    if alpha is not None:
        values = alpha * values
    return values.mean()
