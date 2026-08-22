from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch

from .classification_metrics import ClassificationMetrics, classification_metrics
from .model import FewShotMetabolic
from .regression_metrics import (
    IntervalMetrics,
    RegressionMetrics,
    empirical_prediction_interval,
    interval_metrics,
    regression_metrics,
)
from .schema import MetabolicBatch


@dataclass(frozen=True)
class EvaluationResult:
    classification: ClassificationMetrics
    regression: RegressionMetrics
    intervals: IntervalMetrics | None
    sample_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "classification": asdict(self.classification),
            "regression": asdict(self.regression),
            "intervals": asdict(self.intervals) if self.intervals is not None else None,
            "sample_count": self.sample_count,
        }


def model_predictions(
    model: FewShotMetabolic,
    batches: list[MetabolicBatch],
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    glycemic: list[np.ndarray] = []
    classification_targets: list[np.ndarray] = []
    regression_targets: list[np.ndarray] = []
    with torch.no_grad():
        for source in batches:
            batch = source.to(device)
            output = model(batch.features, batch.phenotype)
            probabilities.append(
                torch.softmax(output.classification_logits, dim=1)[:, 1].cpu().numpy()
            )
            glycemic.append(output.glycemic_mean.cpu().numpy())
            classification_targets.append(batch.classification_target.cpu().numpy())
            regression_targets.append(batch.regression_target.cpu().numpy())
    return tuple(
        np.concatenate(values)
        for values in (
            probabilities,
            glycemic,
            classification_targets,
            regression_targets,
        )
    )


def evaluate_model(
    model: FewShotMetabolic,
    batches: list[MetabolicBatch],
    calibration_bins: int = 15,
    device: torch.device | str = "cpu",
) -> EvaluationResult:
    probability, glycemic, classification_target, regression_target = model_predictions(
        model, batches, device
    )
    return EvaluationResult(
        classification=classification_metrics(
            classification_target, probability, calibration_bins
        ),
        regression=regression_metrics(regression_target, glycemic),
        intervals=None,
        sample_count=classification_target.size,
    )


def repeated_adaptation_intervals(
    adapt_and_predict: Callable[[int], np.ndarray],
    target: np.ndarray,
    repetitions: int = 20,
    confidence: float = 0.95,
) -> IntervalMetrics:
    if repetitions != 20:
        raise ValueError(
            "the uncertainty protocol uses twenty independently adapted models"
        )
    samples = np.stack(
        [adapt_and_predict(index) for index in range(repetitions)], axis=0
    )
    lower, upper = empirical_prediction_interval(samples, confidence)
    return interval_metrics(target, lower, upper)


def phenotype_stratified_results(
    target: np.ndarray,
    probability: np.ndarray,
    glycemic_target: np.ndarray,
    glycemic_prediction: np.ndarray,
    phenotype: np.ndarray,
    calibration_bins: int = 15,
) -> dict[int, EvaluationResult]:
    arrays = [
        np.asarray(value).reshape(-1)
        for value in (
            target,
            probability,
            glycemic_target,
            glycemic_prediction,
            phenotype,
        )
    ]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("stratified evaluation arrays must share a shape")
    results: dict[int, EvaluationResult] = {}
    for value in np.unique(arrays[4]):
        members = arrays[4] == value
        results[int(value)] = EvaluationResult(
            classification=classification_metrics(
                arrays[0][members], arrays[1][members], calibration_bins
            ),
            regression=regression_metrics(arrays[2][members], arrays[3][members]),
            intervals=None,
            sample_count=int(np.sum(members)),
        )
    return results


def gate_summary(gates: np.ndarray, phenotype: np.ndarray) -> dict[int, np.ndarray]:
    gate_array = np.asarray(gates, dtype=np.float64)
    phenotype_array = np.asarray(phenotype, dtype=np.int64).reshape(-1)
    if gate_array.ndim != 2 or gate_array.shape[0] != phenotype_array.size:
        raise ValueError("gate and phenotype shapes differ")
    return {
        int(value): np.mean(gate_array[phenotype_array == value], axis=0)
        for value in np.unique(phenotype_array)
    }
