from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    auc_roc: float
    auc_pr: float
    f1: float
    sensitivity: float
    specificity: float
    expected_calibration_error: float


@dataclass(frozen=True)
class ConfusionCounts:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


def _binary_arrays(
    target: np.ndarray, probability: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    target_array = np.asarray(target, dtype=np.int64).reshape(-1)
    probability_array = np.asarray(probability, dtype=np.float64).reshape(-1)
    if target_array.shape != probability_array.shape:
        raise ValueError("target and probability shapes differ")
    if not np.all(np.isin(target_array, (0, 1))):
        raise ValueError("classification targets must be binary")
    if np.any(~np.isfinite(probability_array)):
        raise ValueError("probabilities must be finite")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probabilities must lie in the unit interval")
    return target_array, probability_array


def confusion_counts(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
) -> ConfusionCounts:
    target_array, probability_array = _binary_arrays(target, probability)
    prediction = probability_array >= threshold
    positive = target_array == 1
    negative = ~positive
    return ConfusionCounts(
        true_positive=int(np.sum(prediction & positive)),
        true_negative=int(np.sum(~prediction & negative)),
        false_positive=int(np.sum(prediction & negative)),
        false_negative=int(np.sum(~prediction & positive)),
    )


def accuracy_score(counts: ConfusionCounts) -> float:
    total = (
        counts.true_positive
        + counts.true_negative
        + counts.false_positive
        + counts.false_negative
    )
    if total == 0:
        raise ValueError("accuracy is undefined for empty data")
    return (counts.true_positive + counts.true_negative) / total


def sensitivity_score(counts: ConfusionCounts) -> float:
    denominator = counts.true_positive + counts.false_negative
    if denominator == 0:
        return float("nan")
    return counts.true_positive / denominator


def specificity_score(counts: ConfusionCounts) -> float:
    denominator = counts.true_negative + counts.false_positive
    if denominator == 0:
        return float("nan")
    return counts.true_negative / denominator


def precision_score(counts: ConfusionCounts) -> float:
    denominator = counts.true_positive + counts.false_positive
    if denominator == 0:
        return 0.0
    return counts.true_positive / denominator


def f1_score(counts: ConfusionCounts) -> float:
    precision = precision_score(counts)
    sensitivity = sensitivity_score(counts)
    if not np.isfinite(sensitivity) or precision + sensitivity == 0.0:
        return 0.0
    return 2.0 * precision * sensitivity / (precision + sensitivity)


def expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 15,
) -> float:
    target_array, probability_array = _binary_arrays(target, probability)
    if bins <= 1:
        raise ValueError("calibration requires at least two bins")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = target_array.size
    error = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == bins - 1:
            members = (probability_array >= lower) & (probability_array <= upper)
        else:
            members = (probability_array >= lower) & (probability_array < upper)
        count = int(np.sum(members))
        if count == 0:
            continue
        confidence = float(np.mean(probability_array[members]))
        frequency = float(np.mean(target_array[members]))
        error += count / total * abs(confidence - frequency)
    return error


def brier_score(target: np.ndarray, probability: np.ndarray) -> float:
    target_array, probability_array = _binary_arrays(target, probability)
    return float(np.mean(np.square(probability_array - target_array)))


def negative_log_likelihood(
    target: np.ndarray,
    probability: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    target_array, probability_array = _binary_arrays(target, probability)
    clipped = np.clip(probability_array, epsilon, 1.0 - epsilon)
    values = target_array * np.log(clipped) + (1 - target_array) * np.log(1.0 - clipped)
    return float(-np.mean(values))


def classification_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 15,
) -> ClassificationMetrics:
    target_array, probability_array = _binary_arrays(target, probability)
    counts = confusion_counts(target_array, probability_array)
    if np.unique(target_array).size != 2:
        auc_roc = float("nan")
        auc_pr = float("nan")
    else:
        auc_roc = float(roc_auc_score(target_array, probability_array))
        auc_pr = float(average_precision_score(target_array, probability_array))
    return ClassificationMetrics(
        accuracy=accuracy_score(counts),
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        f1=f1_score(counts),
        sensitivity=sensitivity_score(counts),
        specificity=specificity_score(counts),
        expected_calibration_error=expected_calibration_error(
            target_array,
            probability_array,
            bins,
        ),
    )


def calibration_curve(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_array, probability_array = _binary_arrays(target, probability)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    mean_probability: list[float] = []
    observed_frequency: list[float] = []
    counts: list[int] = []
    for index in range(bins):
        upper_comparison = (
            probability_array <= boundaries[index + 1]
            if index == bins - 1
            else probability_array < boundaries[index + 1]
        )
        members = (probability_array >= boundaries[index]) & upper_comparison
        count = int(np.sum(members))
        if count:
            mean_probability.append(float(np.mean(probability_array[members])))
            observed_frequency.append(float(np.mean(target_array[members])))
            counts.append(count)
    return (
        np.asarray(mean_probability),
        np.asarray(observed_frequency),
        np.asarray(counts, dtype=np.int64),
    )
