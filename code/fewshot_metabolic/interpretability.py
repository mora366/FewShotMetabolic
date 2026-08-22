from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PathwayContribution:
    pathway: str
    gate_activation: float
    metric_change: float


def mean_gate_activations(
    gates: np.ndarray, pathway_names: tuple[str, ...]
) -> dict[str, float]:
    values = np.asarray(gates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(pathway_names):
        raise ValueError("gate matrix does not match pathway names")
    return {
        name: float(np.mean(values[:, index]))
        for index, name in enumerate(pathway_names)
    }


def gate_histograms(
    gates: np.ndarray, pathway_names: tuple[str, ...], bins: int = 20
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values = np.asarray(gates, dtype=np.float64)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("gate activations must lie in the unit interval")
    return {
        name: np.histogram(values[:, index], bins=bins, range=(0.0, 1.0))
        for index, name in enumerate(pathway_names)
    }


def pathway_removal_analysis(
    base_metric: float,
    evaluate_without: Callable[[int], float],
    pathway_names: tuple[str, ...],
    gates: np.ndarray,
) -> list[PathwayContribution]:
    activations = mean_gate_activations(gates, pathway_names)
    output = []
    for index, name in enumerate(pathway_names):
        removed_metric = evaluate_without(index)
        output.append(
            PathwayContribution(name, activations[name], removed_metric - base_metric)
        )
    return output


def permutation_importance(
    features: np.ndarray,
    target: np.ndarray,
    score: Callable[[np.ndarray, np.ndarray], float],
    predict: Callable[[np.ndarray], np.ndarray],
    repeats: int = 20,
    seed: int = 42,
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    target_array = np.asarray(target).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != target_array.size:
        raise ValueError("permutation inputs have incompatible shapes")
    generator = np.random.default_rng(seed)
    baseline = score(target_array, predict(matrix))
    importance = np.empty((repeats, matrix.shape[1]), dtype=np.float64)
    for repeat in range(repeats):
        for column in range(matrix.shape[1]):
            permuted = matrix.copy()
            permuted[:, column] = generator.permutation(permuted[:, column])
            importance[repeat, column] = baseline - score(
                target_array, predict(permuted)
            )
    return importance


def aggregate_feature_importance(
    importance: np.ndarray, groups: dict[str, tuple[int, ...]]
) -> dict[str, float]:
    values = np.asarray(importance, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("importance must have repeat and feature dimensions")
    output = {}
    for name, indices in groups.items():
        if not indices or max(indices) >= values.shape[1]:
            raise ValueError("pathway feature indices are invalid")
        output[name] = float(np.mean(np.abs(values[:, indices])))
    return output


def phenotype_gate_contrasts(
    gates: np.ndarray, phenotype: np.ndarray
) -> dict[tuple[int, int], np.ndarray]:
    gate_array = np.asarray(gates, dtype=np.float64)
    phenotype_array = np.asarray(phenotype).reshape(-1)
    if gate_array.shape[0] != phenotype_array.size:
        raise ValueError("gate and phenotype shapes differ")
    means = {
        int(value): np.mean(gate_array[phenotype_array == value], axis=0)
        for value in np.unique(phenotype_array)
    }
    return {
        (first, second): means[first] - means[second]
        for first, second in combinations_sorted(tuple(means))
    }


def combinations_sorted(values: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (values[first], values[second])
        for first in range(len(values))
        for second in range(first + 1, len(values))
    )
