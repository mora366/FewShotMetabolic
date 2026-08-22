from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, ttest_rel


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class PairedComparison:
    mean_difference: float
    statistic: float
    raw_p_value: float
    holm_p_value: float
    fdr_p_value: float


def bootstrap_indices(
    size: int, iterations: int, generator: np.random.Generator
) -> np.ndarray:
    if size <= 1 or iterations <= 0:
        raise ValueError("bootstrap dimensions must be nontrivial")
    return generator.integers(0, size, size=(iterations, size))


def percentile_interval(
    samples: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def bootstrap_confidence_interval(
    values: np.ndarray, iterations: int = 1000, confidence: float = 0.95, seed: int = 42
) -> ConfidenceInterval:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    generator = np.random.default_rng(seed)
    estimates = np.mean(
        array[bootstrap_indices(array.size, iterations, generator)], axis=1
    )
    lower, upper = percentile_interval(estimates, confidence)
    return ConfidenceInterval(float(np.mean(array)), lower, upper, confidence)


def stratified_bootstrap(
    values: np.ndarray, strata: np.ndarray, iterations: int, seed: int
) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64).reshape(-1)
    strata_array = np.asarray(strata).reshape(-1)
    if values_array.shape != strata_array.shape:
        raise ValueError("values and strata must share a shape")
    generator = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata_array == key) for key in np.unique(strata_array)]
    output = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = [
            generator.choice(group, size=group.size, replace=True) for group in groups
        ]
        output[iteration] = np.mean(values_array[np.concatenate(sampled)])
    return output


def bca_interval(
    values: np.ndarray, bootstrap_estimates: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    bootstrap = np.asarray(bootstrap_estimates, dtype=np.float64).reshape(-1)
    estimate = float(np.mean(array))
    proportion = float(np.clip(np.mean(bootstrap < estimate), 1e-8, 1.0 - 1e-8))
    bias = float(norm.ppf(proportion))
    jackknife = np.asarray(
        [np.mean(np.delete(array, index)) for index in range(array.size)]
    )
    centered = np.mean(jackknife) - jackknife
    denominator = 6.0 * np.power(np.sum(np.square(centered)), 1.5)
    acceleration = (
        float(np.sum(np.power(centered, 3)) / denominator) if denominator else 0.0
    )
    tail = (1.0 - confidence) / 2.0
    lower_z = norm.ppf(tail)
    upper_z = norm.ppf(1.0 - tail)
    lower_q = norm.cdf(
        bias + (bias + lower_z) / (1.0 - acceleration * (bias + lower_z))
    )
    upper_q = norm.cdf(
        bias + (bias + upper_z) / (1.0 - acceleration * (bias + upper_z))
    )
    return float(np.quantile(bootstrap, lower_q)), float(
        np.quantile(bootstrap, upper_q)
    )


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64).reshape(-1)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (values.size - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64).reshape(-1)
    order = np.argsort(values)[::-1]
    adjusted = np.empty_like(values)
    running = 1.0
    for reverse_rank, index in enumerate(order):
        rank = values.size - reverse_rank
        running = min(running, min(1.0, values[index] * values.size / rank))
        adjusted[index] = running
    return adjusted


def paired_comparisons(
    reference: np.ndarray, alternatives: list[np.ndarray]
) -> list[PairedComparison]:
    reference_array = np.asarray(reference, dtype=np.float64).reshape(-1)
    tests: list[tuple[float, float, float]] = []
    for alternative in alternatives:
        alternative_array = np.asarray(alternative, dtype=np.float64).reshape(-1)
        if alternative_array.shape != reference_array.shape:
            raise ValueError("paired comparison shapes differ")
        result = ttest_rel(reference_array, alternative_array)
        tests.append(
            (
                float(np.mean(reference_array - alternative_array)),
                float(result.statistic),
                float(result.pvalue),
            )
        )
    raw = np.asarray([item[2] for item in tests])
    holm = holm_bonferroni(raw)
    fdr = benjamini_hochberg(raw)
    return [
        PairedComparison(
            item[0], item[1], item[2], float(holm[index]), float(fdr[index])
        )
        for index, item in enumerate(tests)
    ]


def cohens_d_paired(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.asarray(first, dtype=np.float64).reshape(-1) - np.asarray(
        second, dtype=np.float64
    ).reshape(-1)
    deviation = float(np.std(difference, ddof=1))
    return float(np.mean(difference) / deviation) if deviation else float("nan")


def corrected_resampled_t_statistic(
    differences: np.ndarray, test_fraction: float, train_fraction: float
) -> float:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if values.size < 2 or test_fraction <= 0.0 or train_fraction <= 0.0:
        raise ValueError("corrected test inputs are invalid")
    variance = float(np.var(values, ddof=1))
    correction = 1.0 / values.size + test_fraction / train_fraction
    denominator = np.sqrt(correction * variance)
    return float(np.mean(values) / denominator) if denominator else float("nan")


def mcnemar_test(
    target: np.ndarray, first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    target_array = np.asarray(target).reshape(-1)
    first_array = np.asarray(first).reshape(-1)
    second_array = np.asarray(second).reshape(-1)
    if len({target_array.shape, first_array.shape, second_array.shape}) != 1:
        raise ValueError("McNemar arrays must share a shape")
    first_correct = first_array == target_array
    second_correct = second_array == target_array
    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    denominator = first_only + second_only
    if denominator == 0:
        return 0.0, 1.0
    statistic = (abs(first_only - second_only) - 1.0) ** 2 / denominator
    return float(statistic), float(2.0 * (1.0 - norm.cdf(np.sqrt(statistic))))
