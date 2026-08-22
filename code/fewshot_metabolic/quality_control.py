from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sampling import CohortArrays


@dataclass(frozen=True)
class RangeRule:
    feature: str
    lower: float
    upper: float


@dataclass(frozen=True)
class RangeViolation:
    feature: str
    below: int
    above: int
    nonfinite: int


@dataclass(frozen=True)
class SplitAudit:
    train_count: int
    validation_count: int
    overlapping_patients: int
    train_phenotypes: tuple[int, ...]
    validation_phenotypes: tuple[int, ...]


CLINICAL_RANGE_RULES = (
    RangeRule("hba1c", 2.0, 25.0),
    RangeRule("insulin", 0.0, 1000.0),
    RangeRule("homa_ir", 0.0, 500.0),
    RangeRule("homa_beta", -1000.0, 10000.0),
    RangeRule("total_cholesterol", 20.0, 1000.0),
    RangeRule("ldl_c", 0.0, 800.0),
    RangeRule("apob", 0.0, 500.0),
    RangeRule("apoa1", 0.0, 500.0),
    RangeRule("crp", 0.0, 500.0),
    RangeRule("fibrinogen", 0.0, 2000.0),
    RangeRule("wbc", 0.0, 200.0),
    RangeRule("leptin", 0.0, 1000.0),
    RangeRule("adiponectin", 0.0, 1000.0),
    RangeRule("resistin", 0.0, 1000.0),
    RangeRule("bmi", 8.0, 100.0),
    RangeRule("hip_circumference", 30.0, 250.0),
    RangeRule("body_fat_percent", 0.0, 80.0),
    RangeRule("age", 20.0, 120.0),
    RangeRule("sex", 0.0, 2.0),
    RangeRule("physical_activity", 0.0, 10.0),
)


def range_violations(
    features: np.ndarray, names: tuple[str, ...]
) -> tuple[RangeViolation, ...]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("feature matrix and names have incompatible shapes")
    rules = {rule.feature: rule for rule in CLINICAL_RANGE_RULES}
    output = []
    for index, name in enumerate(names):
        if name not in rules:
            raise ValueError(f"clinical range rule missing for {name}")
        values = matrix[:, index]
        finite = np.isfinite(values)
        rule = rules[name]
        output.append(
            RangeViolation(
                feature=name,
                below=int(np.sum(finite & (values < rule.lower))),
                above=int(np.sum(finite & (values > rule.upper))),
                nonfinite=int(np.sum(~finite)),
            )
        )
    return tuple(output)


def assert_binary_target(target: np.ndarray) -> None:
    values = np.asarray(target).reshape(-1)
    if values.size == 0 or not np.all(np.isin(values, (0, 1))):
        raise ValueError("classification target must contain binary values")


def assert_phenotype_codes(phenotype: np.ndarray) -> None:
    values = np.asarray(phenotype).reshape(-1)
    if values.size == 0 or not np.all(np.isin(values, np.arange(6))):
        raise ValueError("phenotype codes must range from zero through five")


def assert_finite_regression_target(target: np.ndarray) -> None:
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("regression target must be finite")


def assert_unique_patients(patient_id: np.ndarray) -> None:
    values = np.asarray(patient_id).reshape(-1)
    if values.size != np.unique(values).size:
        raise ValueError("patient identifiers are duplicated")


def audit_split(
    cohort: CohortArrays, train_indices: np.ndarray, validation_indices: np.ndarray
) -> SplitAudit:
    train = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    validation = np.asarray(validation_indices, dtype=np.int64).reshape(-1)
    if np.any(train < 0) or np.any(validation < 0):
        raise ValueError("split indices cannot be negative")
    if np.any(train >= cohort.features.shape[0]) or np.any(
        validation >= cohort.features.shape[0]
    ):
        raise ValueError("split index exceeds cohort bounds")
    train_patients = set(cohort.patient_id[train].tolist())
    validation_patients = set(cohort.patient_id[validation].tolist())
    return SplitAudit(
        train_count=train.size,
        validation_count=validation.size,
        overlapping_patients=len(train_patients.intersection(validation_patients)),
        train_phenotypes=tuple(
            sorted(int(value) for value in np.unique(cohort.phenotype[train]))
        ),
        validation_phenotypes=tuple(
            sorted(int(value) for value in np.unique(cohort.phenotype[validation]))
        ),
    )


def validate_cohort(cohort: CohortArrays) -> None:
    cohort.validate()
    assert_binary_target(cohort.classification_target)
    assert_phenotype_codes(cohort.phenotype)
    assert_finite_regression_target(cohort.regression_target)
    assert_unique_patients(cohort.patient_id)


def missingness_pattern(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features)
    if matrix.ndim != 2:
        raise ValueError("missingness analysis requires a matrix")
    patterns, counts = np.unique(~np.isfinite(matrix), axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    return patterns[order], counts[order]


def feature_correlations(features: np.ndarray, minimum_pairs: int = 10) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("correlation analysis requires a matrix")
    count = matrix.shape[1]
    output = np.eye(count, dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            valid = np.isfinite(matrix[:, first]) & np.isfinite(matrix[:, second])
            if np.sum(valid) < minimum_pairs:
                value = np.nan
            else:
                value = float(
                    np.corrcoef(matrix[valid, first], matrix[valid, second])[0, 1]
                )
            output[first, second] = value
            output[second, first] = value
    return output


def near_duplicate_rows(
    features: np.ndarray, tolerance: float = 1e-8
) -> list[tuple[int, int]]:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("duplicate analysis requires a matrix")
    output = []
    for first in range(matrix.shape[0]):
        for second in range(first + 1, matrix.shape[0]):
            valid = np.isfinite(matrix[first]) & np.isfinite(matrix[second])
            if np.any(valid) and np.all(
                np.abs(matrix[first, valid] - matrix[second, valid]) <= tolerance
            ):
                output.append((first, second))
    return output
