from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .features import EXCLUDED_LABEL_FEATURES, canonical_feature_names
from .sampling import CohortArrays


@dataclass(frozen=True)
class FeatureAudit:
    name: str
    count: int
    missing: int
    missing_fraction: float
    mean: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class PhenotypeAudit:
    phenotype: int
    count: int
    positive_count: int
    negative_count: int
    positive_fraction: float


@dataclass(frozen=True)
class CohortAudit:
    sample_count: int
    feature_count: int
    duplicate_patients: int
    feature_audits: tuple[FeatureAudit, ...]
    phenotype_audits: tuple[PhenotypeAudit, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_feature(name: str, values: np.ndarray) -> FeatureAudit:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    missing = array.size - finite.size
    if finite.size == 0:
        return FeatureAudit(name, array.size, missing, 1.0, None, None, None, None)
    return FeatureAudit(
        name=name,
        count=array.size,
        missing=missing,
        missing_fraction=missing / array.size,
        mean=float(np.mean(finite)),
        standard_deviation=float(np.std(finite)),
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
    )


def audit_phenotype(
    value: int, phenotype: np.ndarray, target: np.ndarray
) -> PhenotypeAudit:
    members = phenotype == value
    count = int(np.sum(members))
    positive = int(np.sum(target[members] == 1))
    negative = int(np.sum(target[members] == 0))
    return PhenotypeAudit(
        value, count, positive, negative, positive / count if count else float("nan")
    )


def audit_cohort(cohort: CohortArrays) -> CohortAudit:
    cohort.validate()
    names = canonical_feature_names()
    if cohort.features.shape[1] != len(names):
        raise ValueError("cohort feature width differs from canonical schema")
    feature_audits = tuple(
        audit_feature(name, cohort.features[:, index])
        for index, name in enumerate(names)
    )
    phenotype_audits = tuple(
        audit_phenotype(int(value), cohort.phenotype, cohort.classification_target)
        for value in np.unique(cohort.phenotype)
    )
    duplicates = cohort.patient_id.size - np.unique(cohort.patient_id).size
    return CohortAudit(
        sample_count=cohort.features.shape[0],
        feature_count=cohort.features.shape[1],
        duplicate_patients=int(duplicates),
        feature_audits=feature_audits,
        phenotype_audits=phenotype_audits,
    )


def leakage_audit(frame: pd.DataFrame) -> tuple[str, ...]:
    lower_columns = {str(column).lower() for column in frame.columns}
    forbidden = tuple(
        name for name in EXCLUDED_LABEL_FEATURES if name.lower() in lower_columns
    )
    aliases = {
        "waist",
        "triglyceride",
        "hdl",
        "systolic_bp",
        "diastolic_bp",
        "fasting_blood_glucose",
    }
    return tuple(sorted(set(forbidden).union(lower_columns.intersection(aliases))))


def assert_no_leakage(frame: pd.DataFrame) -> None:
    detected = leakage_audit(frame)
    if detected:
        raise ValueError(f"label leakage variables detected: {detected}")


def missingness_by_phenotype(cohort: CohortArrays) -> dict[int, np.ndarray]:
    output = {}
    for phenotype in np.unique(cohort.phenotype):
        values = cohort.features[cohort.phenotype == phenotype]
        output[int(phenotype)] = np.mean(~np.isfinite(values), axis=0)
    return output


def distribution_shift(
    reference: CohortArrays, external: CohortArrays, epsilon: float = 1e-6
) -> np.ndarray:
    if reference.features.shape[1] != external.features.shape[1]:
        raise ValueError("cohorts have different feature spaces")
    reference_mean = np.nanmean(reference.features, axis=0)
    reference_std = np.nanstd(reference.features, axis=0)
    external_mean = np.nanmean(external.features, axis=0)
    return np.abs(external_mean - reference_mean) / (reference_std + epsilon)


def class_balance_weights(target: np.ndarray) -> np.ndarray:
    array = np.asarray(target, dtype=np.int64).reshape(-1)
    counts = np.bincount(array, minlength=2)
    if np.any(counts == 0):
        raise ValueError("both outcome classes are required")
    weights = array.size / (2.0 * counts)
    return weights[array]
