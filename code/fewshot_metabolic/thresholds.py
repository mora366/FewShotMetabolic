from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phenotyping import MetabolicCriteria


@dataclass(frozen=True)
class HarmonizedThresholds:
    waist_male_cm: float
    waist_female_cm: float
    triglycerides_mg_dl: float
    hdl_male_mg_dl: float
    hdl_female_mg_dl: float
    systolic_mm_hg: float
    diastolic_mm_hg: float
    fasting_glucose_mg_dl: float


DEFAULT_THRESHOLDS = HarmonizedThresholds(
    waist_male_cm=102.0,
    waist_female_cm=88.0,
    triglycerides_mg_dl=150.0,
    hdl_male_mg_dl=40.0,
    hdl_female_mg_dl=50.0,
    systolic_mm_hg=130.0,
    diastolic_mm_hg=85.0,
    fasting_glucose_mg_dl=100.0,
)


def harmonized_criteria(
    waist_cm: np.ndarray,
    triglycerides_mg_dl: np.ndarray,
    hdl_mg_dl: np.ndarray,
    systolic_mm_hg: np.ndarray,
    diastolic_mm_hg: np.ndarray,
    fasting_glucose_mg_dl: np.ndarray,
    female: np.ndarray,
    thresholds: HarmonizedThresholds = DEFAULT_THRESHOLDS,
) -> MetabolicCriteria:
    arrays = [
        np.asarray(value).reshape(-1)
        for value in (
            waist_cm,
            triglycerides_mg_dl,
            hdl_mg_dl,
            systolic_mm_hg,
            diastolic_mm_hg,
            fasting_glucose_mg_dl,
            female,
        )
    ]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("harmonized criteria arrays must share a shape")
    waist_threshold = np.where(
        arrays[6].astype(bool),
        thresholds.waist_female_cm,
        thresholds.waist_male_cm,
    )
    hdl_threshold = np.where(
        arrays[6].astype(bool),
        thresholds.hdl_female_mg_dl,
        thresholds.hdl_male_mg_dl,
    )
    return MetabolicCriteria(
        waist_high=arrays[0] >= waist_threshold,
        triglycerides_high=arrays[1] >= thresholds.triglycerides_mg_dl,
        hdl_low=arrays[2] < hdl_threshold,
        blood_pressure_high=(arrays[3] >= thresholds.systolic_mm_hg)
        | (arrays[4] >= thresholds.diastolic_mm_hg),
        fasting_glucose_high=arrays[5] >= thresholds.fasting_glucose_mg_dl,
    )


def validate_thresholds(thresholds: HarmonizedThresholds) -> None:
    values = np.asarray(
        [
            thresholds.waist_male_cm,
            thresholds.waist_female_cm,
            thresholds.triglycerides_mg_dl,
            thresholds.hdl_male_mg_dl,
            thresholds.hdl_female_mg_dl,
            thresholds.systolic_mm_hg,
            thresholds.diastolic_mm_hg,
            thresholds.fasting_glucose_mg_dl,
        ]
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("harmonized thresholds must be finite and positive")
