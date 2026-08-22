from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import Phenotype


@dataclass(frozen=True)
class MetabolicCriteria:
    waist_high: np.ndarray
    triglycerides_high: np.ndarray
    hdl_low: np.ndarray
    blood_pressure_high: np.ndarray
    fasting_glucose_high: np.ndarray

    def count(self) -> np.ndarray:
        arrays = (
            self.waist_high,
            self.triglycerides_high,
            self.hdl_low,
            self.blood_pressure_high,
            self.fasting_glucose_high,
        )
        lengths = {array.shape for array in arrays}
        if len(lengths) != 1:
            raise ValueError("criteria arrays must share a shape")
        return np.stack(arrays, axis=1).astype(np.int64).sum(axis=1)


def metabolic_risk_label(criteria: MetabolicCriteria) -> np.ndarray:
    return (criteria.count() >= 2).astype(np.int64)


def assign_phenotype(
    bmi: np.ndarray,
    criteria: MetabolicCriteria,
    low_muscle_mass: np.ndarray,
    normal_bmi_upper: float = 25.0,
    obesity_threshold: float = 30.0,
) -> np.ndarray:
    if bmi.shape != low_muscle_mass.shape:
        raise ValueError("BMI and muscle status must share a shape")
    counts = criteria.count()
    healthy = counts <= 1
    obese = bmi >= obesity_threshold
    normal = bmi < normal_bmi_upper
    output = np.full(bmi.shape, int(Phenotype.MUNO), dtype=np.int64)
    output[healthy & ~obese] = int(Phenotype.MHNO)
    output[~healthy & ~obese] = int(Phenotype.MUNO)
    output[normal & ~healthy] = int(Phenotype.MONW)
    output[obese & healthy] = int(Phenotype.MHO)
    output[obese & ~healthy] = int(Phenotype.MUO)
    output[obese & low_muscle_mass.astype(bool)] = int(Phenotype.SO)
    return output
