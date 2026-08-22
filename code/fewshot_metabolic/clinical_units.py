from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnitConversion:
    feature: str
    source_unit: str
    target_unit: str
    scale: float
    offset: float = 0.0

    def apply(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        return array * self.scale + self.offset


CONVERSIONS = (
    UnitConversion("glucose", "mmol/L", "mg/dL", 18.0182),
    UnitConversion("total_cholesterol", "mmol/L", "mg/dL", 38.67),
    UnitConversion("ldl_c", "mmol/L", "mg/dL", 38.67),
    UnitConversion("hdl_c", "mmol/L", "mg/dL", 38.67),
    UnitConversion("triglycerides", "mmol/L", "mg/dL", 88.57),
    UnitConversion("crp", "mg/dL", "mg/L", 10.0),
    UnitConversion("fibrinogen", "g/L", "mg/dL", 100.0),
    UnitConversion("insulin", "pmol/L", "uU/mL", 1.0 / 6.0),
    UnitConversion("leptin", "ug/L", "ng/mL", 1.0),
    UnitConversion("adiponectin", "mg/L", "ug/mL", 1.0),
    UnitConversion("resistin", "ug/L", "ng/mL", 1.0),
    UnitConversion("hip_circumference", "m", "cm", 100.0),
    UnitConversion("body_fat_percent", "fraction", "%", 100.0),
)


def conversion(feature: str, source_unit: str, target_unit: str) -> UnitConversion:
    matches = [
        item
        for item in CONVERSIONS
        if item.feature == feature
        and item.source_unit == source_unit
        and item.target_unit == target_unit
    ]
    if len(matches) != 1:
        raise ValueError(
            f"unit conversion is undefined for {feature}: {source_unit} to {target_unit}"
        )
    return matches[0]


def convert(
    values: np.ndarray, feature: str, source_unit: str, target_unit: str
) -> np.ndarray:
    if source_unit == target_unit:
        return np.asarray(values, dtype=np.float64).copy()
    return conversion(feature, source_unit, target_unit).apply(values)


def glucose_mmol_to_mg_dl(values: np.ndarray) -> np.ndarray:
    return convert(values, "glucose", "mmol/L", "mg/dL")


def glucose_mg_dl_to_mmol(values: np.ndarray) -> np.ndarray:
    return (
        np.asarray(values, dtype=np.float64)
        / conversion("glucose", "mmol/L", "mg/dL").scale
    )


def cholesterol_mmol_to_mg_dl(values: np.ndarray) -> np.ndarray:
    return convert(values, "total_cholesterol", "mmol/L", "mg/dL")


def triglycerides_mmol_to_mg_dl(values: np.ndarray) -> np.ndarray:
    return convert(values, "triglycerides", "mmol/L", "mg/dL")


def crp_mg_dl_to_mg_l(values: np.ndarray) -> np.ndarray:
    return convert(values, "crp", "mg/dL", "mg/L")


def insulin_pmol_to_micro_u(values: np.ndarray) -> np.ndarray:
    return convert(values, "insulin", "pmol/L", "uU/mL")


def assert_plausible_conversion(
    source: np.ndarray, converted: np.ndarray, scale: float
) -> None:
    source_array = np.asarray(source, dtype=np.float64)
    converted_array = np.asarray(converted, dtype=np.float64)
    if source_array.shape != converted_array.shape:
        raise ValueError("unit conversion changed array shape")
    finite = np.isfinite(source_array) & np.isfinite(converted_array)
    if np.any(finite) and not np.allclose(
        converted_array[finite], source_array[finite] * scale
    ):
        raise ValueError("unit conversion does not match declared scale")


def unit_registry() -> dict[str, tuple[str, ...]]:
    output: dict[str, set[str]] = {}
    for item in CONVERSIONS:
        output.setdefault(item.feature, set()).update(
            (item.source_unit, item.target_unit)
        )
    return {feature: tuple(sorted(units)) for feature, units in output.items()}


def validate_registry() -> None:
    keys = [(item.feature, item.source_unit, item.target_unit) for item in CONVERSIONS]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate unit conversion definitions")
    for item in CONVERSIONS:
        if (
            item.scale <= 0.0
            or not np.isfinite(item.scale)
            or not np.isfinite(item.offset)
        ):
            raise ValueError("unit conversion parameters must be finite and positive")
