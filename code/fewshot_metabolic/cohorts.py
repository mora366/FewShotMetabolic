from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, canonical_feature_names
from .sampling import CohortArrays


@dataclass(frozen=True)
class ColumnBinding:
    canonical: str
    source_candidates: tuple[str, ...]
    multiplier: float = 1.0


NHANES_BINDINGS = (
    ColumnBinding("hba1c", ("LBXGH", "GHB")),
    ColumnBinding("insulin", ("LBXIN", "LBDINSI")),
    ColumnBinding("homa_ir", ("HOMA_IR",)),
    ColumnBinding("homa_beta", ("HOMA_BETA",)),
    ColumnBinding("total_cholesterol", ("LBXTC", "LBDTCSI")),
    ColumnBinding("ldl_c", ("LBDLDL", "LBDLDLSI")),
    ColumnBinding("apob", ("LBXAPB", "APOB")),
    ColumnBinding("apoa1", ("LBXAPA", "APOA1")),
    ColumnBinding("crp", ("LBXCRP", "LBXHSCRP")),
    ColumnBinding("fibrinogen", ("LBXFIB", "FIBRINOGEN")),
    ColumnBinding("wbc", ("LBXWBCSI", "LBXWBC")),
    ColumnBinding("leptin", ("LBXLEP", "LEPTIN")),
    ColumnBinding("adiponectin", ("LBXADP", "ADIPONECTIN")),
    ColumnBinding("resistin", ("LBXRES", "RESISTIN")),
    ColumnBinding("bmi", ("BMXBMI",)),
    ColumnBinding("hip_circumference", ("BMXHIP",)),
    ColumnBinding("body_fat_percent", ("DXDTOPF", "BMXBODYFAT")),
    ColumnBinding("age", ("RIDAGEYR",)),
    ColumnBinding("sex", ("RIAGENDR",)),
    ColumnBinding("physical_activity", ("PAQ650", "PAQ605")),
)

UK_BIOBANK_BINDINGS = (
    ColumnBinding("hba1c", ("30750-0.0", "hba1c")),
    ColumnBinding("insulin", ("insulin", "fasting_insulin")),
    ColumnBinding("homa_ir", ("homa_ir",)),
    ColumnBinding("homa_beta", ("homa_beta",)),
    ColumnBinding("total_cholesterol", ("30690-0.0", "cholesterol")),
    ColumnBinding("ldl_c", ("30780-0.0", "ldl_direct")),
    ColumnBinding("apob", ("30640-0.0", "apolipoprotein_b")),
    ColumnBinding("apoa1", ("30630-0.0", "apolipoprotein_a")),
    ColumnBinding("crp", ("30710-0.0", "c_reactive_protein")),
    ColumnBinding("fibrinogen", ("30010-0.0", "fibrinogen")),
    ColumnBinding("wbc", ("30000-0.0", "white_blood_cell_count")),
    ColumnBinding("leptin", ("leptin",)),
    ColumnBinding("adiponectin", ("adiponectin",)),
    ColumnBinding("resistin", ("resistin",)),
    ColumnBinding("bmi", ("21001-0.0", "bmi")),
    ColumnBinding("hip_circumference", ("49-0.0", "hip_circumference")),
    ColumnBinding("body_fat_percent", ("23099-0.0", "body_fat_percentage")),
    ColumnBinding("age", ("21022-0.0", "age")),
    ColumnBinding("sex", ("31-0.0", "sex")),
    ColumnBinding("physical_activity", ("22032-0.0", "physical_activity")),
)

CHARLS_BINDINGS = (
    ColumnBinding("hba1c", ("hba1c", "bl_hba1c")),
    ColumnBinding("insulin", ("insulin", "bl_insulin")),
    ColumnBinding("homa_ir", ("homa_ir",)),
    ColumnBinding("homa_beta", ("homa_beta",)),
    ColumnBinding("total_cholesterol", ("tc", "total_cholesterol")),
    ColumnBinding("ldl_c", ("ldl", "ldl_c")),
    ColumnBinding("apob", ("apob",)),
    ColumnBinding("apoa1", ("apoa1",)),
    ColumnBinding("crp", ("crp", "hs_crp")),
    ColumnBinding("fibrinogen", ("fibrinogen",)),
    ColumnBinding("wbc", ("wbc", "white_blood_cell_count")),
    ColumnBinding("leptin", ("leptin",)),
    ColumnBinding("adiponectin", ("adiponectin",)),
    ColumnBinding("resistin", ("resistin",)),
    ColumnBinding("bmi", ("bmi",)),
    ColumnBinding("hip_circumference", ("hip", "hip_circumference")),
    ColumnBinding("body_fat_percent", ("body_fat_percent",)),
    ColumnBinding("age", ("age",)),
    ColumnBinding("sex", ("gender", "sex")),
    ColumnBinding("physical_activity", ("physical_activity", "activity")),
)


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix in {".xpt", ".sas7bdat"}:
        return pd.read_sas(source)
    raise ValueError(f"unsupported cohort table format: {suffix}")


def bind_columns(
    frame: pd.DataFrame, bindings: tuple[ColumnBinding, ...]
) -> pd.DataFrame:
    output: dict[str, pd.Series] = {}
    for binding in bindings:
        selected = next(
            (name for name in binding.source_candidates if name in frame.columns), None
        )
        if selected is None:
            output[binding.canonical] = pd.Series(
                np.nan, index=frame.index, dtype=np.float64
            )
        else:
            output[binding.canonical] = (
                pd.to_numeric(frame[selected], errors="coerce") * binding.multiplier
            )
    bound = pd.DataFrame(output, index=frame.index)
    if tuple(bound.columns) != canonical_feature_names():
        raise RuntimeError("cohort binding order differs from model feature order")
    return bound


def cohort_from_frame(
    frame: pd.DataFrame,
    bindings: tuple[ColumnBinding, ...],
    phenotype_column: str,
    classification_column: str,
    regression_column: str,
    patient_column: str,
) -> CohortArrays:
    missing = {
        phenotype_column,
        classification_column,
        regression_column,
        patient_column,
    }.difference(frame.columns)
    if missing:
        raise ValueError(f"cohort outcome columns missing: {sorted(missing)}")
    features = bind_columns(frame, bindings).to_numpy(dtype=np.float32)
    cohort = CohortArrays(
        features=features,
        phenotype=frame[phenotype_column].to_numpy(dtype=np.int64),
        classification_target=frame[classification_column].to_numpy(dtype=np.int64),
        regression_target=frame[regression_column].to_numpy(dtype=np.float32),
        patient_id=frame[patient_column].to_numpy(),
    )
    cohort.validate()
    return cohort


def validate_feature_coverage(
    frame: pd.DataFrame, bindings: tuple[ColumnBinding, ...]
) -> dict[str, float]:
    bound = bind_columns(frame, bindings)
    return {
        column: float(1.0 - bound[column].isna().mean()) for column in bound.columns
    }


def validate_missingness(
    coverage: dict[str, float], maximum_missing: float = 0.5
) -> None:
    failures = {
        name: 1.0 - value
        for name, value in coverage.items()
        if 1.0 - value > maximum_missing
    }
    if failures:
        raise ValueError(f"features exceed missingness limit: {failures}")


def binding_for_cohort(name: str) -> tuple[ColumnBinding, ...]:
    normalized = name.lower().replace(" ", "_")
    if normalized == "nhanes":
        return NHANES_BINDINGS
    if normalized in {"uk_biobank", "ukb"}:
        return UK_BIOBANK_BINDINGS
    if normalized == "charls":
        return CHARLS_BINDINGS
    raise ValueError(f"unknown cohort: {name}")


def feature_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [item.canonical_name for item in FEATURES],
            "pathway": [item.pathway.value for item in FEATURES],
            "unit": [item.unit for item in FEATURES],
            "aliases": ["|".join(item.aliases) for item in FEATURES],
        }
    )
