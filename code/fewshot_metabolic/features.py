from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PathwayName(str, Enum):
    GLUCOSE = "glucose"
    LIPID = "lipid"
    INFLAMMATORY = "inflammatory"
    ADIPOKINE = "adipokine"
    ANTHROPOMETRIC = "anthropometric"
    DEMOGRAPHIC = "demographic"


@dataclass(frozen=True)
class FeatureDefinition:
    canonical_name: str
    pathway: PathwayName
    aliases: tuple[str, ...]
    unit: str


FEATURES = (
    FeatureDefinition("hba1c", PathwayName.GLUCOSE, ("glycohemoglobin", "a1c"), "%"),
    FeatureDefinition("insulin", PathwayName.GLUCOSE, ("fasting_insulin",), "uU/mL"),
    FeatureDefinition("homa_ir", PathwayName.GLUCOSE, ("homair",), "index"),
    FeatureDefinition("homa_beta", PathwayName.GLUCOSE, ("homab",), "index"),
    FeatureDefinition("total_cholesterol", PathwayName.LIPID, ("tc",), "mg/dL"),
    FeatureDefinition("ldl_c", PathwayName.LIPID, ("ldl",), "mg/dL"),
    FeatureDefinition("apob", PathwayName.LIPID, ("apolipoprotein_b",), "mg/dL"),
    FeatureDefinition("apoa1", PathwayName.LIPID, ("apolipoprotein_a1",), "mg/dL"),
    FeatureDefinition("crp", PathwayName.INFLAMMATORY, ("c_reactive_protein",), "mg/L"),
    FeatureDefinition("fibrinogen", PathwayName.INFLAMMATORY, (), "mg/dL"),
    FeatureDefinition(
        "wbc", PathwayName.INFLAMMATORY, ("white_blood_cell_count",), "10^9/L"
    ),
    FeatureDefinition("leptin", PathwayName.ADIPOKINE, (), "ng/mL"),
    FeatureDefinition("adiponectin", PathwayName.ADIPOKINE, (), "ug/mL"),
    FeatureDefinition("resistin", PathwayName.ADIPOKINE, (), "ng/mL"),
    FeatureDefinition("bmi", PathwayName.ANTHROPOMETRIC, ("body_mass_index",), "kg/m2"),
    FeatureDefinition("hip_circumference", PathwayName.ANTHROPOMETRIC, ("hip",), "cm"),
    FeatureDefinition(
        "body_fat_percent", PathwayName.ANTHROPOMETRIC, ("body_fat",), "%"
    ),
    FeatureDefinition("age", PathwayName.DEMOGRAPHIC, (), "years"),
    FeatureDefinition("sex", PathwayName.DEMOGRAPHIC, ("gender",), "category"),
    FeatureDefinition(
        "physical_activity", PathwayName.DEMOGRAPHIC, ("activity",), "category"
    ),
)

EXCLUDED_LABEL_FEATURES = (
    "waist_circumference",
    "triglycerides",
    "hdl_c",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "fasting_glucose",
)


def pathway_indices(names: tuple[str, ...]) -> dict[PathwayName, tuple[int, ...]]:
    aliases: dict[str, FeatureDefinition] = {}
    for definition in FEATURES:
        aliases[definition.canonical_name] = definition
        for alias in definition.aliases:
            aliases[alias] = definition
    groups: dict[PathwayName, list[int]] = {pathway: [] for pathway in PathwayName}
    for index, name in enumerate(names):
        if name in EXCLUDED_LABEL_FEATURES:
            raise ValueError(f"label-defining feature cannot be an input: {name}")
        if name not in aliases:
            raise ValueError(f"unknown metabolic feature: {name}")
        groups[aliases[name].pathway].append(index)
    if any(not values for values in groups.values()):
        raise ValueError("each metabolic pathway must contain at least one feature")
    return {key: tuple(values) for key, values in groups.items()}


def canonical_feature_names() -> tuple[str, ...]:
    return tuple(feature.canonical_name for feature in FEATURES)
