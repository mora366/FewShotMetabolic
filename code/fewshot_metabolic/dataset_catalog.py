from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetRelease:
    cohort: str
    release: str
    access: str
    role: str
    controlled: bool


@dataclass(frozen=True)
class NhanesCycle:
    label: str
    start_year: int
    end_year: int
    component_suffix: str


NHANES_CYCLES = (
    NhanesCycle("2007-2008", 2007, 2008, "E"),
    NhanesCycle("2009-2010", 2009, 2010, "F"),
    NhanesCycle("2011-2012", 2011, 2012, "G"),
    NhanesCycle("2013-2014", 2013, 2014, "H"),
    NhanesCycle("2015-2016", 2015, 2016, "I"),
    NhanesCycle("2017-2018", 2017, 2018, "J"),
)

DATASET_RELEASES = (
    DatasetRelease(
        "NHANES",
        "2007-2018",
        "dataset_urls.txt line 1",
        "training and internal validation",
        False,
    ),
    DatasetRelease(
        "UK Biobank",
        "not specified",
        "dataset_urls.txt line 2",
        "external validation",
        True,
    ),
    DatasetRelease(
        "CHARLS",
        "not specified",
        "dataset_urls.txt line 3",
        "external validation",
        True,
    ),
)

NHANES_COMPONENTS = (
    "DEMO",
    "BMX",
    "GHB",
    "INS",
    "TCHOL",
    "HDL",
    "TRIGLY",
    "CBC",
    "HSCRP",
    "PAQ",
    "BPX",
    "GLU",
)


def nhanes_component_filename(component: str, cycle: NhanesCycle) -> str:
    normalized = component.upper()
    if normalized not in NHANES_COMPONENTS:
        raise ValueError(f"unknown NHANES component: {component}")
    return f"{normalized}_{cycle.component_suffix}.XPT"


def nhanes_cycle_files(cycle: NhanesCycle) -> tuple[str, ...]:
    return tuple(
        nhanes_component_filename(component, cycle) for component in NHANES_COMPONENTS
    )


def validate_cycles(cycles: tuple[NhanesCycle, ...] = NHANES_CYCLES) -> None:
    if not cycles:
        raise ValueError("at least one NHANES cycle is required")
    ordered = sorted(cycles, key=lambda item: item.start_year)
    for index, cycle in enumerate(ordered):
        if cycle.end_year != cycle.start_year + 1:
            raise ValueError("NHANES cycles must span two calendar years")
        if index and cycle.start_year != ordered[index - 1].end_year + 1:
            raise ValueError("NHANES cycles must form a continuous sequence")
    suffixes = [cycle.component_suffix for cycle in cycles]
    if len(suffixes) != len(set(suffixes)):
        raise ValueError("NHANES component suffixes must be unique")


def controlled_cohorts() -> tuple[str, ...]:
    return tuple(release.cohort for release in DATASET_RELEASES if release.controlled)


def primary_cohort() -> DatasetRelease:
    matches = [release for release in DATASET_RELEASES if "training" in release.role]
    if len(matches) != 1:
        raise RuntimeError("dataset catalog must define one training cohort")
    return matches[0]


def external_cohorts() -> tuple[DatasetRelease, ...]:
    return tuple(
        release for release in DATASET_RELEASES if release.role == "external validation"
    )
