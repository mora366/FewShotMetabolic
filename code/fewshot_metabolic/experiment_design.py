from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from .configuration import ModelConfig, TrainingConfig
from .sampling import CohortArrays, stratified_patient_folds


@dataclass(frozen=True)
class FoldPartition:
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class FewShotTrial:
    fold: int
    trial: int
    phenotype: int
    shots: int
    support_indices: np.ndarray
    query_indices: np.ndarray


@dataclass(frozen=True)
class AblationSpec:
    name: str
    model: ModelConfig
    training: TrainingConfig
    disable_meta_learning: bool = False
    disable_bilevel: bool = False
    first_order: bool = False
    disable_pathways: bool = False
    disable_gating: bool = False
    full_fine_tuning: bool = False
    uniform_pathway_weights: bool = False
    disable_pathway_loss: bool = False
    disable_l2: bool = False
    focal_loss: bool = False


def cross_validation_partitions(
    cohort: CohortArrays, folds: int = 5, seed: int = 42
) -> list[FoldPartition]:
    assignments = stratified_patient_folds(
        cohort.phenotype, cohort.classification_target, folds, seed
    )
    universe = np.arange(cohort.features.shape[0])
    partitions = []
    for fold, validation in enumerate(assignments):
        mask = np.ones(universe.size, dtype=bool)
        mask[validation] = False
        partitions.append(FoldPartition(fold, universe[mask], validation))
    validate_partitions(partitions, universe.size)
    return partitions


def validate_partitions(partitions: list[FoldPartition], sample_count: int) -> None:
    validation_sets = [set(item.validation_indices.tolist()) for item in partitions]
    if set.union(*validation_sets) != set(range(sample_count)):
        raise ValueError("validation folds do not cover all patients")
    for first, second in combinations(validation_sets, 2):
        if first.intersection(second):
            raise ValueError("validation folds overlap")
    for partition in partitions:
        if set(partition.train_indices).intersection(partition.validation_indices):
            raise ValueError("training and validation patients overlap")


def few_shot_trials(
    cohort: CohortArrays,
    partition: FoldPartition,
    shots: int,
    trials: int = 20,
    seed: int = 42,
) -> list[FewShotTrial]:
    if shots not in {1, 5, 10, 20}:
        raise ValueError("few-shot protocol accepts 1, 5, 10, or 20 support samples")
    generator = np.random.default_rng(seed + partition.fold)
    output: list[FewShotTrial] = []
    for trial in range(trials):
        for phenotype in np.unique(cohort.phenotype):
            support_pool = partition.train_indices[
                cohort.phenotype[partition.train_indices] == phenotype
            ]
            query_pool = partition.validation_indices[
                cohort.phenotype[partition.validation_indices] == phenotype
            ]
            if support_pool.size < shots or query_pool.size == 0:
                raise ValueError("fold lacks patients for requested phenotype trial")
            support = generator.choice(support_pool, size=shots, replace=False)
            if set(cohort.patient_id[support]).intersection(
                cohort.patient_id[query_pool]
            ):
                raise RuntimeError("support and query patient pools overlap")
            output.append(
                FewShotTrial(
                    partition.fold,
                    trial,
                    int(phenotype),
                    shots,
                    np.sort(support),
                    np.sort(query_pool),
                )
            )
    return output


def ablation_registry(
    model: ModelConfig, training: TrainingConfig
) -> tuple[AblationSpec, ...]:
    return (
        AblationSpec("full", model, training),
        AblationSpec(
            "without_meta_learning", model, training, disable_meta_learning=True
        ),
        AblationSpec(
            "without_bilevel_optimization", model, training, disable_bilevel=True
        ),
        AblationSpec("first_order_approximation", model, training, first_order=True),
        AblationSpec("without_pathway_modules", model, training, disable_pathways=True),
        AblationSpec("without_gating", model, training, disable_gating=True),
        AblationSpec(
            "without_lora_full_fine_tuning", model, training, full_fine_tuning=True
        ),
        AblationSpec(
            "uniform_pathway_weights", model, training, uniform_pathway_weights=True
        ),
        AblationSpec("two_transformer_layers", replace(model, layers=2), training),
        AblationSpec("six_transformer_layers", replace(model, layers=6), training),
        AblationSpec(
            "hidden_dimension_128",
            replace(model, hidden_dim=128, feedforward_dim=512),
            training,
        ),
        AblationSpec(
            "hidden_dimension_512",
            replace(model, hidden_dim=512, feedforward_dim=2048),
            training,
        ),
        AblationSpec(
            "without_pathway_consistency",
            model,
            replace(training, pathway_weight=0.0),
            disable_pathway_loss=True,
        ),
        AblationSpec(
            "without_l2_regularization",
            model,
            replace(training, adapter_weight=0.0),
            disable_l2=True,
        ),
        AblationSpec("focal_loss", model, training, focal_loss=True),
    )


def rank_sensitivity_registry(
    model: ModelConfig, training: TrainingConfig
) -> tuple[AblationSpec, ...]:
    return tuple(
        AblationSpec(f"lora_rank_{rank}", replace(model, lora_rank=rank), training)
        for rank in (2, 4, 8, 16, 32)
    )


def interaction_registry(
    model: ModelConfig, training: TrainingConfig
) -> tuple[AblationSpec, ...]:
    return (
        AblationSpec(
            "without_meta_and_pathways",
            model,
            training,
            disable_meta_learning=True,
            disable_pathways=True,
        ),
        AblationSpec(
            "without_meta_and_gating",
            model,
            training,
            disable_meta_learning=True,
            disable_gating=True,
        ),
        AblationSpec(
            "without_pathways_and_gating",
            model,
            training,
            disable_pathways=True,
            disable_gating=True,
        ),
    )
