from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from .schema import Episode, MetabolicBatch, Phenotype


@dataclass(frozen=True)
class CohortArrays:
    features: np.ndarray
    phenotype: np.ndarray
    classification_target: np.ndarray
    regression_target: np.ndarray
    patient_id: np.ndarray

    def validate(self) -> None:
        size = self.features.shape[0]
        arrays = (
            self.phenotype,
            self.classification_target,
            self.regression_target,
            self.patient_id,
        )
        if self.features.ndim != 2 or any(array.shape != (size,) for array in arrays):
            raise ValueError("cohort arrays have inconsistent shapes")
        if np.unique(self.patient_id).size != size:
            raise ValueError("patient identifiers must be unique")


class EpisodeSampler:
    def __init__(
        self,
        cohort: CohortArrays,
        support_size: int,
        query_size: int,
        seed: int = 42,
    ) -> None:
        cohort.validate()
        self.cohort = cohort
        self.support_size = support_size
        self.query_size = query_size
        self.generator = np.random.default_rng(seed)
        self.by_phenotype = {
            phenotype: np.flatnonzero(cohort.phenotype == int(phenotype))
            for phenotype in Phenotype
        }
        required = support_size + query_size
        sparse = [
            key.name
            for key, values in self.by_phenotype.items()
            if values.size < required
        ]
        if sparse:
            raise ValueError(f"phenotypes lack enough disjoint patients: {sparse}")

    def _batch(self, indices: np.ndarray) -> MetabolicBatch:
        return MetabolicBatch(
            features=torch.from_numpy(self.cohort.features[indices]).float(),
            phenotype=torch.from_numpy(self.cohort.phenotype[indices]).long(),
            classification_target=torch.from_numpy(
                self.cohort.classification_target[indices]
            ).long(),
            regression_target=torch.from_numpy(
                self.cohort.regression_target[indices]
            ).float(),
        )

    def sample(self, phenotype: Phenotype) -> Episode:
        pool = self.by_phenotype[phenotype]
        selected = self.generator.choice(
            pool,
            size=self.support_size + self.query_size,
            replace=False,
        )
        support_indices = selected[: self.support_size]
        query_indices = selected[self.support_size :]
        support_ids = set(self.cohort.patient_id[support_indices].tolist())
        query_ids = set(self.cohort.patient_id[query_indices].tolist())
        if support_ids.intersection(query_ids):
            raise RuntimeError("support and query patients overlap")
        return Episode(
            support=self._batch(support_indices),
            query=self._batch(query_indices),
            phenotype=phenotype,
        )

    def batches(self, count: int, meta_batch_size: int) -> Iterator[list[Episode]]:
        phenotypes = tuple(Phenotype)
        for _ in range(count):
            selected = self.generator.choice(
                phenotypes, size=meta_batch_size, replace=True
            )
            yield [self.sample(Phenotype(int(value))) for value in selected]


def stratified_patient_folds(
    phenotype: np.ndarray,
    target: np.ndarray,
    folds: int,
    seed: int,
) -> list[np.ndarray]:
    if phenotype.shape != target.shape:
        raise ValueError("stratification arrays must share a shape")
    generator = np.random.default_rng(seed)
    assignments = [[] for _ in range(folds)]
    strata = np.stack((phenotype, target), axis=1)
    for key in np.unique(strata, axis=0):
        members = np.flatnonzero(np.all(strata == key, axis=1))
        generator.shuffle(members)
        for index, member in enumerate(members):
            assignments[index % folds].append(int(member))
    return [np.asarray(sorted(values), dtype=np.int64) for values in assignments]
