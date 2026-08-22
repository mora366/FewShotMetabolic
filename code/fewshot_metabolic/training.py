from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from .checkpoint import CheckpointMetadata, save_checkpoint
from .configuration import ExperimentConfig
from .losses import FewShotMetabolicLoss
from .meta_learning import MetaLearner, MetaStepResult
from .model import FewShotMetabolic
from .randomness import set_seed
from .sampling import EpisodeSampler

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingRecord:
    iteration: int
    query_loss: float
    prediction_loss: float
    pathway_loss: float
    adapter_loss: float


@dataclass(frozen=True)
class TrainingSummary:
    records: tuple[TrainingRecord, ...]
    best_iteration: int
    best_validation_loss: float
    stopped_early: bool


class EarlyStopping:
    def __init__(self, patience: int, minimum_delta: float = 0.0) -> None:
        if patience <= 0:
            raise ValueError("patience must be positive")
        self.patience = patience
        self.minimum_delta = minimum_delta
        self.best = float("inf")
        self.best_iteration = -1

    def update(self, value: float, iteration: int) -> bool:
        if value < self.best - self.minimum_delta:
            self.best = value
            self.best_iteration = iteration
            return False
        return iteration - self.best_iteration >= self.patience


class MetaTrainer:
    def __init__(
        self,
        model: FewShotMetabolic,
        config: ExperimentConfig,
        checkpoint_path: str | Path | None = None,
        first_order: bool = False,
    ) -> None:
        self.model = model
        self.config = config
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        criterion = FewShotMetabolicLoss(
            pathway_weight=config.training.pathway_weight,
            adapter_weight=config.training.adapter_weight,
            pathway_margin=config.model.pathway_margin,
        )
        self.learner = MetaLearner(
            model=model,
            criterion=criterion,
            inner_learning_rate=config.training.inner_learning_rate,
            outer_learning_rate=config.training.outer_learning_rate,
            inner_steps=config.training.inner_steps,
            weight_decay=config.training.weight_decay,
            first_order=first_order,
        )

    def train(
        self,
        sampler: EpisodeSampler,
        validation: EpisodeSampler | None = None,
        validation_interval: int = 100,
    ) -> TrainingSummary:
        set_seed(self.config.training.seed)
        stopper = EarlyStopping(
            getattr(self.config.training, "early_stopping_patience", 5000)
        )
        records: list[TrainingRecord] = []
        stopped = False
        iterator = sampler.batches(
            self.config.training.iterations, self.config.training.meta_batch_size
        )
        for iteration, episodes in enumerate(iterator, start=1):
            self.model.train()
            result = self.learner.step(episodes)
            records.append(self._record(iteration, result))
            if (
                iteration % validation_interval == 0
                or iteration == self.config.training.iterations
            ):
                validation_loss = self._validation_loss(validation or sampler)
                should_stop = stopper.update(validation_loss, iteration)
                self._checkpoint(iteration, stopper.best)
                LOGGER.info(
                    "iteration=%d train_loss=%.6f validation_loss=%.6f",
                    iteration,
                    result.query_loss,
                    validation_loss,
                )
                if should_stop:
                    stopped = True
                    break
        return TrainingSummary(
            tuple(records), stopper.best_iteration, stopper.best, stopped
        )

    def _record(self, iteration: int, result: MetaStepResult) -> TrainingRecord:
        return TrainingRecord(
            iteration,
            result.query_loss,
            result.prediction_loss,
            result.pathway_loss,
            result.adapter_loss,
        )

    def _validation_loss(self, sampler: EpisodeSampler) -> float:
        self.model.eval()
        losses = []
        for episodes in sampler.batches(1, self.config.training.meta_batch_size):
            for episode in episodes:
                parameters = self.learner.adapt_for_testing(episode.support)
                losses.append(
                    float(self.learner.query_loss(episode, parameters).total.detach())
                )
        return float(np.mean(losses))

    def _checkpoint(self, iteration: int, best_validation_loss: float) -> None:
        if self.checkpoint_path is None:
            return
        save_checkpoint(
            self.checkpoint_path,
            self.model,
            self.learner.optimizer,
            CheckpointMetadata(
                iteration, self.config.training.seed, best_validation_loss
            ),
        )


def parameter_counts(model: FewShotMetabolic) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    adaptable = sum(parameter.numel() for parameter in model.adapter_parameters())
    return {
        "total": total,
        "trainable": trainable,
        "adaptable": adaptable,
        "adaptable_percent": 100.0 * adaptable / total,
        "trainable_percent": 100.0 * trainable / total,
    }
