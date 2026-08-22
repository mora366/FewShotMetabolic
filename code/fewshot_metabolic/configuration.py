from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    hidden_dim: int = 256
    layers: int = 4
    heads: int = 8
    feedforward_dim: int = 1024
    lora_rank: int = 8
    lora_alpha: float = 1.0
    dropout: float = 0.1
    phenotype_count: int = 6
    phenotype_embedding_dim: int = 32
    pathway_margin: float = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 50000
    meta_batch_size: int = 16
    support_size: int = 10
    query_size: int = 10
    inner_steps: int = 5
    inner_learning_rate: float = 0.01
    outer_learning_rate: float = 0.001
    weight_decay: float = 0.0001
    pathway_weight: float = 0.1
    adapter_weight: float = 0.0001
    early_stopping_patience: int = 5000
    seed: int = 42
    precision: str = "float32"
    scheduler: str = "constant"


@dataclass(frozen=True)
class EvaluationConfig:
    folds: int = 5
    sampling_trials: int = 20
    bootstrap_iterations: int = 1000
    confidence: float = 0.95
    calibration_bins: int = 15


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"configuration section {name} must be a mapping")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    model = ModelConfig(**_section(raw, "model"))
    training = TrainingConfig(**_section(raw, "training"))
    evaluation = EvaluationConfig(**_section(raw, "evaluation"))
    validate_config(model, training, evaluation)
    return ExperimentConfig(model=model, training=training, evaluation=evaluation)


def validate_config(
    model: ModelConfig, training: TrainingConfig, evaluation: EvaluationConfig
) -> None:
    if model.hidden_dim % model.heads != 0:
        raise ValueError("hidden dimension must be divisible by attention heads")
    if model.lora_rank <= 0 or model.lora_rank >= model.hidden_dim:
        raise ValueError("LoRA rank must be between zero and hidden dimension")
    if model.layers <= 0 or model.phenotype_count <= 1:
        raise ValueError("model depth and phenotype count must be positive")
    if (
        training.inner_steps <= 0
        or training.support_size <= 0
        or training.query_size <= 0
    ):
        raise ValueError("episodic settings must be positive")
    if training.inner_learning_rate <= 0.0 or training.outer_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if evaluation.folds < 2 or evaluation.sampling_trials <= 0:
        raise ValueError("evaluation requires repeated folds and trials")
