from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .randomness import capture_random_state, restore_random_state, set_seed


@dataclass(frozen=True)
class CheckpointMetadata:
    iteration: int
    seed: int
    best_validation_loss: float


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: CheckpointMetadata,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": metadata.iteration,
        "seed": metadata.seed,
        "best_validation_loss": metadata.best_validation_loss,
        "random_state": capture_random_state(),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name, suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointMetadata:
    payload: dict[str, Any] = torch.load(
        Path(path), map_location=map_location, weights_only=False
    )
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    seed = int(payload["seed"])
    set_seed(seed)
    restore_random_state(payload["random_state"])
    return CheckpointMetadata(
        int(payload["iteration"]), seed, float(payload["best_validation_loss"])
    )
