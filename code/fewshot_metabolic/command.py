from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from .cohorts import binding_for_cohort, cohort_from_frame, read_table
from .configuration import load_config
from .data_audit import audit_cohort
from .evaluation import evaluate_model
from .features import canonical_feature_names, pathway_indices
from .losses import FewShotMetabolicLoss
from .meta_learning import MetaLearner
from .model import FewShotMetabolic
from .reporting import write_json
from .sampling import EpisodeSampler
from .schema import MetabolicBatch

LOGGER = logging.getLogger(__name__)


def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fewshot-metabolic")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--cohort", choices=("nhanes", "uk_biobank", "charls"), required=True
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--phenotype-column", default="phenotype")
    parser.add_argument("--classification-column", default="metabolic_risk")
    parser.add_argument("--regression-column", default="glycemic_response")
    parser.add_argument("--patient-column", default="patient_id")
    return parser


def load_command_cohort(arguments: argparse.Namespace):
    frame = read_table(arguments.data)
    bindings = binding_for_cohort(arguments.cohort)
    return cohort_from_frame(
        frame,
        bindings,
        arguments.phenotype_column,
        arguments.classification_column,
        arguments.regression_column,
        arguments.patient_column,
    )


def build_model(config_path: Path) -> FewShotMetabolic:
    config = load_config(config_path)
    return FewShotMetabolic(config.model, pathway_indices(canonical_feature_names()))


def audit_main(arguments: list[str] | None = None) -> int:
    parser = common_parser()
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    cohort = load_command_cohort(parsed)
    write_json(parsed.output, audit_cohort(cohort).to_dict())
    return 0


def train_main(arguments: list[str] | None = None) -> int:
    parser = common_parser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parsed = parser.parse_args(arguments)
    config = load_config(parsed.config)
    cohort = load_command_cohort(parsed)
    model = FewShotMetabolic(
        config.model, pathway_indices(canonical_feature_names())
    ).to(parsed.device)
    sampler = EpisodeSampler(
        cohort,
        config.training.support_size,
        config.training.query_size,
        config.training.seed,
    )
    criterion = FewShotMetabolicLoss(
        config.training.pathway_weight,
        config.training.adapter_weight,
        config.model.pathway_margin,
    )
    learner = MetaLearner(
        model,
        criterion,
        config.training.inner_learning_rate,
        config.training.outer_learning_rate,
        config.training.inner_steps,
        config.training.weight_decay,
    )
    history = []
    for iteration, episodes in enumerate(
        sampler.batches(config.training.iterations, config.training.meta_batch_size),
        start=1,
    ):
        moved = [
            type(episode)(
                episode.support.to(parsed.device),
                episode.query.to(parsed.device),
                episode.phenotype,
            )
            for episode in episodes
        ]
        result = learner.step(moved)
        history.append(
            {
                "iteration": iteration,
                "query_loss": result.query_loss,
                "prediction_loss": result.prediction_loss,
                "pathway_loss": result.pathway_loss,
                "adapter_loss": result.adapter_loss,
            }
        )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "seed": config.training.seed}, parsed.output
    )
    write_json(
        parsed.output.with_suffix(".json"), {"status": "PASS", "history": history}
    )
    return 0


def evaluation_batches(cohort, batch_size: int) -> list[MetabolicBatch]:
    batches = []
    for start in range(0, cohort.features.shape[0], batch_size):
        stop = min(start + batch_size, cohort.features.shape[0])
        indices = np.arange(start, stop)
        batches.append(
            MetabolicBatch(
                features=torch.from_numpy(cohort.features[indices]).float(),
                phenotype=torch.from_numpy(cohort.phenotype[indices]).long(),
                classification_target=torch.from_numpy(
                    cohort.classification_target[indices]
                ).long(),
                regression_target=torch.from_numpy(
                    cohort.regression_target[indices]
                ).float(),
            )
        )
    return batches


def evaluate_main(arguments: list[str] | None = None) -> int:
    parser = common_parser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parsed = parser.parse_args(arguments)
    config = load_config(parsed.config)
    cohort = load_command_cohort(parsed)
    model = FewShotMetabolic(
        config.model, pathway_indices(canonical_feature_names())
    ).to(parsed.device)
    payload = torch.load(parsed.weights, map_location=parsed.device, weights_only=False)
    model.load_state_dict(payload["model"])
    result = evaluate_model(
        model,
        evaluation_batches(cohort, parsed.batch_size),
        config.evaluation.calibration_bins,
        parsed.device,
    )
    write_json(parsed.output, result.to_dict())
    return 0


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fewshot-metabolic")
    parser.add_argument("command", choices=("audit", "train", "evaluate"))
    known, remaining = parser.parse_known_args(arguments)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if known.command == "audit":
        return audit_main(remaining)
    if known.command == "train":
        return train_main(remaining)
    return evaluate_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
