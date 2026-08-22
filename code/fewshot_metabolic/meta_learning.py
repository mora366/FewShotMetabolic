from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch.func import functional_call

from .losses import FewShotMetabolicLoss
from .model import FewShotMetabolic
from .schema import Episode, LossOutput, MetabolicBatch, ModelOutput


@dataclass(frozen=True)
class MetaStepResult:
    query_loss: float
    prediction_loss: float
    pathway_loss: float
    adapter_loss: float
    tasks: int


def _output_with_parameters(
    model: FewShotMetabolic,
    parameters: OrderedDict[str, torch.Tensor],
    batch: MetabolicBatch,
) -> ModelOutput:
    output = functional_call(model, parameters, (batch.features, batch.phenotype))
    if not isinstance(output, ModelOutput):
        raise TypeError("functional model returned an invalid output")
    return output


class MetaLearner:
    def __init__(
        self,
        model: FewShotMetabolic,
        criterion: FewShotMetabolicLoss,
        inner_learning_rate: float,
        outer_learning_rate: float,
        inner_steps: int,
        weight_decay: float,
        first_order: bool = False,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.inner_learning_rate = inner_learning_rate
        self.inner_steps = inner_steps
        self.first_order = first_order
        selected_ids = {id(parameter) for parameter in model.adapter_parameters()}
        self.adapter_names = {
            name
            for name, parameter in model.named_parameters()
            if id(parameter) in selected_ids
        }
        if not self.adapter_names:
            raise ValueError("model contains no adaptable parameters")
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=outer_learning_rate,
            weight_decay=weight_decay,
        )

    def adapt(
        self, support: MetabolicBatch, create_graph: bool | None = None
    ) -> OrderedDict[str, torch.Tensor]:
        graph = not self.first_order if create_graph is None else create_graph
        parameters = OrderedDict(self.model.named_parameters())
        for _ in range(self.inner_steps):
            output = _output_with_parameters(self.model, parameters, support)
            adapter_values = [parameters[name] for name in self.adapter_names]
            loss = self.criterion(output, support, adapter_values).total
            selected_names = [name for name in parameters if name in self.adapter_names]
            gradients = torch.autograd.grad(
                loss,
                [parameters[name] for name in selected_names],
                create_graph=graph,
                retain_graph=graph,
            )
            updates = dict(zip(selected_names, gradients, strict=True))
            parameters = OrderedDict(
                (
                    name,
                    (
                        value - self.inner_learning_rate * updates[name]
                        if name in updates
                        else value
                    ),
                )
                for name, value in parameters.items()
            )
        return parameters

    def query_loss(
        self, episode: Episode, parameters: OrderedDict[str, torch.Tensor]
    ) -> LossOutput:
        output = _output_with_parameters(self.model, parameters, episode.query)
        return self.criterion(
            output, episode.query, [parameters[name] for name in self.adapter_names]
        )

    def step(self, episodes: list[Episode]) -> MetaStepResult:
        if not episodes:
            raise ValueError("meta-batch cannot be empty")
        self.optimizer.zero_grad(set_to_none=True)
        losses = [
            self.query_loss(episode, self.adapt(episode.support))
            for episode in episodes
        ]
        total = torch.stack([item.total for item in losses]).sum()
        total.backward()
        self.optimizer.step()
        return MetaStepResult(
            float(total.detach()),
            float(torch.stack([item.prediction for item in losses]).mean().detach()),
            float(torch.stack([item.pathway for item in losses]).mean().detach()),
            float(torch.stack([item.adapter for item in losses]).mean().detach()),
            len(episodes),
        )

    def adapt_for_testing(
        self, support: MetabolicBatch
    ) -> OrderedDict[str, torch.Tensor]:
        return self.adapt(support, create_graph=False)
