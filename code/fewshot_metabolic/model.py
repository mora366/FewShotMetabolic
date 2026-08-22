from __future__ import annotations

import torch
from torch import nn

from .configuration import ModelConfig
from .features import PathwayName
from .pathways import PathwayAdapter, PathwayGate
from .schema import ModelOutput
from .transformer import HierarchicalTransformer


class FewShotMetabolic(nn.Module):
    def __init__(
        self, config: ModelConfig, pathway_indices: dict[PathwayName, tuple[int, ...]]
    ) -> None:
        super().__init__()
        self.config = config
        self.pathway_names = tuple(PathwayName)
        if set(pathway_indices) != set(self.pathway_names):
            raise ValueError("all six pathway definitions are required")
        self.register_buffer(
            "feature_mean",
            torch.zeros(config.input_dim),
        )
        self.register_buffer(
            "feature_standard_deviation",
            torch.ones(config.input_dim),
        )
        self.index_names: list[str] = []
        for pathway in self.pathway_names:
            name = f"indices_{pathway.value}"
            self.register_buffer(
                name, torch.tensor(pathway_indices[pathway], dtype=torch.long)
            )
            self.index_names.append(name)
        self.adapters = nn.ModuleList(
            PathwayAdapter(
                len(pathway_indices[pathway]),
                config.hidden_dim,
                config.lora_rank,
                config.lora_alpha,
            )
            for pathway in self.pathway_names
        )
        self.gate = PathwayGate(
            config.phenotype_count,
            config.phenotype_embedding_dim,
            len(self.pathway_names),
        )
        self.backbone = HierarchicalTransformer(
            config.hidden_dim,
            config.layers,
            config.heads,
            config.feedforward_dim,
            config.dropout,
        )
        self.classification_head = nn.Linear(config.hidden_dim, 2)
        self.regression_head = nn.Linear(config.hidden_dim, 2)

    def set_normalization(
        self, mean: torch.Tensor, standard_deviation: torch.Tensor
    ) -> None:
        if (
            mean.shape != (self.config.input_dim,)
            or standard_deviation.shape != mean.shape
        ):
            raise ValueError("normalization vectors have incorrect shape")
        self.feature_mean.copy_(mean)
        self.feature_standard_deviation.copy_(standard_deviation)

    def normalized_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean) / (self.feature_standard_deviation + 1e-6)

    def split_pathways(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            features.index_select(1, getattr(self, name)) for name in self.index_names
        )

    def forward(self, features: torch.Tensor, phenotype: torch.Tensor) -> ModelOutput:
        if features.ndim != 2 or features.shape[1] != self.config.input_dim:
            raise ValueError("feature tensor has incorrect shape")
        if phenotype.shape != (features.shape[0],):
            raise ValueError("phenotype tensor has incorrect shape")
        normalized = self.normalized_features(features)
        pathway_inputs = self.split_pathways(normalized)
        activations = self.gate(phenotype, pathway_inputs)
        representations = tuple(
            adapter(values, activations[:, index])
            for index, (adapter, values) in enumerate(
                zip(self.adapters, pathway_inputs, strict=True)
            )
        )
        tokens = torch.stack(representations, dim=1)
        contextual = self.backbone(tokens)
        pooled = contextual.mean(dim=1)
        classification_logits = self.classification_head(pooled)
        regression = self.regression_head(pooled)
        return ModelOutput(
            classification_logits=classification_logits,
            glycemic_mean=regression[:, 0],
            glycemic_log_variance=regression[:, 1].clamp(-10.0, 10.0),
            pathway_representations=representations,
            gate_activations=activations.squeeze(-1),
        )

    def adapter_parameters(self) -> list[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        for adapter in self.adapters:
            parameters.extend(adapter.projection.adapter_parameters())
        return parameters

    def shared_parameters(self) -> list[nn.Parameter]:
        adapter_ids = {id(parameter) for parameter in self.adapter_parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in adapter_ids
        ]
