"""Context-to-edge-cost predictors and safe checkpoint persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

CHECKPOINT_SCHEMA_VERSION = "1.0"
FEATURE_SCHEMA_VERSION = "context-cost-v1"


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    context_dim: int
    edge_count: int
    hidden_dim: int = 96
    hidden_layers: int = 2

    def __post_init__(self) -> None:
        if self.context_dim <= 0 or self.edge_count <= 0:
            raise ValueError("context_dim and edge_count must be positive")
        if self.hidden_dim <= 0 or self.hidden_layers <= 0:
            raise ValueError("hidden dimensions must be positive")


class CostPredictor(nn.Module):
    """A compact nonlinear predictor shared by all training objectives."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        self.config = config
        modules: list[nn.Module] = []
        input_dim = config.context_dim
        for _ in range(config.hidden_layers):
            modules.extend([nn.Linear(input_dim, config.hidden_dim), nn.SiLU()])
            input_dim = config.hidden_dim
        modules.append(nn.Linear(input_dim, config.edge_count))
        self.network = nn.Sequential(*modules)
        self.register_buffer("context_mean", torch.zeros(config.context_dim))
        self.register_buffer("context_scale", torch.ones(config.context_dim))
        self.register_buffer("cost_mean", torch.zeros(config.edge_count))
        self.register_buffer("cost_scale", torch.ones(config.edge_count))
        self.register_buffer("normalizer_fitted", torch.tensor(False))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def fit_normalizers(self, contexts: Tensor, costs: Tensor) -> None:
        if contexts.ndim != 2 or contexts.shape[1] != self.config.context_dim:
            raise ValueError("context matrix has the wrong shape")
        if costs.ndim != 2 or costs.shape[1] != self.config.edge_count:
            raise ValueError("cost matrix has the wrong shape")
        if contexts.shape[0] != costs.shape[0] or contexts.shape[0] == 0:
            raise ValueError("normalizer data must be aligned and nonempty")
        context_mean = torch.mean(contexts, dim=0)
        context_scale = torch.std(contexts, dim=0, unbiased=False).clamp_min(1e-6)
        cost_mean = torch.mean(costs, dim=0)
        cost_scale = torch.std(costs, dim=0, unbiased=False).clamp_min(1e-6)
        self.context_mean.copy_(context_mean.detach().to(self.context_mean.device))
        self.context_scale.copy_(context_scale.detach().to(self.context_scale.device))
        self.cost_mean.copy_(cost_mean.detach().to(self.cost_mean.device))
        self.cost_scale.copy_(cost_scale.detach().to(self.cost_scale.device))
        self.normalizer_fitted.copy_(torch.tensor(True, device=self.normalizer_fitted.device))

    def forward(self, contexts: Tensor) -> Tensor:
        if contexts.ndim != 2 or contexts.shape[1] != self.config.context_dim:
            raise ValueError("context tensor has the wrong shape")
        if not bool(self.normalizer_fitted):
            raise RuntimeError("predictor normalizers must be fitted before use")
        normalized = (contexts - self.context_mean) / self.context_scale
        prediction = self.network(normalized)
        costs = prediction * self.cost_scale + self.cost_mean
        if not torch.all(torch.isfinite(costs)):
            raise RuntimeError("predictor produced non-finite edge costs")
        return costs


def clone_model(model: CostPredictor) -> CostPredictor:
    clone = CostPredictor(model.config)
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.to(model.device)
    return clone


def save_checkpoint(
    model: CostPredictor,
    path: str | Path,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_config": json.dumps(asdict(model.config), sort_keys=True),
        "metadata": json.dumps(metadata or {}, sort_keys=True),
    }
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(tensors, str(output), metadata=header)


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[CostPredictor, dict[str, object]]:
    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        header = handle.metadata()
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    if header is None:
        raise ValueError("checkpoint metadata is missing")
    if header.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if header.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint feature schema is incompatible")
    raw_config = json.loads(header["model_config"])
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint model configuration is invalid")
    config = PredictorConfig(
        context_dim=int(raw_config["context_dim"]),
        edge_count=int(raw_config["edge_count"]),
        hidden_dim=int(raw_config["hidden_dim"]),
        hidden_layers=int(raw_config["hidden_layers"]),
    )
    model = CostPredictor(config)
    model.load_state_dict(tensors, strict=True)
    model.to(device)
    raw_metadata = json.loads(header.get("metadata", "{}"))
    if not isinstance(raw_metadata, dict):
        raise ValueError("checkpoint metadata payload is invalid")
    return model, cast(dict[str, object], raw_metadata)
