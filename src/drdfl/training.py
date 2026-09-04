"""Prediction-focused and distributionally robust decision-focused training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from drdfl.ambiguity import kl_dro_aggregate, solve_kl_adversary
from drdfl.dataset import ContextualDataset
from drdfl.losses import (
    conditional_value_at_risk,
    path_regrets,
    prediction_losses,
    spo_plus_losses,
)
from drdfl.model import CostPredictor
from drdfl.utils import set_global_seed

TrainingMode = Literal["mse_erm", "mse_kl_dro", "spo_erm", "spo_kl_dro"]
SUPPORTED_MODES: tuple[TrainingMode, ...] = (
    "mse_erm",
    "mse_kl_dro",
    "spo_erm",
    "spo_kl_dro",
)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    mode: TrainingMode = "spo_kl_dro"
    epochs: int = 120
    warm_start_epochs: int = 20
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    kl_radius: float = 0.15
    radius_warmup_epochs: int = 20
    prediction_weight: float = 0.01
    tail_weight: float = 0.25
    validation_every: int = 5
    patience_checks: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"unsupported training mode: {self.mode}")
        counts = (
            self.epochs,
            self.validation_every,
            self.patience_checks,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("training counts must be positive")
        if self.warm_start_epochs < 0 or self.radius_warmup_epochs < 0:
            raise ValueError("warm-start counts must be nonnegative")
        positives = (self.learning_rate, self.gradient_clip_norm)
        if any(not math.isfinite(value) or value <= 0.0 for value in positives):
            raise ValueError("learning rate and gradient clip must be finite and positive")
        nonnegative = (
            self.weight_decay,
            self.kl_radius,
            self.prediction_weight,
            self.tail_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("regularization and robustness parameters must be nonnegative")


@dataclass(frozen=True, slots=True)
class TrainingPoint:
    stage: str
    epoch: int
    mean_task_loss: float
    robust_task_loss: float
    prediction_mse: float
    validation_mean_regret: float
    validation_cvar90_regret: float
    validation_score: float
    active_kl_radius: float
    adversary_effective_sample_size: float
    adversary_max_weight: float
    gradient_norm: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    mode: str
    epochs_completed: int
    warm_start_epochs_completed: int
    best_validation_score: float
    history: tuple[TrainingPoint, ...]
    config: TrainingConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "epochs_completed": self.epochs_completed,
            "warm_start_epochs_completed": self.warm_start_epochs_completed,
            "best_validation_score": self.best_validation_score,
            "history": [point.to_dict() for point in self.history],
            "config": asdict(self.config),
        }


def _clone_state(model: CostPredictor) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _validation_metrics(
    model: CostPredictor,
    dataset: ContextualDataset,
    tail_weight: float,
) -> tuple[float, float, float]:
    contexts, costs, _decisions = dataset.tensors(device=model.device)
    model.eval()
    with torch.no_grad():
        predictions = model(contexts).cpu().double().numpy()
    regrets, _candidate, _optimal = path_regrets(
        dataset.graph,
        predictions,
        costs.cpu().double().numpy(),
    )
    mean_regret = float(np.mean(regrets))
    cvar90 = conditional_value_at_risk(regrets, 0.9)
    return mean_regret, cvar90, mean_regret + tail_weight * cvar90


def _task_losses(
    model: CostPredictor,
    contexts: Tensor,
    costs: Tensor,
    decisions: Tensor,
    dataset: ContextualDataset,
    mode: TrainingMode,
) -> tuple[Tensor, Tensor]:
    predictions = model(contexts)
    mse = prediction_losses(predictions, costs)
    if mode.startswith("mse"):
        return mse, mse
    return spo_plus_losses(predictions, costs, decisions, dataset.graph), mse


def _aggregate(losses: Tensor, mode: TrainingMode, radius: float) -> Tensor:
    if mode.endswith("kl_dro"):
        return kl_dro_aggregate(losses, radius)
    return torch.mean(losses)


def _run_epoch(
    model: CostPredictor,
    optimizer: torch.optim.Optimizer,
    contexts: Tensor,
    costs: Tensor,
    decisions: Tensor,
    dataset: ContextualDataset,
    mode: TrainingMode,
    radius: float,
    prediction_weight: float,
    gradient_clip_norm: float,
) -> tuple[float, float, float, float, float, float]:
    model.train()
    task_losses, mse_losses = _task_losses(model, contexts, costs, decisions, dataset, mode)
    robust_loss = _aggregate(task_losses, mode, radius)
    objective = robust_loss
    if mode.startswith("spo") and prediction_weight > 0.0:
        objective = objective + prediction_weight * torch.mean(mse_losses)
    if not torch.isfinite(objective):
        raise RuntimeError("training objective became non-finite")
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(objective)
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    )
    if not math.isfinite(gradient_norm):
        raise RuntimeError("gradient norm became non-finite")
    optimizer.step()

    detached = task_losses.detach().cpu().double().numpy()
    diagnostics = solve_kl_adversary(detached, radius if mode.endswith("kl_dro") else 0.0)
    return (
        float(np.mean(detached)),
        float(robust_loss.detach().cpu()),
        float(torch.mean(mse_losses).detach().cpu()),
        diagnostics.effective_sample_size,
        diagnostics.max_weight,
        gradient_norm,
    )


def train_model(
    model: CostPredictor,
    train_dataset: ContextualDataset,
    validation_dataset: ContextualDataset,
    *,
    config: TrainingConfig | None = None,
) -> TrainingSummary:
    config = config or TrainingConfig()
    if train_dataset.graph != validation_dataset.graph:
        raise ValueError("training and validation datasets must use the same graph")
    if train_dataset.context_dim != validation_dataset.context_dim:
        raise ValueError("training and validation context dimensions must match")
    if model.config.context_dim != train_dataset.context_dim:
        raise ValueError("model context dimension does not match the dataset")
    if model.config.edge_count != train_dataset.graph.edge_count:
        raise ValueError("model edge dimension does not match the dataset")

    set_global_seed(config.seed)
    contexts, costs, decisions = train_dataset.tensors(device=model.device)
    model.fit_normalizers(contexts, costs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = _clone_state(model)
    best_validation = float("inf")
    checks_without_improvement = 0
    history: list[TrainingPoint] = []
    warm_completed = 0

    if config.mode.startswith("spo"):
        for epoch in range(1, config.warm_start_epochs + 1):
            metrics = _run_epoch(
                model,
                optimizer,
                contexts,
                costs,
                decisions,
                train_dataset,
                "mse_erm",
                0.0,
                0.0,
                config.gradient_clip_norm,
            )
            warm_completed = epoch
            if epoch == config.warm_start_epochs:
                validation = _validation_metrics(model, validation_dataset, config.tail_weight)
                history.append(
                    TrainingPoint(
                        stage="mse_warm_start",
                        epoch=epoch,
                        mean_task_loss=metrics[0],
                        robust_task_loss=metrics[1],
                        prediction_mse=metrics[2],
                        validation_mean_regret=validation[0],
                        validation_cvar90_regret=validation[1],
                        validation_score=validation[2],
                        active_kl_radius=0.0,
                        adversary_effective_sample_size=metrics[3],
                        adversary_max_weight=metrics[4],
                        gradient_norm=metrics[5],
                    )
                )

    epochs_completed = 0
    for epoch in range(1, config.epochs + 1):
        if config.mode.endswith("kl_dro") and config.radius_warmup_epochs > 0:
            active_radius = config.kl_radius * min(
                1.0, epoch / float(config.radius_warmup_epochs)
            )
        else:
            active_radius = config.kl_radius if config.mode.endswith("kl_dro") else 0.0
        metrics = _run_epoch(
            model,
            optimizer,
            contexts,
            costs,
            decisions,
            train_dataset,
            config.mode,
            active_radius,
            config.prediction_weight,
            config.gradient_clip_norm,
        )
        epochs_completed = epoch
        if epoch % config.validation_every != 0 and epoch != config.epochs:
            continue
        validation = _validation_metrics(model, validation_dataset, config.tail_weight)
        history.append(
            TrainingPoint(
                stage="main",
                epoch=epoch,
                mean_task_loss=metrics[0],
                robust_task_loss=metrics[1],
                prediction_mse=metrics[2],
                validation_mean_regret=validation[0],
                validation_cvar90_regret=validation[1],
                validation_score=validation[2],
                active_kl_radius=active_radius,
                adversary_effective_sample_size=metrics[3],
                adversary_max_weight=metrics[4],
                gradient_norm=metrics[5],
            )
        )
        if validation[2] < best_validation - 1e-10:
            best_validation = validation[2]
            best_state = _clone_state(model)
            checks_without_improvement = 0
        else:
            checks_without_improvement += 1
            if checks_without_improvement >= config.patience_checks:
                break

    if not math.isfinite(best_validation):
        validation = _validation_metrics(model, validation_dataset, config.tail_weight)
        best_validation = validation[2]
        best_state = _clone_state(model)
    model.load_state_dict(best_state, strict=True)
    return TrainingSummary(
        mode=config.mode,
        epochs_completed=epochs_completed,
        warm_start_epochs_completed=warm_completed,
        best_validation_score=best_validation,
        history=tuple(history),
        config=config,
    )
