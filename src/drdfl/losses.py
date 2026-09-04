"""Structured decision losses and exact solver-oracle subgradients."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function

from drdfl.domain import LayeredGraph


class _SPOPlus(Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: object,
        predicted_costs: Tensor,
        true_costs: Tensor,
        true_decisions: Tensor,
        graph: LayeredGraph,
    ) -> Tensor:
        if predicted_costs.ndim != 2:
            raise ValueError("predicted costs must be a matrix")
        if true_costs.shape != predicted_costs.shape:
            raise ValueError("true and predicted cost matrices must have equal shape")
        if true_decisions.shape != predicted_costs.shape:
            raise ValueError("true decisions must match the cost matrix")
        if predicted_costs.shape[1] != graph.edge_count:
            raise ValueError("cost matrix edge dimension does not match the graph")
        transformed = 2.0 * predicted_costs.detach() - true_costs.detach()
        gradients = torch.empty_like(predicted_costs)
        losses = torch.empty(
            predicted_costs.shape[0],
            dtype=predicted_costs.dtype,
            device=predicted_costs.device,
        )
        for index in range(predicted_costs.shape[0]):
            oracle = graph.shortest_path(transformed[index].cpu().double().numpy())
            oracle_decision = torch.tensor(
                oracle.decision,
                dtype=predicted_costs.dtype,
                device=predicted_costs.device,
            )
            difference = true_decisions[index] - oracle_decision
            loss = torch.dot(transformed[index], difference)
            value = float(loss.cpu())
            if value < -1e-5:
                raise RuntimeError("SPO+ oracle produced a materially negative surrogate")
            losses[index] = torch.clamp(loss, min=0.0)
            gradients[index] = 2.0 * difference
        ctx.save_for_backward(gradients)  # type: ignore[attr-defined]
        return losses

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: object,
        grad_output: Tensor,
    ) -> tuple[Tensor, None, None, None]:
        (gradients,) = ctx.saved_tensors  # type: ignore[attr-defined]
        return grad_output.unsqueeze(-1) * gradients, None, None, None


def spo_plus_losses(
    predicted_costs: Tensor,
    true_costs: Tensor,
    true_decisions: Tensor,
    graph: LayeredGraph,
) -> Tensor:
    """Compute one SPO+ surrogate value per contextual observation."""

    return _SPOPlus.apply(predicted_costs, true_costs, true_decisions, graph)


def prediction_losses(predicted_costs: Tensor, true_costs: Tensor) -> Tensor:
    if predicted_costs.shape != true_costs.shape or predicted_costs.ndim != 2:
        raise ValueError("prediction loss expects equal two-dimensional cost tensors")
    return torch.mean((predicted_costs - true_costs) ** 2, dim=1)


def path_regrets(
    graph: LayeredGraph,
    predicted_costs: np.ndarray,
    true_costs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.asarray(predicted_costs, dtype=float)
    truths = np.asarray(true_costs, dtype=float)
    if predictions.shape != truths.shape or predictions.ndim != 2:
        raise ValueError("predicted and true costs must be equal matrices")
    if predictions.shape[1] != graph.edge_count:
        raise ValueError("cost matrices do not match graph edge count")
    regrets = np.zeros(predictions.shape[0], dtype=float)
    candidate_objectives = np.zeros(predictions.shape[0], dtype=float)
    optimal_objectives = np.zeros(predictions.shape[0], dtype=float)
    for index in range(predictions.shape[0]):
        candidate = graph.shortest_path(predictions[index])
        optimum = graph.shortest_path(truths[index])
        candidate_objective = graph.audit(candidate.decision, truths[index]).objective
        regret = candidate_objective - optimum.objective
        tolerance = 1e-8 * max(1.0, abs(candidate_objective), abs(optimum.objective))
        if regret < -tolerance:
            raise RuntimeError("candidate path outperformed the exact shortest-path oracle")
        regrets[index] = max(0.0, regret)
        candidate_objectives[index] = candidate_objective
        optimal_objectives[index] = optimum.objective
    return regrets, candidate_objectives, optimal_objectives


def conditional_value_at_risk(values: np.ndarray, alpha: float = 0.9) -> float:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("CVaR values must be a nonempty finite vector")
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    tail_count = max(1, math.ceil((1.0 - alpha) * data.size))
    return float(np.mean(np.sort(data)[-tail_count:]))
