from __future__ import annotations

import numpy as np
import pytest
import torch

from drdfl.losses import conditional_value_at_risk, path_regrets, spo_plus_losses


def test_spo_plus_upper_bounds_actual_regret(tiny_train) -> None:
    _contexts, costs, decisions = tiny_train.tensors()
    predictions = costs + 0.8 * torch.randn_like(costs)
    surrogate = spo_plus_losses(predictions, costs, decisions, tiny_train.graph)
    regret, _candidate, _optimal = path_regrets(
        tiny_train.graph,
        predictions.detach().numpy(),
        costs.numpy(),
    )
    assert np.all(surrogate.detach().numpy() + 1e-6 >= regret)
    assert torch.all(surrogate >= 0.0)


def test_spo_plus_has_solver_oracle_subgradient(tiny_train) -> None:
    _contexts, costs, decisions = tiny_train.tensors()
    predictions = (costs + 0.5).clone().requires_grad_(True)
    losses = spo_plus_losses(predictions, costs, decisions, tiny_train.graph)
    torch.sum(losses).backward()
    assert predictions.grad is not None
    assert predictions.grad.shape == predictions.shape
    assert torch.all(torch.isfinite(predictions.grad))


def test_perfect_predictions_have_zero_regret(tiny_train) -> None:
    costs = np.asarray([sample.realized_costs for sample in tiny_train.samples])
    regret, candidate, optimum = path_regrets(tiny_train.graph, costs, costs)
    assert np.allclose(regret, 0.0)
    assert np.allclose(candidate, optimum)


def test_cvar_uses_upper_tail() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 10.0])
    assert conditional_value_at_risk(values, 0.75) == 10.0
    with pytest.raises(ValueError):
        conditional_value_at_risk(values, 1.0)
