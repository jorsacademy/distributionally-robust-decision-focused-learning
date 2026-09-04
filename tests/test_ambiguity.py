from __future__ import annotations

import numpy as np
import pytest
import torch

from drdfl.ambiguity import ambiguity_profile, kl_dro_aggregate, solve_kl_adversary


def test_zero_radius_is_empirical_mean() -> None:
    result = solve_kl_adversary([1.0, 2.0, 4.0], 0.0)
    assert result.value == pytest.approx(7.0 / 3.0)
    assert result.weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert result.kl_divergence == 0.0


def test_robust_value_and_concentration_are_monotone() -> None:
    losses = np.asarray([0.0, 1.0, 3.0, 5.0])
    nominal = solve_kl_adversary(losses, 0.0)
    moderate = solve_kl_adversary(losses, 0.2)
    concentrated = solve_kl_adversary(losses, 10.0)
    assert nominal.value < moderate.value < concentrated.value
    assert concentrated.value == 5.0
    assert concentrated.concentration_boundary
    assert sum(concentrated.weights) == pytest.approx(1.0)


def test_interior_adversary_satisfies_primal_dual_and_radius() -> None:
    result = solve_kl_adversary([1.0, 2.0, 3.0, 6.0], 0.15)
    assert result.value == pytest.approx(result.dual_value, rel=1e-7)
    assert result.kl_divergence == pytest.approx(0.15, abs=1e-6)
    assert sum(result.weights) == pytest.approx(1.0)
    assert result.max_weight > 0.25
    assert result.effective_sample_size < 4.0


def test_custom_autograd_returns_danskin_weights() -> None:
    losses = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    robust = kl_dro_aggregate(losses, 0.2)
    robust.backward()
    expected = solve_kl_adversary([1.0, 2.0, 4.0], 0.2)
    assert losses.grad is not None
    assert losses.grad.detach().numpy() == pytest.approx(expected.weights)


def test_ambiguity_profile_and_invalid_input() -> None:
    profile = ambiguity_profile([0.0, 1.0, 2.0], (0.0, 0.1))
    assert set(profile) == {"0", "0.1"}
    with pytest.raises(ValueError):
        solve_kl_adversary([], 0.1)
    with pytest.raises(ValueError):
        solve_kl_adversary([1.0], -0.1)
