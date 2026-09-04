"""Exact KL-divergence empirical-distribution adversaries."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function


@dataclass(frozen=True, slots=True)
class KLDROResult:
    value: float
    weights: tuple[float, ...]
    radius: float
    kl_divergence: float
    eta: float | None
    dual_value: float
    effective_sample_size: float
    entropy: float
    max_weight: float
    concentration_boundary: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_inputs(
    losses: np.ndarray,
    radius: float,
    nominal: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("losses must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("losses must be finite")
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("KL radius must be finite and nonnegative")
    if nominal is None:
        probabilities = np.full(values.size, 1.0 / values.size, dtype=float)
    else:
        probabilities = np.asarray(nominal, dtype=float)
        if probabilities.shape != values.shape:
            raise ValueError("nominal probabilities must match losses")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0.0):
            raise ValueError("nominal probabilities must be finite and strictly positive")
        probabilities = probabilities / float(np.sum(probabilities))
    return values, probabilities


def _tilted_distribution(
    losses: np.ndarray,
    nominal: np.ndarray,
    eta: float,
) -> tuple[np.ndarray, float, float]:
    if eta <= 0.0 or not math.isfinite(eta):
        raise ValueError("eta must be finite and positive")
    logits = np.log(nominal) + losses / eta
    maximum = float(np.max(logits))
    exponentials = np.exp(logits - maximum)
    normalizer = float(np.sum(exponentials))
    weights = exponentials / normalizer
    log_normalizer = maximum + math.log(normalizer)
    log_ratio = np.log(weights) - np.log(nominal)
    divergence = float(np.dot(weights, log_ratio))
    return weights, divergence, log_normalizer


def solve_kl_adversary(
    losses: np.ndarray | tuple[float, ...] | list[float],
    radius: float,
    *,
    nominal: np.ndarray | None = None,
    tolerance: float = 1e-10,
    maximum_iterations: int = 200,
) -> KLDROResult:
    """Maximize expected loss in a forward-KL ball around a nominal law.

    The interior solution is an exponential tilt. A one-dimensional bisection finds
    the dual temperature whose KL divergence equals the requested radius.
    """

    values, probabilities = _validate_inputs(np.asarray(losses, dtype=float), radius, nominal)
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")

    if radius <= tolerance or float(np.ptp(values)) <= tolerance:
        weights = probabilities
        value = float(np.dot(weights, values))
        entropy = float(-np.dot(weights, np.log(weights)))
        return KLDROResult(
            value=value,
            weights=tuple(float(weight) for weight in weights),
            radius=radius,
            kl_divergence=0.0,
            eta=None,
            dual_value=value,
            effective_sample_size=1.0 / float(np.dot(weights, weights)),
            entropy=entropy,
            max_weight=float(np.max(weights)),
            concentration_boundary=False,
        )

    maximum_loss = float(np.max(values))
    maximizers = np.isclose(values, maximum_loss, atol=tolerance, rtol=0.0)
    maximum_mass = float(np.sum(probabilities[maximizers]))
    concentration_radius = -math.log(maximum_mass)
    if radius >= concentration_radius - tolerance:
        weights = np.zeros_like(probabilities)
        weights[maximizers] = probabilities[maximizers] / maximum_mass
        divergence = float(
            np.dot(weights[maximizers], np.log(weights[maximizers] / probabilities[maximizers]))
        )
        entropy = float(-np.dot(weights[maximizers], np.log(weights[maximizers])))
        return KLDROResult(
            value=maximum_loss,
            weights=tuple(float(weight) for weight in weights),
            radius=radius,
            kl_divergence=divergence,
            eta=0.0,
            dual_value=maximum_loss,
            effective_sample_size=1.0 / float(np.dot(weights, weights)),
            entropy=entropy,
            max_weight=float(np.max(weights)),
            concentration_boundary=True,
        )

    spread = max(float(np.ptp(values)), 1.0)
    lower = np.finfo(float).eps * spread
    upper = spread
    _, upper_divergence, _ = _tilted_distribution(values, probabilities, upper)
    expansion_count = 0
    while upper_divergence > radius:
        upper *= 2.0
        expansion_count += 1
        if expansion_count > 200 or not math.isfinite(upper):
            raise RuntimeError("failed to bracket the KL dual temperature")
        _, upper_divergence, _ = _tilted_distribution(values, probabilities, upper)

    for _ in range(maximum_iterations):
        midpoint = 0.5 * (lower + upper)
        _, divergence, _ = _tilted_distribution(values, probabilities, midpoint)
        if divergence > radius:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance * max(1.0, upper):
            break

    eta = upper
    weights, divergence, log_normalizer = _tilted_distribution(values, probabilities, eta)
    value = float(np.dot(weights, values))
    dual_value = eta * radius + eta * log_normalizer
    scale = max(1.0, abs(value), abs(dual_value))
    if abs(value - dual_value) > 5e-8 * scale:
        raise RuntimeError("KL adversary failed primal-dual consistency")
    if abs(divergence - radius) > 5e-7 * max(1.0, radius):
        raise RuntimeError("KL adversary did not activate the requested radius")
    entropy = float(-np.dot(weights, np.log(weights)))
    return KLDROResult(
        value=value,
        weights=tuple(float(weight) for weight in weights),
        radius=radius,
        kl_divergence=divergence,
        eta=eta,
        dual_value=dual_value,
        effective_sample_size=1.0 / float(np.dot(weights, weights)),
        entropy=entropy,
        max_weight=float(np.max(weights)),
        concentration_boundary=False,
    )


class _KLDROAggregate(Function):
    @staticmethod
    def forward(
        ctx: Any,
        losses: Tensor,
        radius: float,
    ) -> Tensor:
        if losses.ndim != 1:
            raise ValueError("KL-DRO aggregation expects a one-dimensional loss tensor")
        result = solve_kl_adversary(losses.detach().cpu().double().numpy(), radius)
        weights = torch.tensor(result.weights, dtype=losses.dtype, device=losses.device)
        ctx.save_for_backward(weights)
        return torch.dot(losses, weights)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> tuple[Tensor, None]:
        weights = cast(Tensor, ctx.saved_tensors[0])
        return grad_output * weights, None


def kl_dro_aggregate(losses: Tensor, radius: float) -> Tensor:
    """Return the exact KL-robust empirical risk with a Danskin gradient."""

    return cast(Tensor, _KLDROAggregate.apply(losses, radius))


def ambiguity_profile(
    losses: np.ndarray | tuple[float, ...] | list[float],
    radii: tuple[float, ...],
) -> dict[str, dict[str, float | None]]:
    if not radii:
        raise ValueError("at least one ambiguity radius is required")
    profile: dict[str, dict[str, float | None]] = {}
    for radius in radii:
        result = solve_kl_adversary(losses, radius)
        profile[f"{radius:.6g}"] = {
            "robust_value": result.value,
            "effective_sample_size": result.effective_sample_size,
            "max_weight": result.max_weight,
            "eta": result.eta,
            "kl_divergence": result.kl_divergence,
        }
    return profile
