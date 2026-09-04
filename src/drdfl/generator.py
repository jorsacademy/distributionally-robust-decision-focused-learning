"""Deterministic contextual edge-cost generators and distribution shifts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from drdfl.domain import LayeredGraph

CostRegime = Literal[
    "in_distribution",
    "mild_stress",
    "mean_shift",
    "covariance_shift",
    "heavy_tail",
    "corridor_shift",
    "nonlinear_shift",
    "combined_shift",
]

SUPPORTED_REGIMES: tuple[CostRegime, ...] = (
    "in_distribution",
    "mild_stress",
    "mean_shift",
    "covariance_shift",
    "heavy_tail",
    "corridor_shift",
    "nonlinear_shift",
    "combined_shift",
)


@dataclass(frozen=True, slots=True)
class GeneratorSpec:
    graph: LayeredGraph
    context_dim: int = 8
    structure_seed: int = 2026
    noise_scale: float = 0.35

    def __post_init__(self) -> None:
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not math.isfinite(self.noise_scale) or self.noise_scale < 0.0:
            raise ValueError("noise_scale must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class GeneratedObservation:
    context: tuple[float, ...]
    true_mean_costs: tuple[float, ...]
    realized_costs: tuple[float, ...]
    regime: CostRegime
    seed: int


class ContextualCostGenerator:
    """A fixed nonlinear context-to-cost mechanism with controlled shifts."""

    def __init__(self, spec: GeneratorSpec) -> None:
        self.spec = spec
        rng = np.random.default_rng(spec.structure_seed)
        edge_count = spec.graph.edge_count
        context_dim = spec.context_dim
        self.base = rng.uniform(3.5, 8.0, size=edge_count)
        self.linear = rng.normal(0.0, 0.85, size=(edge_count, context_dim))
        self.nonlinear = rng.normal(0.0, 1.0, size=(edge_count, context_dim))
        self.factor_loading = rng.normal(0.0, 0.16, size=edge_count)
        first_layer = set(spec.graph.layer_nodes(0)[: max(1, spec.graph.width // 2)])
        self.corridor_mask = np.asarray(
            [edge.head in first_layer or edge.tail in first_layer for edge in spec.graph.edges],
            dtype=float,
        )

    def _context(self, regime: CostRegime, rng: np.random.Generator) -> np.ndarray:
        dimension = self.spec.context_dim
        standard = rng.normal(0.0, 1.0, size=dimension)
        if regime == "mean_shift":
            offset = np.resize(np.asarray([1.35, -0.85, 0.55, 0.0]), dimension)
            return standard + offset
        if regime == "covariance_shift":
            scales = np.resize(np.asarray([1.8, 0.55, 1.4, 0.7]), dimension)
            correlated = standard.copy()
            if dimension >= 2:
                correlated[1] = 0.75 * standard[0] + math.sqrt(1.0 - 0.75**2) * standard[1]
            return scales * correlated
        if regime == "combined_shift":
            scales = np.resize(np.asarray([1.7, 0.7, 1.5, 0.8]), dimension)
            offset = np.resize(np.asarray([1.0, -0.65, 0.4, 0.2]), dimension)
            shifted = scales * standard + offset
            if dimension >= 3:
                shifted[2] += 0.55 * shifted[0]
            return shifted
        if regime == "mild_stress":
            offset = np.resize(np.asarray([0.45, -0.25, 0.15, 0.0]), dimension)
            return 1.15 * standard + offset
        return standard

    def generate(self, *, regime: CostRegime = "in_distribution", seed: int) -> GeneratedObservation:
        if regime not in SUPPORTED_REGIMES:
            raise ValueError(f"unsupported cost regime: {regime}")
        rng = np.random.default_rng(seed)
        context = self._context(regime, rng)
        scale = math.sqrt(float(self.spec.context_dim))
        nonlinear_multiplier = 1.0
        nonlinear_phase = 0.0
        if regime == "nonlinear_shift":
            nonlinear_multiplier = 1.9
            nonlinear_phase = 0.65
        elif regime == "combined_shift":
            nonlinear_multiplier = 1.65
            nonlinear_phase = 0.45
        raw_mean = (
            self.base
            + (self.linear @ context) / scale
            + nonlinear_multiplier
            * 0.75
            * np.sin((self.nonlinear @ context) / scale + nonlinear_phase)
        )
        if regime == "corridor_shift":
            raw_mean = raw_mean + 2.8 * self.corridor_mask
        elif regime == "combined_shift":
            raw_mean = raw_mean + 2.0 * self.corridor_mask
        true_mean = np.maximum(raw_mean, 0.15)

        factor = float(rng.normal())
        idiosyncratic = rng.normal(size=self.spec.graph.edge_count)
        noise_scale = self.spec.noise_scale
        if regime == "mild_stress":
            noise_scale *= 1.35
        elif regime == "covariance_shift":
            noise_scale *= 1.9
        elif regime == "heavy_tail":
            factor = float(rng.standard_t(df=3))
            idiosyncratic = rng.standard_t(df=3, size=self.spec.graph.edge_count)
            noise_scale *= 1.25
        elif regime == "combined_shift":
            factor = float(rng.standard_t(df=4))
            idiosyncratic = rng.standard_t(df=4, size=self.spec.graph.edge_count)
            noise_scale *= 1.7
        noise = noise_scale * (0.55 * idiosyncratic + self.factor_loading * factor)
        realized = np.maximum(true_mean + noise, 0.05)
        return GeneratedObservation(
            context=tuple(float(value) for value in context),
            true_mean_costs=tuple(float(value) for value in true_mean),
            realized_costs=tuple(float(value) for value in realized),
            regime=regime,
            seed=seed,
        )
