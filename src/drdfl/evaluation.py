"""Solver-grounded prediction, regret, tail-risk, and ambiguity evaluation."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from drdfl.ambiguity import ambiguity_profile, solve_kl_adversary
from drdfl.baselines import fit_ridge, global_mean_predictions
from drdfl.dataset import ContextualDataset
from drdfl.losses import conditional_value_at_risk, path_regrets
from drdfl.model import CostPredictor


@dataclass(frozen=True, slots=True)
class MethodMetrics:
    method: str
    scenario: str
    sample_count: int
    prediction_rmse: float
    prediction_mae: float
    realized_mean_regret: float
    realized_median_regret: float
    realized_p90_regret: float
    realized_cvar90_regret: float
    realized_cvar95_regret: float
    realized_max_regret: float
    realized_mean_relative_regret_percent: float
    realized_path_hit_rate: float
    conditional_mean_regret: float
    conditional_cvar90_regret: float
    conditional_path_hit_rate: float
    mean_candidate_cost: float
    mean_optimal_cost: float
    feasible_rate: float
    unique_path_count: int
    worst_regime_mean_regret: float
    mean_regret_ci95_low: float
    mean_regret_ci95_high: float
    selected_radius_robust_regret: float
    selected_radius_effective_sample_size: float
    selected_radius_max_weight: float
    ambiguity_profile: dict[str, dict[str, float | None]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    rows: tuple[MethodMetrics, ...]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "metadata": self.metadata,
        }


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    draws: int = 500,
) -> tuple[float, float]:
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, values.size, size=(draws, values.size))
    means = np.mean(values[samples], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _path_signatures(dataset: ContextualDataset, predictions: np.ndarray) -> tuple[tuple[int, ...], ...]:
    signatures: list[tuple[int, ...]] = []
    for row in predictions:
        signatures.append(dataset.graph.shortest_path(row).edge_indices)
    return tuple(signatures)


def _worst_regime_mean(dataset: ContextualDataset, regrets: np.ndarray) -> float:
    by_regime: dict[str, list[float]] = {}
    for sample, regret in zip(dataset.samples, regrets, strict=True):
        by_regime.setdefault(sample.regime, []).append(float(regret))
    return max(statistics.fmean(values) for values in by_regime.values())


def evaluate_predictions(
    method: str,
    dataset: ContextualDataset,
    predictions: np.ndarray,
    *,
    scenario: str,
    ambiguity_radii: tuple[float, ...] = (0.0, 0.05, 0.15, 0.30),
    selected_radius: float = 0.15,
    bootstrap_seed: int = 0,
    bootstrap_draws: int = 500,
) -> MethodMetrics:
    predicted = np.asarray(predictions, dtype=float)
    realized = np.asarray([sample.realized_costs for sample in dataset.samples], dtype=float)
    true_mean = np.asarray([sample.true_mean_costs for sample in dataset.samples], dtype=float)
    if predicted.shape != realized.shape:
        raise ValueError("prediction matrix does not match the dataset")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("prediction matrix must be finite")

    realized_regret, candidate_costs, optimal_costs = path_regrets(
        dataset.graph, predicted, realized
    )
    conditional_regret, _mean_candidate, _mean_optimal = path_regrets(
        dataset.graph, predicted, true_mean
    )
    relative = 100.0 * realized_regret / np.maximum(1.0, np.abs(optimal_costs))
    signatures = _path_signatures(dataset, predicted)
    realized_optimal_signatures = tuple(
        dataset.graph.shortest_path(row).edge_indices for row in realized
    )
    conditional_optimal_signatures = tuple(
        dataset.graph.shortest_path(row).edge_indices for row in true_mean
    )
    audits = [
        dataset.graph.audit(
            dataset.graph.shortest_path(predicted[index]).decision,
            realized[index],
        )
        for index in range(predicted.shape[0])
    ]
    if not all(audit.feasible for audit in audits):
        raise RuntimeError("shortest-path decision failed the independent feasibility audit")
    profile = ambiguity_profile(realized_regret, ambiguity_radii)
    selected = solve_kl_adversary(realized_regret, selected_radius)
    ci_low, ci_high = _bootstrap_mean_interval(
        realized_regret, seed=bootstrap_seed, draws=bootstrap_draws
    )
    return MethodMetrics(
        method=method,
        scenario=scenario,
        sample_count=len(dataset.samples),
        prediction_rmse=float(np.sqrt(np.mean((predicted - realized) ** 2))),
        prediction_mae=float(np.mean(np.abs(predicted - realized))),
        realized_mean_regret=float(np.mean(realized_regret)),
        realized_median_regret=float(np.median(realized_regret)),
        realized_p90_regret=float(np.quantile(realized_regret, 0.9)),
        realized_cvar90_regret=conditional_value_at_risk(realized_regret, 0.9),
        realized_cvar95_regret=conditional_value_at_risk(realized_regret, 0.95),
        realized_max_regret=float(np.max(realized_regret)),
        realized_mean_relative_regret_percent=float(np.mean(relative)),
        realized_path_hit_rate=float(
            np.mean(
                [
                    signature == optimum
                    for signature, optimum in zip(
                        signatures, realized_optimal_signatures, strict=True
                    )
                ]
            )
        ),
        conditional_mean_regret=float(np.mean(conditional_regret)),
        conditional_cvar90_regret=conditional_value_at_risk(conditional_regret, 0.9),
        conditional_path_hit_rate=float(
            np.mean(
                [
                    signature == optimum
                    for signature, optimum in zip(
                        signatures, conditional_optimal_signatures, strict=True
                    )
                ]
            )
        ),
        mean_candidate_cost=float(np.mean(candidate_costs)),
        mean_optimal_cost=float(np.mean(optimal_costs)),
        feasible_rate=float(np.mean([audit.feasible for audit in audits])),
        unique_path_count=len(set(signatures)),
        worst_regime_mean_regret=_worst_regime_mean(dataset, realized_regret),
        mean_regret_ci95_low=ci_low,
        mean_regret_ci95_high=ci_high,
        selected_radius_robust_regret=selected.value,
        selected_radius_effective_sample_size=selected.effective_sample_size,
        selected_radius_max_weight=selected.max_weight,
        ambiguity_profile=profile,
    )


def evaluate_models(
    models: dict[str, CostPredictor],
    train_dataset: ContextualDataset,
    test_dataset: ContextualDataset,
    *,
    scenario: str,
    ambiguity_radii: tuple[float, ...] = (0.0, 0.05, 0.15, 0.30),
    selected_radius: float = 0.15,
    bootstrap_seed: int = 0,
    bootstrap_draws: int = 500,
    include_baselines: bool = True,
) -> EvaluationReport:
    if train_dataset.graph != test_dataset.graph:
        raise ValueError("training and test datasets must share the graph")
    contexts = torch.tensor(
        [sample.context for sample in test_dataset.samples], dtype=torch.float32
    )
    prediction_map: dict[str, np.ndarray] = {}
    for name, model in models.items():
        if model.config.context_dim != test_dataset.context_dim:
            raise ValueError(f"model {name} has an incompatible context dimension")
        if model.config.edge_count != test_dataset.graph.edge_count:
            raise ValueError(f"model {name} has an incompatible edge dimension")
        model.eval()
        with torch.no_grad():
            prediction_map[name] = model(contexts.to(model.device)).cpu().double().numpy()

    if include_baselines:
        test_contexts = np.asarray([sample.context for sample in test_dataset.samples], dtype=float)
        ridge = fit_ridge(train_dataset)
        prediction_map["ridge_prediction"] = ridge.predict(test_contexts)
        prediction_map["global_mean"] = global_mean_predictions(train_dataset, test_dataset)
        prediction_map["true_mean_oracle"] = np.asarray(
            [sample.true_mean_costs for sample in test_dataset.samples], dtype=float
        )
        prediction_map["perfect_information_oracle"] = np.asarray(
            [sample.realized_costs for sample in test_dataset.samples], dtype=float
        )

    rows = tuple(
        evaluate_predictions(
            name,
            test_dataset,
            predictions,
            scenario=scenario,
            ambiguity_radii=ambiguity_radii,
            selected_radius=selected_radius,
            bootstrap_seed=bootstrap_seed + 10_007 * index,
            bootstrap_draws=bootstrap_draws,
        )
        for index, (name, predictions) in enumerate(sorted(prediction_map.items()))
    )
    return EvaluationReport(
        rows=rows,
        metadata={
            "scenario": scenario,
            "train_fingerprint": train_dataset.fingerprint,
            "test_fingerprint": test_dataset.fingerprint,
            "graph_fingerprint": test_dataset.graph.fingerprint,
            "ambiguity_radii": list(ambiguity_radii),
            "selected_radius": selected_radius,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_draws": bootstrap_draws,
        },
    )


def save_report_json(report: EvaluationReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_report_csv(report: EvaluationReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for row in report.rows:
        payload = row.to_dict()
        payload["ambiguity_profile"] = json.dumps(payload["ambiguity_profile"], sort_keys=True)
        rows.append(payload)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
