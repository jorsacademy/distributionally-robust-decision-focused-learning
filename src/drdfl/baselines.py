"""Deterministic prediction and decision baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drdfl.dataset import ContextualDataset


@dataclass(frozen=True, slots=True)
class RidgePredictor:
    coefficients: np.ndarray
    context_mean: np.ndarray
    context_scale: np.ndarray

    def predict(self, contexts: np.ndarray) -> np.ndarray:
        values = np.asarray(contexts, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.context_mean.size:
            raise ValueError("context matrix has the wrong shape")
        standardized = (values - self.context_mean) / self.context_scale
        design = np.column_stack([np.ones(values.shape[0]), standardized])
        predictions = design @ self.coefficients
        if not np.all(np.isfinite(predictions)):
            raise RuntimeError("ridge baseline produced non-finite predictions")
        return predictions


def fit_ridge(dataset: ContextualDataset, *, regularization: float = 1e-2) -> RidgePredictor:
    if regularization < 0.0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and nonnegative")
    contexts = np.asarray([sample.context for sample in dataset.samples], dtype=float)
    costs = np.asarray([sample.realized_costs for sample in dataset.samples], dtype=float)
    context_mean = np.mean(contexts, axis=0)
    context_scale = np.std(contexts, axis=0)
    context_scale = np.maximum(context_scale, 1e-8)
    standardized = (contexts - context_mean) / context_scale
    design = np.column_stack([np.ones(contexts.shape[0]), standardized])
    penalty = regularization * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ costs)
    return RidgePredictor(coefficients, context_mean, context_scale)


def global_mean_predictions(
    train_dataset: ContextualDataset,
    target_dataset: ContextualDataset,
) -> np.ndarray:
    mean_cost = np.mean(
        np.asarray([sample.realized_costs for sample in train_dataset.samples], dtype=float),
        axis=0,
    )
    return np.repeat(mean_cost[None, :], len(target_dataset.samples), axis=0)
