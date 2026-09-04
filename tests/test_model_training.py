from __future__ import annotations

from pathlib import Path

import pytest
import torch

from drdfl.model import CostPredictor, PredictorConfig, load_checkpoint, save_checkpoint
from drdfl.training import TrainingConfig, train_model


def _model(tiny_train, hidden_dim: int = 12) -> CostPredictor:
    return CostPredictor(
        PredictorConfig(
            context_dim=tiny_train.context_dim,
            edge_count=tiny_train.graph.edge_count,
            hidden_dim=hidden_dim,
            hidden_layers=1,
        )
    )


def test_model_requires_normalizers_and_then_predicts(tiny_train) -> None:
    model = _model(tiny_train)
    contexts, costs, _decisions = tiny_train.tensors()
    with pytest.raises(RuntimeError):
        model(contexts)
    model.fit_normalizers(contexts, costs)
    output = model(contexts)
    assert output.shape == costs.shape
    assert torch.all(torch.isfinite(output))


@pytest.mark.parametrize("mode", ["mse_erm", "mse_kl_dro", "spo_erm", "spo_kl_dro"])
def test_all_training_modes_complete(mode, tiny_train, tiny_validation) -> None:
    model = _model(tiny_train)
    summary = train_model(
        model,
        tiny_train,
        tiny_validation,
        config=TrainingConfig(
            mode=mode,
            epochs=2,
            warm_start_epochs=1,
            validation_every=1,
            patience_checks=2,
            radius_warmup_epochs=1,
            seed=9,
        ),
    )
    assert summary.epochs_completed == 2
    assert summary.history
    assert summary.best_validation_score >= 0.0


def test_checkpoint_round_trip(tiny_train, tiny_validation, tmp_path: Path) -> None:
    model = _model(tiny_train)
    train_model(
        model,
        tiny_train,
        tiny_validation,
        config=TrainingConfig(mode="mse_erm", epochs=1, validation_every=1, seed=3),
    )
    path = tmp_path / "model.safetensors"
    save_checkpoint(model, path, metadata={"training_mode": "mse_erm"})
    loaded, metadata = load_checkpoint(path)
    contexts, _costs, _decisions = tiny_validation.tensors()
    assert torch.allclose(model(contexts), loaded(contexts))
    assert metadata["training_mode"] == "mse_erm"
