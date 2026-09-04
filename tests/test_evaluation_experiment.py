from __future__ import annotations

from pathlib import Path

from drdfl.evaluation import evaluate_models, save_report_csv, save_report_json
from drdfl.experiment import ResearchConfig, run_research_experiment
from drdfl.model import CostPredictor, PredictorConfig
from drdfl.training import TrainingConfig, train_model


def test_evaluation_reports_oracles_and_ambiguity(tiny_train, tiny_validation, tmp_path: Path) -> None:
    model = CostPredictor(
        PredictorConfig(tiny_train.context_dim, tiny_train.graph.edge_count, 12, 1)
    )
    train_model(
        model,
        tiny_train,
        tiny_validation,
        config=TrainingConfig(mode="spo_kl_dro", epochs=1, warm_start_epochs=1, validation_every=1),
    )
    report = evaluate_models(
        {"spo_kl_dro": model},
        tiny_train,
        tiny_validation,
        scenario="unit",
        bootstrap_draws=20,
    )
    methods = {row.method for row in report.rows}
    assert "perfect_information_oracle" in methods
    assert "spo_kl_dro" in methods
    perfect = next(row for row in report.rows if row.method == "perfect_information_oracle")
    assert perfect.realized_mean_regret == 0.0
    assert all(row.feasible_rate == 1.0 for row in report.rows)
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    save_report_json(report, json_path)
    save_report_csv(report, csv_path)
    assert json_path.exists() and csv_path.exists()


def test_compact_research_protocol_runs() -> None:
    _models, report = run_research_experiment(
        ResearchConfig(
            layer_count=2,
            width=2,
            context_dim=3,
            train_samples=8,
            validation_samples=4,
            evaluation_samples=3,
            hidden_dim=8,
            hidden_layers=1,
            epochs=1,
            warm_start_epochs=0,
            bootstrap_draws=10,
            seed=17,
        )
    )
    assert set(report.training) == {"mse_erm", "mse_kl_dro", "spo_erm", "spo_kl_dro"}
    assert len(report.scenario_metadata) == 7
    assert report.evaluation_rows
