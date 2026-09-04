"""Frozen train/evaluate protocol for distributionally robust DFL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from drdfl.dataset import ContextualDataset, collect_dataset
from drdfl.domain import LayeredGraph
from drdfl.evaluation import EvaluationReport, MethodMetrics, evaluate_models
from drdfl.generator import CostRegime
from drdfl.model import CostPredictor, PredictorConfig, save_checkpoint
from drdfl.training import TrainingConfig, TrainingMode, TrainingSummary, train_model
from drdfl.utils import write_json


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    layer_count: int = 4
    width: int = 4
    context_dim: int = 8
    train_samples: int = 128
    validation_samples: int = 40
    evaluation_samples: int = 48
    hidden_dim: int = 96
    hidden_layers: int = 2
    epochs: int = 100
    warm_start_epochs: int = 20
    kl_radius: float = 0.15
    bootstrap_draws: int = 500
    seed: int = 2026

    def __post_init__(self) -> None:
        counts = (
            self.layer_count,
            self.width,
            self.context_dim,
            self.train_samples,
            self.validation_samples,
            self.evaluation_samples,
            self.hidden_dim,
            self.hidden_layers,
            self.epochs,
            self.bootstrap_draws,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("research counts must be positive")
        if self.warm_start_epochs < 0:
            raise ValueError("warm_start_epochs must be nonnegative")
        if self.kl_radius < 0.0:
            raise ValueError("kl_radius must be nonnegative")


@dataclass(frozen=True, slots=True)
class ResearchReport:
    training: dict[str, dict[str, object]]
    evaluation_rows: tuple[MethodMetrics, ...]
    scenario_metadata: dict[str, dict[str, object]]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "training": self.training,
            "evaluation_rows": [row.to_dict() for row in self.evaluation_rows],
            "scenario_metadata": self.scenario_metadata,
            "metadata": self.metadata,
        }


def _training_datasets(
    config: ResearchConfig, graph: LayeredGraph
) -> tuple[ContextualDataset, ContextualDataset]:
    training = collect_dataset(
        graph,
        count=config.train_samples,
        context_dim=config.context_dim,
        seed=config.seed + 1_000,
        regimes=(
            "in_distribution",
            "in_distribution",
            "in_distribution",
            "mild_stress",
        ),
    )
    validation = collect_dataset(
        graph,
        count=config.validation_samples,
        context_dim=config.context_dim,
        seed=config.seed + 2_000,
        regimes=("in_distribution", "in_distribution", "mild_stress"),
    )
    return training, validation


def _scenario_specs() -> tuple[tuple[str, CostRegime], ...]:
    return (
        ("in_distribution", "in_distribution"),
        ("mean_shift", "mean_shift"),
        ("covariance_shift", "covariance_shift"),
        ("heavy_tail", "heavy_tail"),
        ("corridor_shift", "corridor_shift"),
        ("nonlinear_shift", "nonlinear_shift"),
        ("combined_shift", "combined_shift"),
    )


def _train_all_models(
    config: ResearchConfig,
    graph: LayeredGraph,
    training: ContextualDataset,
    validation: ContextualDataset,
) -> tuple[dict[str, CostPredictor], dict[str, TrainingSummary]]:
    modes: tuple[TrainingMode, ...] = (
        "mse_erm",
        "mse_kl_dro",
        "spo_erm",
        "spo_kl_dro",
    )
    models: dict[str, CostPredictor] = {}
    summaries: dict[str, TrainingSummary] = {}
    for index, mode in enumerate(modes):
        model = CostPredictor(
            PredictorConfig(
                context_dim=config.context_dim,
                edge_count=graph.edge_count,
                hidden_dim=config.hidden_dim,
                hidden_layers=config.hidden_layers,
            )
        )
        summary = train_model(
            model,
            training,
            validation,
            config=TrainingConfig(
                mode=mode,
                epochs=config.epochs,
                warm_start_epochs=config.warm_start_epochs,
                kl_radius=config.kl_radius,
                seed=config.seed + 10_000 * (index + 1),
            ),
        )
        models[mode] = model
        summaries[mode] = summary
    return models, summaries


def run_research_experiment(
    config: ResearchConfig | None = None,
    *,
    checkpoint_directory: str | Path | None = None,
) -> tuple[dict[str, CostPredictor], ResearchReport]:
    config = config or ResearchConfig()
    graph = LayeredGraph(config.layer_count, config.width)
    training, validation = _training_datasets(config, graph)
    models, summaries = _train_all_models(config, graph, training, validation)
    if checkpoint_directory is not None:
        directory = Path(checkpoint_directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, model in models.items():
            save_checkpoint(
                model,
                directory / f"{name}.safetensors",
                metadata={
                    "training_mode": name,
                    "graph": graph.to_dict(),
                    "graph_fingerprint": graph.fingerprint,
                    "train_fingerprint": training.fingerprint,
                    "validation_fingerprint": validation.fingerprint,
                },
            )

    all_rows: list[MethodMetrics] = []
    scenario_metadata: dict[str, dict[str, object]] = {}
    for scenario_index, (scenario_name, regime) in enumerate(_scenario_specs()):
        test = collect_dataset(
            graph,
            count=config.evaluation_samples,
            context_dim=config.context_dim,
            seed=config.seed + 100_000 + 10_000 * scenario_index,
            regimes=(regime,),
        )
        evaluation: EvaluationReport = evaluate_models(
            models,
            training,
            test,
            scenario=scenario_name,
            selected_radius=config.kl_radius,
            bootstrap_seed=config.seed + 200_000 + scenario_index,
            bootstrap_draws=config.bootstrap_draws,
        )
        all_rows.extend(evaluation.rows)
        scenario_metadata[scenario_name] = evaluation.metadata

    report = ResearchReport(
        training={name: summary.to_dict() for name, summary in summaries.items()},
        evaluation_rows=tuple(all_rows),
        scenario_metadata=scenario_metadata,
        metadata={
            "config": asdict(config),
            "graph": graph.to_dict(),
            "graph_fingerprint": graph.fingerprint,
            "train_fingerprint": training.fingerprint,
            "validation_fingerprint": validation.fingerprint,
            "training_regimes": list(training.regimes),
            "scenario_order": [name for name, _regime in _scenario_specs()],
            "claims_boundary": (
                "Small synthetic layered-DAG methodology benchmark; no universal robustness "
                "or state-of-the-art claim."
            ),
        },
    )
    return models, report


def save_research_report(report: ResearchReport, path: str | Path) -> None:
    write_json(report.to_dict(), path)
