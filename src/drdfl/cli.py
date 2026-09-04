"""Command-line workflows for data, training, and robust evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

from drdfl.dataset import collect_dataset, load_dataset, save_dataset
from drdfl.domain import LayeredGraph
from drdfl.evaluation import evaluate_models, save_report_csv, save_report_json
from drdfl.experiment import ResearchConfig, run_research_experiment, save_research_report
from drdfl.generator import SUPPORTED_REGIMES, ContextualCostGenerator, GeneratorSpec
from drdfl.model import CostPredictor, PredictorConfig, load_checkpoint, save_checkpoint
from drdfl.oracle import verify_shortest_path_by_enumeration
from drdfl.training import SUPPORTED_MODES, TrainingConfig, train_model
from drdfl.utils import write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drdfl",
        description="Distributionally robust decision-focused shortest-path learning",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate one contextual instance")
    generate.add_argument("--layers", type=int, default=4)
    generate.add_argument("--width", type=int, default=4)
    generate.add_argument("--context-dim", type=int, default=8)
    generate.add_argument("--regime", choices=SUPPORTED_REGIMES, default="in_distribution")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--structure-seed", type=int, default=2026)
    generate.add_argument("--output", type=Path, required=True)

    collect = subparsers.add_parser("collect", help="build a deterministic JSONL corpus")
    collect.add_argument("--count", type=int, required=True)
    collect.add_argument("--layers", type=int, default=4)
    collect.add_argument("--width", type=int, default=4)
    collect.add_argument("--context-dim", type=int, default=8)
    collect.add_argument(
        "--regimes", nargs="+", choices=SUPPORTED_REGIMES, default=["in_distribution"]
    )
    collect.add_argument("--seed", type=int, default=1000)
    collect.add_argument("--structure-seed", type=int, default=2026)
    collect.add_argument("--output", type=Path, required=True)

    oracle = subparsers.add_parser("oracle", help="cross-check dynamic programming by enumeration")
    oracle.add_argument("dataset", type=Path)
    oracle.add_argument("--sample-index", type=int, default=0)
    oracle.add_argument("--maximum-paths", type=int, default=1_000_000)
    oracle.add_argument("--output", type=Path)

    train = subparsers.add_parser("train", help="train one prediction or DFL model")
    train.add_argument("dataset", type=Path)
    train.add_argument("--validation", type=Path, required=True)
    train.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--warm-start-epochs", type=int, default=20)
    train.add_argument("--hidden-dim", type=int, default=96)
    train.add_argument("--hidden-layers", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument("--kl-radius", type=float, default=0.15)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--output-report", type=Path)

    benchmark = subparsers.add_parser(
        "benchmark", help="evaluate checkpoints against exact path regret"
    )
    benchmark.add_argument("dataset", type=Path)
    benchmark.add_argument("--train-dataset", type=Path, required=True)
    benchmark.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="repeat for each model",
    )
    benchmark.add_argument("--scenario", default="benchmark")
    benchmark.add_argument("--selected-radius", type=float, default=0.15)
    benchmark.add_argument("--bootstrap-draws", type=int, default=500)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--output-json", type=Path, required=True)
    benchmark.add_argument("--output-csv", type=Path)

    research = subparsers.add_parser("research", help="run the frozen multi-shift protocol")
    research.add_argument("--config", type=Path)
    research.add_argument("--checkpoint-directory", type=Path)
    research.add_argument("--output-report", type=Path, required=True)
    return parser


def _load_research_config(path: Path | None) -> ResearchConfig:
    if path is None:
        return ResearchConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("research config must contain a JSON object")
    return ResearchConfig(**cast(dict[str, object], payload))


def _parse_checkpoints(items: list[str]) -> dict[str, CostPredictor]:
    models: dict[str, CostPredictor] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("checkpoint arguments must use NAME=PATH")
        name, raw_path = item.split("=", 1)
        if not name.strip() or name in models:
            raise ValueError("checkpoint names must be nonempty and unique")
        model, _metadata = load_checkpoint(raw_path)
        models[name] = model
    return models


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "generate":
        graph = LayeredGraph(args.layers, args.width)
        generator = ContextualCostGenerator(
            GeneratorSpec(
                graph=graph,
                context_dim=args.context_dim,
                structure_seed=args.structure_seed,
            )
        )
        observation = generator.generate(regime=args.regime, seed=args.seed)
        solution = graph.shortest_path(observation.realized_costs)
        payload: dict[str, object] = {
            "graph": graph.to_dict(),
            "graph_fingerprint": graph.fingerprint,
            "context": list(observation.context),
            "true_mean_costs": list(observation.true_mean_costs),
            "realized_costs": list(observation.realized_costs),
            "regime": observation.regime,
            "seed": observation.seed,
            "exact_shortest_path": solution.to_dict(),
        }
        write_json(payload, args.output)
        return {"output": str(args.output), **payload}

    if args.command == "collect":
        graph = LayeredGraph(args.layers, args.width)
        dataset = collect_dataset(
            graph,
            count=args.count,
            context_dim=args.context_dim,
            seed=args.seed,
            regimes=tuple(args.regimes),
            structure_seed=args.structure_seed,
        )
        save_dataset(dataset, args.output)
        return {
            "output": str(args.output),
            "record_count": len(dataset.samples),
            "fingerprint": dataset.fingerprint,
            "graph_fingerprint": graph.fingerprint,
            "regimes": list(dataset.regimes),
        }

    if args.command == "oracle":
        dataset = load_dataset(args.dataset)
        if not 0 <= args.sample_index < len(dataset.samples):
            raise ValueError("sample index is outside the dataset")
        sample = dataset.samples[args.sample_index]
        comparison = verify_shortest_path_by_enumeration(
            dataset.graph,
            sample.realized_costs,
            maximum_paths=args.maximum_paths,
        )
        payload = {
            "dataset_fingerprint": dataset.fingerprint,
            "sample_id": sample.sample_id,
            "comparison": comparison.to_dict(),
        }
        if args.output is not None:
            write_json(payload, args.output)
        return payload

    if args.command == "train":
        dataset = load_dataset(args.dataset)
        validation = load_dataset(args.validation)
        model = CostPredictor(
            PredictorConfig(
                context_dim=dataset.context_dim,
                edge_count=dataset.graph.edge_count,
                hidden_dim=args.hidden_dim,
                hidden_layers=args.hidden_layers,
            )
        )
        config = TrainingConfig(
            mode=args.mode,
            epochs=args.epochs,
            warm_start_epochs=args.warm_start_epochs,
            learning_rate=args.learning_rate,
            kl_radius=args.kl_radius,
            seed=args.seed,
        )
        summary = train_model(model, dataset, validation, config=config)
        save_checkpoint(
            model,
            args.checkpoint,
            metadata={
                "training_mode": args.mode,
                "graph": dataset.graph.to_dict(),
                "graph_fingerprint": dataset.graph.fingerprint,
                "train_fingerprint": dataset.fingerprint,
                "validation_fingerprint": validation.fingerprint,
                "training_config": asdict(config),
            },
        )
        payload = {"checkpoint": str(args.checkpoint), **summary.to_dict()}
        if args.output_report is not None:
            write_json(payload, args.output_report)
        return payload

    if args.command == "benchmark":
        dataset = load_dataset(args.dataset)
        train_dataset = load_dataset(args.train_dataset)
        models = _parse_checkpoints(args.checkpoint)
        report = evaluate_models(
            models,
            train_dataset,
            dataset,
            scenario=args.scenario,
            selected_radius=args.selected_radius,
            bootstrap_seed=args.seed,
            bootstrap_draws=args.bootstrap_draws,
        )
        save_report_json(report, args.output_json)
        if args.output_csv is not None:
            save_report_csv(report, args.output_csv)
        return report.to_dict()

    if args.command == "research":
        config = _load_research_config(args.config)
        _models, report = run_research_experiment(
            config,
            checkpoint_directory=args.checkpoint_directory,
        )
        save_research_report(report, args.output_report)
        return report.to_dict()

    raise RuntimeError("unreachable command")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        payload = _run(args)
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"error": type(error).__name__, "message": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
