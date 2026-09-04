"""Versioned contextual shortest-path corpora with exact path labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor

from drdfl.domain import LayeredGraph
from drdfl.generator import ContextualCostGenerator, CostRegime, GeneratorSpec

CORPUS_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class CostSample:
    sample_id: str
    context: tuple[float, ...]
    true_mean_costs: tuple[float, ...]
    realized_costs: tuple[float, ...]
    regime: str
    seed: int
    optimal_decision: tuple[int, ...]
    optimal_objective: float

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "context": list(self.context),
            "true_mean_costs": list(self.true_mean_costs),
            "realized_costs": list(self.realized_costs),
            "regime": self.regime,
            "seed": self.seed,
            "optimal_decision": list(self.optimal_decision),
            "optimal_objective": self.optimal_objective,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object], graph: LayeredGraph) -> CostSample:
        sample = cls(
            sample_id=str(payload["sample_id"]),
            context=tuple(float(str(value)) for value in cast(list[object], payload["context"])),
            true_mean_costs=tuple(
                float(str(value)) for value in cast(list[object], payload["true_mean_costs"])
            ),
            realized_costs=tuple(
                float(str(value)) for value in cast(list[object], payload["realized_costs"])
            ),
            regime=str(payload["regime"]),
            seed=int(str(payload["seed"])),
            optimal_decision=tuple(
                int(str(value)) for value in cast(list[object], payload["optimal_decision"])
            ),
            optimal_objective=float(str(payload["optimal_objective"])),
        )
        if len(sample.true_mean_costs) != graph.edge_count:
            raise ValueError("stored mean cost vector has the wrong size")
        if len(sample.realized_costs) != graph.edge_count:
            raise ValueError("stored realized cost vector has the wrong size")
        audit = graph.audit(sample.optimal_decision, sample.realized_costs)
        if not audit.feasible:
            raise ValueError("stored optimal decision is infeasible")
        exact = graph.shortest_path(sample.realized_costs)
        tolerance = 1e-8 * max(1.0, abs(exact.objective))
        if abs(exact.objective - sample.optimal_objective) > tolerance:
            raise ValueError("stored optimal objective does not match the exact path oracle")
        if abs(audit.objective - sample.optimal_objective) > tolerance:
            raise ValueError("stored decision objective is inconsistent")
        return sample


@dataclass(frozen=True, slots=True)
class ContextualDataset:
    graph: LayeredGraph
    samples: tuple[CostSample, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("dataset must contain at least one sample")
        identifiers = [sample.sample_id for sample in self.samples]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("sample identifiers must be unique")
        context_dimensions = {len(sample.context) for sample in self.samples}
        if len(context_dimensions) != 1:
            raise ValueError("all samples must have the same context dimension")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def context_dim(self) -> int:
        return len(self.samples[0].context)

    @property
    def regimes(self) -> tuple[str, ...]:
        return tuple(sorted({sample.regime for sample in self.samples}))

    @property
    def fingerprint(self) -> str:
        stable = {
            "graph": self.graph.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
        }
        encoded = json.dumps(
            stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def tensors(self, *, device: torch.device | str = "cpu") -> tuple[Tensor, Tensor, Tensor]:
        contexts = torch.tensor(
            [sample.context for sample in self.samples], dtype=torch.float32, device=device
        )
        costs = torch.tensor(
            [sample.realized_costs for sample in self.samples],
            dtype=torch.float32,
            device=device,
        )
        decisions = torch.tensor(
            [sample.optimal_decision for sample in self.samples],
            dtype=torch.float32,
            device=device,
        )
        return contexts, costs, decisions


def collect_dataset(
    graph: LayeredGraph,
    *,
    count: int,
    context_dim: int,
    seed: int,
    regimes: tuple[CostRegime, ...] = ("in_distribution",),
    structure_seed: int = 2026,
    noise_scale: float = 0.35,
) -> ContextualDataset:
    if count <= 0:
        raise ValueError("count must be positive")
    if not regimes:
        raise ValueError("at least one regime is required")
    generator = ContextualCostGenerator(
        GeneratorSpec(
            graph=graph,
            context_dim=context_dim,
            structure_seed=structure_seed,
            noise_scale=noise_scale,
        )
    )
    samples: list[CostSample] = []
    for index in range(count):
        regime = regimes[index % len(regimes)]
        sample_seed = seed + 104_729 * (index + 1)
        observation = generator.generate(regime=regime, seed=sample_seed)
        optimum = graph.shortest_path(observation.realized_costs)
        samples.append(
            CostSample(
                sample_id=f"sample-{index:05d}-{regime}-seed{sample_seed}",
                context=observation.context,
                true_mean_costs=observation.true_mean_costs,
                realized_costs=observation.realized_costs,
                regime=regime,
                seed=sample_seed,
                optimal_decision=optimum.decision,
                optimal_objective=optimum.objective,
            )
        )
    return ContextualDataset(
        graph=graph,
        samples=tuple(samples),
        metadata={
            "generator": "drdfl-contextual-cost-v1",
            "count": count,
            "context_dim": context_dim,
            "seed": seed,
            "structure_seed": structure_seed,
            "noise_scale": noise_scale,
            "regimes": list(regimes),
        },
    )


def split_dataset(
    dataset: ContextualDataset,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[ContextualDataset, ContextualDataset]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    if len(dataset.samples) < 2:
        raise ValueError("at least two samples are required for a split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset.samples))
    validation_count = max(
        1,
        min(len(dataset.samples) - 1, round(validation_fraction * len(dataset.samples))),
    )
    validation_indices = {int(index) for index in order[:validation_count]}
    train = tuple(
        sample for index, sample in enumerate(dataset.samples) if index not in validation_indices
    )
    validation = tuple(
        sample for index, sample in enumerate(dataset.samples) if index in validation_indices
    )
    return (
        ContextualDataset(dataset.graph, train, {**dataset.metadata, "split": "train"}),
        ContextualDataset(
            dataset.graph,
            validation,
            {**dataset.metadata, "split": "validation"},
        ),
    )


def save_dataset(dataset: ContextualDataset, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "type": "manifest",
        "schema_version": CORPUS_SCHEMA_VERSION,
        "graph": dataset.graph.to_dict(),
        "record_count": len(dataset.samples),
        "fingerprint": dataset.fingerprint,
        "metadata": dataset.metadata,
    }
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n")
        for sample in dataset.samples:
            handle.write(
                json.dumps(
                    {"type": "record", **sample.to_dict()},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_dataset(path: str | Path) -> ContextualDataset:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("dataset file is empty")
    manifest = json.loads(lines[0])
    if not isinstance(manifest, dict) or manifest.get("type") != "manifest":
        raise ValueError("first dataset line must be a manifest")
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("unsupported dataset schema version")
    graph_payload = cast(dict[str, object], manifest["graph"])
    graph = LayeredGraph.from_dict(graph_payload)
    samples: list[CostSample] = []
    for line_number, line in enumerate(lines[1:], start=2):
        payload = json.loads(line)
        if not isinstance(payload, dict) or payload.get("type") != "record":
            raise ValueError(f"line {line_number} is not a dataset record")
        samples.append(CostSample.from_dict(cast(dict[str, object], payload), graph))
    dataset = ContextualDataset(
        graph=graph,
        samples=tuple(samples),
        metadata=cast(dict[str, object], manifest.get("metadata", {})),
    )
    if len(dataset.samples) != int(str(manifest["record_count"])):
        raise ValueError("dataset record count does not match the manifest")
    if dataset.fingerprint != str(manifest["fingerprint"]):
        raise ValueError("dataset fingerprint does not match the manifest")
    return dataset
