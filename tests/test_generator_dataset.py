from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drdfl.dataset import collect_dataset, load_dataset, save_dataset, split_dataset
from drdfl.generator import ContextualCostGenerator, GeneratorSpec


def test_generator_is_deterministic_and_shifted(tiny_graph) -> None:
    generator = ContextualCostGenerator(GeneratorSpec(tiny_graph, context_dim=4))
    first = generator.generate(seed=42)
    second = generator.generate(seed=42)
    shifted = generator.generate(regime="corridor_shift", seed=42)
    assert first == second
    assert first.realized_costs != shifted.realized_costs
    assert min(first.realized_costs) > 0.0


def test_dataset_round_trip_and_fingerprint(tiny_train, tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    save_dataset(tiny_train, path)
    loaded = load_dataset(path)
    assert loaded.fingerprint == tiny_train.fingerprint
    assert loaded.graph == tiny_train.graph
    assert loaded.samples == tiny_train.samples


def test_dataset_tamper_is_detected(tiny_train, tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    save_dataset(tiny_train, path)
    lines = path.read_text().splitlines()
    record = json.loads(lines[1])
    record["realized_costs"][0] += 1.0
    lines[1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_split_is_disjoint_and_complete(tiny_train) -> None:
    train, validation = split_dataset(tiny_train, validation_fraction=0.3, seed=7)
    train_ids = {sample.sample_id for sample in train.samples}
    validation_ids = {sample.sample_id for sample in validation.samples}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == {sample.sample_id for sample in tiny_train.samples}


def test_dataset_tensors_have_expected_shapes(tiny_train) -> None:
    contexts, costs, decisions = tiny_train.tensors()
    assert contexts.shape == (10, 4)
    assert costs.shape == decisions.shape == (10, tiny_train.graph.edge_count)
    assert np.allclose(decisions.sum(dim=1).numpy(), 3.0)


def test_collect_requires_positive_count(tiny_graph) -> None:
    with pytest.raises(ValueError):
        collect_dataset(tiny_graph, count=0, context_dim=4, seed=0)
