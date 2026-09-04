from __future__ import annotations

import pytest

from drdfl.dataset import ContextualDataset, collect_dataset
from drdfl.domain import LayeredGraph


@pytest.fixture
def tiny_graph() -> LayeredGraph:
    return LayeredGraph(layer_count=2, width=2)


@pytest.fixture
def tiny_train(tiny_graph: LayeredGraph) -> ContextualDataset:
    return collect_dataset(
        tiny_graph,
        count=10,
        context_dim=4,
        seed=100,
        regimes=("in_distribution", "mild_stress"),
    )


@pytest.fixture
def tiny_validation(tiny_graph: LayeredGraph) -> ContextualDataset:
    return collect_dataset(
        tiny_graph,
        count=6,
        context_dim=4,
        seed=200,
        regimes=("in_distribution",),
    )
