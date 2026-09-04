from __future__ import annotations

import numpy as np
import pytest

from drdfl.domain import LayeredGraph
from drdfl.oracle import verify_shortest_path_by_enumeration


def test_graph_shape_and_deterministic_edges(tiny_graph: LayeredGraph) -> None:
    assert tiny_graph.node_count == 6
    assert tiny_graph.edge_count == 8
    assert tiny_graph.edges[0].tail == tiny_graph.source
    assert tiny_graph.edges[-1].head == tiny_graph.sink
    assert tiny_graph.fingerprint == LayeredGraph(2, 2).fingerprint


def test_shortest_path_matches_exhaustive_enumeration(tiny_graph: LayeredGraph) -> None:
    costs = np.asarray([2.0, 1.0, 4.0, 0.5, 3.0, 2.0, 1.0, 3.0])
    comparison = verify_shortest_path_by_enumeration(tiny_graph, costs)
    solution = tiny_graph.shortest_path(costs)
    assert comparison.verified
    assert comparison.path_count == 4
    assert solution.objective == pytest.approx(comparison.enumeration_objective)
    assert tiny_graph.audit(solution.decision, costs).feasible


def test_tie_breaking_is_deterministic(tiny_graph: LayeredGraph) -> None:
    costs = np.ones(tiny_graph.edge_count)
    first = tiny_graph.shortest_path(costs)
    second = tiny_graph.shortest_path(costs)
    assert first.edge_indices == second.edge_indices
    assert first.objective == 3.0


def test_audit_rejects_non_path_flow(tiny_graph: LayeredGraph) -> None:
    decision = np.zeros(tiny_graph.edge_count)
    decision[0] = 1.0
    audit = tiny_graph.audit(decision, np.ones(tiny_graph.edge_count))
    assert not audit.feasible
    assert audit.flow_violation > 0.0


def test_graph_validation_and_enumeration_limit() -> None:
    with pytest.raises(ValueError):
        LayeredGraph(0, 2)
    graph = LayeredGraph(5, 5)
    with pytest.raises(ValueError):
        graph.enumerate_paths(maximum_paths=100)
