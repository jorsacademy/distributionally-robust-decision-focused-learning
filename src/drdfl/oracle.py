"""Independent finite-support checks for shortest paths and structured losses."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from drdfl.domain import LayeredGraph


@dataclass(frozen=True, slots=True)
class OracleComparison:
    dynamic_programming_objective: float
    enumeration_objective: float
    objective_gap: float
    path_count: int
    verified: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_shortest_path_by_enumeration(
    graph: LayeredGraph,
    costs: np.ndarray | tuple[float, ...] | list[float],
    *,
    maximum_paths: int = 1_000_000,
    tolerance: float = 1e-9,
) -> OracleComparison:
    values = np.asarray(costs, dtype=float)
    if values.shape != (graph.edge_count,):
        raise ValueError("cost vector has the wrong shape")
    dynamic = graph.shortest_path(values)
    paths = graph.enumerate_paths(maximum_paths=maximum_paths)
    objectives = np.asarray(
        [np.dot(values, np.asarray(path.decision, dtype=float)) for path in paths],
        dtype=float,
    )
    enumeration = float(np.min(objectives))
    gap = dynamic.objective - enumeration
    scale = max(1.0, abs(dynamic.objective), abs(enumeration))
    verified = abs(gap) <= tolerance * scale
    if not verified:
        raise RuntimeError("dynamic-programming and exhaustive shortest-path oracles disagree")
    return OracleComparison(
        dynamic_programming_objective=dynamic.objective,
        enumeration_objective=enumeration,
        objective_gap=gap,
        path_count=len(paths),
        verified=True,
    )
