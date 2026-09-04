"""Typed layered-DAG shortest-path domain and independent path audits."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np


@dataclass(frozen=True, slots=True)
class Edge:
    index: int
    tail: int
    head: int


@dataclass(frozen=True, slots=True)
class PathAudit:
    feasible: bool
    integrality_violation: float
    flow_violation: float
    source_outflow: float
    sink_inflow: float
    objective: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PathSolution:
    edge_indices: tuple[int, ...]
    node_sequence: tuple[int, ...]
    decision: tuple[int, ...]
    objective: float

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_indices": list(self.edge_indices),
            "node_sequence": list(self.node_sequence),
            "decision": list(self.decision),
            "objective": self.objective,
        }


@dataclass(frozen=True, slots=True)
class LayeredGraph:
    """Complete bipartite transitions between consecutive hidden layers."""

    layer_count: int
    width: int

    def __post_init__(self) -> None:
        if self.layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.layer_count > 12 or self.width > 16:
            raise ValueError("reference implementation limits are 12 layers and width 16")

    @property
    def source(self) -> int:
        return 0

    @property
    def sink(self) -> int:
        return 1 + self.layer_count * self.width

    @property
    def node_count(self) -> int:
        return self.sink + 1

    def layer_nodes(self, layer: int) -> tuple[int, ...]:
        if not 0 <= layer < self.layer_count:
            raise ValueError("layer is outside the graph")
        start = 1 + layer * self.width
        return tuple(range(start, start + self.width))

    @property
    def edges(self) -> tuple[Edge, ...]:
        pairs: list[tuple[int, int]] = []
        for node in self.layer_nodes(0):
            pairs.append((self.source, node))
        for layer in range(self.layer_count - 1):
            for tail in self.layer_nodes(layer):
                for head in self.layer_nodes(layer + 1):
                    pairs.append((tail, head))
        for node in self.layer_nodes(self.layer_count - 1):
            pairs.append((node, self.sink))
        return tuple(Edge(index, tail, head) for index, (tail, head) in enumerate(pairs))

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def topological_order(self) -> tuple[int, ...]:
        nodes: list[int] = [self.source]
        for layer in range(self.layer_count):
            nodes.extend(self.layer_nodes(layer))
        nodes.append(self.sink)
        return tuple(nodes)

    @property
    def outgoing(self) -> tuple[tuple[int, ...], ...]:
        outgoing: list[list[int]] = [[] for _ in range(self.node_count)]
        for edge in self.edges:
            outgoing[edge.tail].append(edge.index)
        return tuple(tuple(indices) for indices in outgoing)

    @property
    def incoming(self) -> tuple[tuple[int, ...], ...]:
        incoming: list[list[int]] = [[] for _ in range(self.node_count)]
        for edge in self.edges:
            incoming[edge.head].append(edge.index)
        return tuple(tuple(indices) for indices in incoming)

    def shortest_path(self, costs: tuple[float, ...] | list[float] | np.ndarray) -> PathSolution:
        values = np.asarray(costs, dtype=float)
        if values.shape != (self.edge_count,):
            raise ValueError("cost vector has the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("edge costs must be finite")

        distance = np.full(self.node_count, np.inf, dtype=float)
        predecessor_edge = np.full(self.node_count, -1, dtype=int)
        distance[self.source] = 0.0
        edges = self.edges
        outgoing = self.outgoing
        for node in self.topological_order[:-1]:
            if not math.isfinite(float(distance[node])):
                continue
            for edge_index in outgoing[node]:
                edge = edges[edge_index]
                candidate = float(distance[node] + values[edge_index])
                incumbent = float(distance[edge.head])
                if candidate < incumbent - 1e-12 or (
                    abs(candidate - incumbent) <= 1e-12
                    and (
                        predecessor_edge[edge.head] < 0 or edge_index < predecessor_edge[edge.head]
                    )
                ):
                    distance[edge.head] = candidate
                    predecessor_edge[edge.head] = edge_index

        if predecessor_edge[self.sink] < 0:
            raise RuntimeError("layered graph has no source-to-sink path")
        reversed_edges: list[int] = []
        reversed_nodes: list[int] = [self.sink]
        node = self.sink
        while node != self.source:
            edge_index = int(predecessor_edge[node])
            if edge_index < 0:
                raise RuntimeError("predecessor chain is incomplete")
            reversed_edges.append(edge_index)
            node = edges[edge_index].tail
            reversed_nodes.append(node)
        path_edges = tuple(reversed(reversed_edges))
        node_sequence = tuple(reversed(reversed_nodes))
        decision = [0] * self.edge_count
        for edge_index in path_edges:
            decision[edge_index] = 1
        solution = PathSolution(
            edge_indices=path_edges,
            node_sequence=node_sequence,
            decision=tuple(decision),
            objective=float(distance[self.sink]),
        )
        audit = self.audit(solution.decision, values)
        if not audit.feasible or abs(audit.objective - solution.objective) > 1e-8:
            raise RuntimeError("shortest-path solution failed independent audit")
        return solution

    def audit(
        self,
        decision: tuple[float, ...] | list[float] | np.ndarray,
        costs: tuple[float, ...] | list[float] | np.ndarray,
        *,
        tolerance: float = 1e-8,
    ) -> PathAudit:
        values = np.asarray(decision, dtype=float)
        cost_values = np.asarray(costs, dtype=float)
        if values.shape != (self.edge_count,) or cost_values.shape != (self.edge_count,):
            raise ValueError("decision and cost vectors must match the graph edge count")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(cost_values)):
            raise ValueError("decision and cost vectors must be finite")
        rounded = np.rint(values)
        integrality = float(np.max(np.abs(values - rounded)))
        bound_violation = float(
            max(
                np.max(np.maximum(0.0, -values)),
                np.max(np.maximum(0.0, values - 1.0)),
            )
        )
        outgoing = self.outgoing
        incoming = self.incoming
        balances = np.zeros(self.node_count, dtype=float)
        for node in range(self.node_count):
            balances[node] = float(
                np.sum(values[list(outgoing[node])]) - np.sum(values[list(incoming[node])])
            )
        target = np.zeros(self.node_count, dtype=float)
        target[self.source] = 1.0
        target[self.sink] = -1.0
        flow_violation = float(np.max(np.abs(balances - target)))
        feasible = (
            integrality <= tolerance
            and bound_violation <= tolerance
            and flow_violation <= tolerance
        )
        return PathAudit(
            feasible=feasible,
            integrality_violation=max(integrality, bound_violation),
            flow_violation=flow_violation,
            source_outflow=float(np.sum(values[list(outgoing[self.source])])),
            sink_inflow=float(np.sum(values[list(incoming[self.sink])])),
            objective=float(np.dot(cost_values, values)),
        )

    def enumerate_paths(self, *, maximum_paths: int = 1_000_000) -> tuple[PathSolution, ...]:
        path_count = self.width**self.layer_count
        if path_count > maximum_paths:
            raise ValueError("path support exceeds maximum_paths")
        paths: list[PathSolution] = []
        edges_by_pair = {(edge.tail, edge.head): edge.index for edge in self.edges}

        def visit(layer: int, nodes: list[int]) -> None:
            if layer == self.layer_count:
                sequence = [self.source, *nodes, self.sink]
                edge_indices = tuple(
                    edges_by_pair[(sequence[index], sequence[index + 1])]
                    for index in range(len(sequence) - 1)
                )
                decision = [0] * self.edge_count
                for edge_index in edge_indices:
                    decision[edge_index] = 1
                paths.append(
                    PathSolution(
                        edge_indices=edge_indices,
                        node_sequence=tuple(sequence),
                        decision=tuple(decision),
                        objective=0.0,
                    )
                )
                return
            for node in self.layer_nodes(layer):
                nodes.append(node)
                visit(layer + 1, nodes)
                nodes.pop()

        visit(0, [])
        return tuple(paths)

    def to_dict(self) -> dict[str, object]:
        return {"layer_count": self.layer_count, "width": self.width}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> LayeredGraph:
        return cls(layer_count=int(str(payload["layer_count"])), width=int(str(payload["width"])))


def save_graph(graph: LayeredGraph, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(graph.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_graph(path: str | Path) -> LayeredGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("graph file must contain a JSON object")
    return LayeredGraph.from_dict(cast(dict[str, object], payload))
