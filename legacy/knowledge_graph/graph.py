from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from .extractor import Triple


class KnowledgeGraph:
    """
    Directed knowledge graph where nodes are entities and edges are relations.

    Backed by a NetworkX MultiDiGraph to support multiple distinct relations
    between the same pair of entities (e.g. Nokia→Tampere via both
    "located_in" and "found_in").
    """

    def __init__(self) -> None:
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Building the graph
    # ------------------------------------------------------------------

    def add_triple(self, triple: Triple) -> None:
        if not self._g.has_node(triple.subject):
            self._g.add_node(triple.subject, entity_type=triple.subject_type)
        if not self._g.has_node(triple.object):
            self._g.add_node(triple.object, entity_type=triple.object_type)
        self._g.add_edge(
            triple.subject,
            triple.object,
            relation=triple.relation,
            source=triple.source,
        )

    def add_triples(self, triples: list[Triple]) -> None:
        for triple in triples:
            self.add_triple(triple)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    @property
    def nodes(self):
        return self._g.nodes(data=True)

    @property
    def edges(self):
        return self._g.edges(data=True)

    def neighbors(self, entity: str) -> list[tuple[str, str]]:
        """Return [(relation, neighbor_entity)] for all outgoing edges from entity."""
        return [
            (data["relation"], neighbor)
            for _, neighbor, data in self._g.out_edges(entity, data=True)
        ]

    def entity_type(self, entity: str) -> str | None:
        return self._g.nodes[entity].get("entity_type") if self._g.has_node(entity) else None

    def multihop_paths(self, source: str, max_hops: int = 2) -> list[list[tuple]]:
        """
        DFS from source, returning all edge-paths up to max_hops long.

        Each path is a list of (from_entity, relation, to_entity) tuples.
        Stops when a node is revisited to avoid cycles.
        """
        paths: list[list[tuple]] = []
        self._dfs(source, [], set(), paths, max_hops)
        return paths

    def _dfs(
        self,
        node: str,
        path: list[tuple],
        visited: set[str],
        paths: list,
        remaining: int,
    ) -> None:
        if remaining == 0:
            return
        visited = visited | {node}
        for _, neighbor, data in self._g.out_edges(node, data=True):
            if neighbor in visited:
                continue
            edge = (node, data["relation"], neighbor)
            new_path = path + [edge]
            paths.append(new_path)
            self._dfs(neighbor, new_path, visited, paths, remaining - 1)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"KnowledgeGraph  nodes={self._g.number_of_nodes()}  edges={self._g.number_of_edges()}",
            "\nNodes:",
        ]
        for node, data in self._g.nodes(data=True):
            etype = data.get("entity_type") or "—"
            lines.append(f"  {node!r}  [{etype}]")
        lines.append("\nEdges:")
        for src, dst, data in self._g.edges(data=True):
            lines.append(f"  ({src!r}, {data['relation']!r}, {dst!r})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"KnowledgeGraph(nodes={self._g.number_of_nodes()}, "
            f"edges={self._g.number_of_edges()})"
        )
