import logging
import pickle
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


class GraphStore:
    """Graph store for managing entity relationships."""

    def __init__(self, data_dir: str = "./data"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()
        self._load()

    def add_triples(self, triples: list[dict]) -> None:
        for triple in triples:
            head, tail = triple.get("head"), triple.get("tail")
            # A null/blank endpoint cannot be a node; skip rather than crash the
            # whole batch (networkx raises "None cannot be a node").
            if not head or not tail:
                logger.warning("Skipping triple with empty head/tail: %s", triple)
                continue
            self.graph.add_edge(head, tail, relation=triple.get("relation", ""))
        if self.graph.number_of_nodes() > 100000:
            logger.warning("Knowledge graph has %d nodes; consider migrating to a graph database for scale", self.graph.number_of_nodes())
        self._save()

    def get_neighbors(self, entity: str, depth: int = 1) -> list[tuple[str, str]]:
        if entity not in self.graph:
            return []
        visited = set()
        queue = [(entity, 0)]
        result = []
        while queue:
            node, d = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for _, neighbor, data in self.graph.edges(node, data=True):
                result.append((neighbor, data.get("relation", "")))
                if d < depth and neighbor not in visited:
                    queue.append((neighbor, d + 1))
            for pred, _, data in self.graph.in_edges(node, data=True):
                result.append((pred, data.get("relation", "")))
                if d < depth and pred not in visited:
                    queue.append((pred, d + 1))
        return result

    def _save(self) -> None:
        try:
            with open(self._data_dir / "knowledge_graph.gpickle", "wb") as f:
                pickle.dump(self.graph, f)
            logger.debug("Knowledge graph saved: %d nodes, %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges())
        except Exception as exc:
            logger.error("Failed to save knowledge graph: %s", exc)

    def _load(self) -> None:
        path = self._data_dir / "knowledge_graph.gpickle"
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                self.graph = pickle.load(f)
            logger.info("Knowledge graph loaded: %d nodes, %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges())
        except Exception as exc:
            logger.error("Failed to load knowledge graph: %s", exc)
