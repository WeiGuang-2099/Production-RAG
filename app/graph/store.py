import pickle
from pathlib import Path
import networkx as nx


class GraphStore:
    """Graph store for managing entity relationships."""

    def __init__(self, data_dir: str = "./data"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()
        self._load()

    def add_triples(self, triples: list[dict]) -> None:
        for triple in triples:
            self.graph.add_edge(
                triple["head"],
                triple["tail"],
                relation=triple["relation"],
            )
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
        with open(self._data_dir / "knowledge_graph.gpickle", "wb") as f:
            pickle.dump(self.graph, f)

    def _load(self) -> None:
        path = self._data_dir / "knowledge_graph.gpickle"
        if path.exists():
            with open(path, "rb") as f:
                self.graph = pickle.load(f)
