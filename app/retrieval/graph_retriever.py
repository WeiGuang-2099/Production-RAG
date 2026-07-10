import re

from langchain_core.documents import Document

from app.graph.store import GraphStore


def _extract_candidates(query: str) -> list[str]:
    """Extract candidate entity terms from a natural language query."""
    stop_words = {
        "what", "who", "how", "why", "when", "where", "is", "are", "was", "were",
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "and", "or",
        "does", "do", "did", "can", "could", "would", "should", "tell", "me", "about",
    }
    tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    candidates = []
    # n-grams: try longer phrases first for better match
    for n in (4, 3, 2):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if not all(t in stop_words for t in tokens[i : i + n]):
                candidates.append(phrase)
    for token in tokens:
        if token not in stop_words:
            candidates.append(token)
    return candidates


class GraphRetriever:
    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def retrieve(self, query: str, depth: int = 1) -> list[Document]:
        # Refresh at entry: _find_entity reads graph_store.graph directly, so
        # the staleness check cannot live inside get_neighbors.
        self.graph_store.refresh_if_stale()
        matched_entity = self._find_entity(query)
        if not matched_entity:
            return []
        neighbors = self.graph_store.get_neighbors(matched_entity, depth=depth)
        if not neighbors:
            return []
        documents = []
        for entity, relation in neighbors:
            documents.append(
                Document(
                    page_content=f"{matched_entity} --[{relation}]--> {entity}",
                    metadata={
                        "source": "graph",
                        "query_entity": matched_entity,
                        "entity": entity,
                        "relation": relation,
                    },
                )
            )
        return documents

    def _find_entity(self, query: str) -> str | None:
        """Find the best-matching graph node for the given query."""
        graph = self.graph_store.graph
        if not graph.nodes:
            return None
        # Build lowercase lookup for case-insensitive matching
        node_lookup = {node.lower(): node for node in graph.nodes}
        candidates = _extract_candidates(query)
        for candidate in candidates:
            if candidate in node_lookup:
                return node_lookup[candidate]
        # Fallback: substring match
        query_lower = query.lower()
        for node in graph.nodes:
            if node.lower() in query_lower:
                return node
        return None
