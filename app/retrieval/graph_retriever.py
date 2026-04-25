from langchain_core.documents import Document
from app.graph.store import GraphStore


class GraphRetriever:
    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def retrieve(self, query: str, depth: int = 1) -> list[Document]:
        neighbors = self.graph_store.get_neighbors(query, depth=depth)
        if not neighbors:
            return []
        documents = []
        for entity, relation in neighbors:
            documents.append(
                Document(
                    page_content=f"{query} --[{relation}]--> {entity}",
                    metadata={"source": "graph", "entity": entity, "relation": relation},
                )
            )
        return documents
