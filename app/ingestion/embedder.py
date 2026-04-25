from langchain_core.embeddings import Embeddings


class Embedder:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)
