from langchain_core.documents import Document
from langchain_cohere import CohereRerank


class RerankerService:
    def __init__(self, reranker: CohereRerank | None = None):
        self.reranker = reranker

    def rerank(self, query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
        if self.reranker is None:
            return documents[:top_k]

        results = self.reranker.rerank(
            documents=[doc.page_content for doc in documents],
            query=query,
            top_n=top_k,
        )
        reranked = []
        for result in results.results:
            doc = documents[result.index]
            doc.metadata["relevance_score"] = result.relevance_score
            reranked.append(doc)
        return reranked
