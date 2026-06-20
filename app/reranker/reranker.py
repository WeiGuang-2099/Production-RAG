from langchain_cohere import CohereRerank
from langchain_core.documents import Document


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
        for result in results:
            doc = documents[result["index"]]
            new_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": result["relevance_score"]},
            )
            reranked.append(new_doc)
        return reranked
