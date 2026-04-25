import json
import re
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel


EXTRACTION_PROMPT = """Extract entity-relationship triples from the following text.
Return ONLY a JSON array of objects with keys: head, relation, tail.

Text: {text}

JSON:"""


class GraphBuilder:
    def __init__(self, extractor_type: str = "llm", llm: BaseChatModel | None = None):
        self.extractor_type = extractor_type
        self.llm = llm

    def extract(self, documents: list[Document]) -> list[dict]:
        if self.extractor_type == "llm":
            return self._extract_llm(documents)
        elif self.extractor_type == "nlp":
            return self._extract_nlp(documents)
        return []

    def _extract_llm(self, documents: list[Document]) -> list[dict]:
        if not self.llm:
            return []
        all_triples = []
        for doc in documents:
            prompt = EXTRACTION_PROMPT.format(text=doc.page_content)
            response = self.llm.invoke(prompt)
            triples = self._parse_response(response.content)
            all_triples.extend(triples)
        return all_triples

    def _extract_nlp(self, documents: list[Document]) -> list[dict]:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")

        triples = []
        for doc in documents:
            spacy_doc = nlp(doc.page_content)
            entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]
            for i, (text_a, _) in enumerate(entities):
                for text_b, _ in entities[i + 1:]:
                    triples.append({"head": text_a, "relation": "related_to", "tail": text_b})
        return triples

    def _parse_response(self, content: str) -> list[dict]:
        try:
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return []
