import json
import logging
import re
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_spacy_cache = None

EXTRACTION_PROMPT = """Extract entity-relationship triples from the following text.
Return ONLY a JSON array of objects with keys: head, relation, tail.

Text: {text}

JSON:"""


class GraphBuilder:
    def __init__(self, extractor_type: str = "llm", llm: BaseChatModel | None = None):
        self.extractor_type = extractor_type
        self.llm = llm
        self._nlp = None
        if extractor_type == "nlp":
            self._nlp = self._load_spacy()

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
        if self._nlp is None:
            return []
        triples = []
        for doc in documents:
            spacy_doc = self._nlp(doc.page_content)
            entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]
            for i, (text_a, _) in enumerate(entities):
                for text_b, _ in entities[i + 1:]:
                    triples.append({"head": text_a, "relation": "related_to", "tail": text_b})
        return triples

    @staticmethod
    def _load_spacy():
        global _spacy_cache
        if _spacy_cache is not None:
            return _spacy_cache
        import spacy
        try:
            _spacy_cache = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found, using blank model")
            _spacy_cache = spacy.blank("en")
        return _spacy_cache

    def _parse_response(self, content: str) -> list[dict]:
        try:
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if not match:
                logger.warning("No JSON array found in LLM response: %s...", content[:200])
                return []
            triples = json.loads(match.group())
            valid = []
            for triple in triples:
                if all(k in triple for k in ("head", "relation", "tail")):
                    valid.append(triple)
                else:
                    logger.warning("Invalid triple structure (missing keys): %s", triple)
            logger.info("Parsed %d triples from LLM response", len(valid))
            return valid
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.error("Failed to parse LLM response: %s — content: %s...", exc, content[:200])
            return []
