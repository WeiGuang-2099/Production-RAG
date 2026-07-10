"""A small in-process query cache (exact + optional semantic).

On a hit, the whole query is skipped — no retrieval, no generation — which is
the single biggest latency/cost win for repeated or near-duplicate questions
(common in demos and FAQ-style traffic). Exact matches are normalized for
case/whitespace; if an ``embed_fn`` is provided, near-duplicates above a
cosine-similarity threshold also hit.

This is a process-local cache (cleared on restart); a production deployment
would back it with Redis, but the lookup semantics are identical.
"""
from __future__ import annotations

import math
import re
from collections import OrderedDict
from collections.abc import Callable


def _normalize(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryBackend:
    """Process-local FIFO entry store (the original QueryCache storage)."""

    def __init__(self, max_size: int = 256) -> None:
        self.max_size = max_size
        self._entries: OrderedDict[str, dict] = OrderedDict()

    def get(self, key: str) -> dict | None:
        return self._entries.get(key)

    def put(self, key: str, entry: dict) -> None:
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)  # evict oldest (FIFO)

    def entries(self) -> list[dict]:
        return list(self._entries.values())


class QueryCache:
    def __init__(
        self,
        embed_fn: Callable[[str], list[float]] | None = None,
        threshold: float = 0.95,
        max_size: int = 256,
        backend=None,
    ) -> None:
        self.embed_fn = embed_fn
        self.threshold = threshold
        self._backend = backend if backend is not None else InMemoryBackend(max_size)

    def get(self, question: str) -> dict | None:
        key = _normalize(question)
        entry = self._backend.get(key)
        if entry is not None:
            return entry["result"]

        if self.embed_fn is None:
            return None
        cached_entries = self._backend.entries()
        if not cached_entries:
            return None

        query_vec = self.embed_fn(question)
        best_result = None
        best_sim = self.threshold
        for cached in cached_entries:
            emb = cached.get("embedding")
            if emb is None:
                continue
            sim = _cosine(query_vec, emb)
            if sim >= best_sim:
                best_sim = sim
                best_result = cached["result"]
        return best_result

    def put(self, question: str, result: dict) -> None:
        key = _normalize(question)
        embedding = self.embed_fn(question) if self.embed_fn is not None else None
        self._backend.put(key, {"embedding": embedding, "result": result})
