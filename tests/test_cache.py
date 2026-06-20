from app.core.cache import QueryCache


def test_exact_hit_without_embedding():
    cache = QueryCache(embed_fn=None)
    cache.put("what is x", {"answer": "X"})
    assert cache.get("what is x") == {"answer": "X"}


def test_miss_returns_none():
    cache = QueryCache(embed_fn=None)
    assert cache.get("unknown") is None


def test_exact_match_normalizes_case_and_whitespace():
    cache = QueryCache(embed_fn=None)
    cache.put("What  IS  X?", {"answer": "X"})
    assert cache.get("what is x?") == {"answer": "X"}


def test_semantic_hit_above_threshold():
    vectors = {"a": [1.0, 0.0], "b": [0.99, 0.141]}  # cosine ~0.99
    cache = QueryCache(embed_fn=lambda q: vectors[q], threshold=0.95)
    cache.put("a", {"answer": "A"})
    assert cache.get("b") == {"answer": "A"}


def test_semantic_miss_below_threshold():
    vectors = {"a": [1.0, 0.0], "c": [0.0, 1.0]}  # cosine 0
    cache = QueryCache(embed_fn=lambda q: vectors[q], threshold=0.95)
    cache.put("a", {"answer": "A"})
    assert cache.get("c") is None


def test_zero_vector_does_not_crash():
    vectors = {"a": [0.0, 0.0], "b": [1.0, 0.0]}
    cache = QueryCache(embed_fn=lambda q: vectors[q], threshold=0.95)
    cache.put("a", {"answer": "A"})
    assert cache.get("b") is None


def test_eviction_caps_size_fifo():
    cache = QueryCache(embed_fn=None, max_size=2)
    cache.put("q1", {"a": 1})
    cache.put("q2", {"a": 2})
    cache.put("q3", {"a": 3})
    assert cache.get("q1") is None       # oldest evicted
    assert cache.get("q3") == {"a": 3}
