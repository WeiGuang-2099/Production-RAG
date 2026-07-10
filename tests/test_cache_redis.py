from unittest.mock import patch

import fakeredis
import pytest

from app.core.cache import QueryCache, RedisBackend


@pytest.fixture
def redis_backend():
    fake = fakeredis.FakeRedis(decode_responses=True)
    with patch("redis.Redis") as mock_redis:
        mock_redis.from_url.return_value = fake
        yield RedisBackend("redis://test:6379/0", max_size=2)


def test_exact_hit_roundtrip(redis_backend):
    cache = QueryCache(backend=redis_backend)
    cache.put("What is RAG?", {"answer": "retrieval augmented generation"})
    assert cache.get("what is  rag?") == {"answer": "retrieval augmented generation"}


def test_semantic_hit_over_redis_entries(redis_backend):
    vecs = {"a question": [1.0, 0.0], "a  question": [1.0, 0.0], "similar question": [0.99, 0.14]}
    cache = QueryCache(embed_fn=lambda q: vecs[q.strip().lower()], threshold=0.95,
                       backend=redis_backend)
    cache.put("a question", {"answer": "42"})
    assert cache.get("similar question") == {"answer": "42"}


def test_fifo_eviction_at_cap(redis_backend):
    cache = QueryCache(backend=redis_backend)
    cache.put("q1", {"answer": "1"})
    cache.put("q2", {"answer": "2"})
    cache.put("q3", {"answer": "3"})  # max_size=2 -> q1 evicted
    assert cache.get("q1") is None
    assert cache.get("q2") == {"answer": "2"}
    assert cache.get("q3") == {"answer": "3"}


def test_unreachable_redis_raises_at_construction():
    with patch("redis.Redis") as mock_redis:
        mock_redis.from_url.return_value.ping.side_effect = ConnectionError("refused")
        with pytest.raises(ConnectionError):
            RedisBackend("redis://down:6379/0")


def test_runtime_redis_error_degrades_to_miss(redis_backend):
    cache = QueryCache(backend=redis_backend)
    cache.put("q1", {"answer": "1"})
    with patch.object(redis_backend, "_client") as broken:
        broken.get.side_effect = RuntimeError("connection lost")
        broken.lrange.side_effect = RuntimeError("connection lost")
        assert cache.get("q1") is None  # miss, not crash


def test_pipeline_falls_back_to_memory_when_redis_down():
    from unittest.mock import MagicMock

    from app.core.cache import InMemoryBackend
    from app.core.pipeline import _make_cache_backend

    settings = MagicMock()
    settings.REDIS_URL = "redis://down:6379/0"
    with patch("app.core.cache.RedisBackend", side_effect=ConnectionError("refused")):
        backend = _make_cache_backend(settings)
    assert isinstance(backend, InMemoryBackend)
