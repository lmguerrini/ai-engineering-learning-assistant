"""Tests for the caching service."""

import time
from pathlib import Path

import pytest

from src.services.cache import (
    build_cache_key,
    clear_expired_cache,
    get_cached_value,
    set_cached_value,
)


@pytest.fixture()
def cache_db(tmp_path: Path) -> Path:
    return tmp_path / "test_cache.db"


# ---------------------------------------------------------------------------
# build_cache_key
# ---------------------------------------------------------------------------

class TestBuildCacheKey:
    def test_deterministic(self):
        key1 = build_cache_key("op", {"a": 1, "b": 2})
        key2 = build_cache_key("op", {"b": 2, "a": 1})
        assert key1 == key2, "Keys should be identical regardless of dict order"

    def test_different_operations(self):
        key1 = build_cache_key("learn", {"topic": "AI"})
        key2 = build_cache_key("quiz", {"topic": "AI"})
        assert key1 != key2

    def test_different_payloads(self):
        key1 = build_cache_key("op", {"topic": "A"})
        key2 = build_cache_key("op", {"topic": "B"})
        assert key1 != key2

    def test_key_format(self):
        key = build_cache_key("learn_guide", {"x": 1})
        assert key.startswith("learn_guide:")
        assert len(key) > len("learn_guide:")


# ---------------------------------------------------------------------------
# set / get
# ---------------------------------------------------------------------------

class TestSetGet:
    def test_round_trip(self, cache_db: Path):
        set_cached_value("k1", {"hello": "world"}, db_path=cache_db)
        result = get_cached_value("k1", db_path=cache_db)
        assert result == {"hello": "world"}

    def test_miss_returns_none(self, cache_db: Path):
        assert get_cached_value("nonexistent", db_path=cache_db) is None

    def test_overwrite(self, cache_db: Path):
        set_cached_value("k1", "v1", db_path=cache_db)
        set_cached_value("k1", "v2", db_path=cache_db)
        assert get_cached_value("k1", db_path=cache_db) == "v2"

    def test_list_value(self, cache_db: Path):
        set_cached_value("k1", [1, 2, 3], db_path=cache_db)
        assert get_cached_value("k1", db_path=cache_db) == [1, 2, 3]

    def test_metadata_does_not_affect_value(self, cache_db: Path):
        set_cached_value("k1", "val", metadata={"source": "test"}, db_path=cache_db)
        assert get_cached_value("k1", db_path=cache_db) == "val"


# ---------------------------------------------------------------------------
# Expiration
# ---------------------------------------------------------------------------

class TestExpiration:
    def test_expired_entry_returns_none(self, cache_db: Path):
        set_cached_value("k1", "val", ttl_seconds=0, db_path=cache_db)
        time.sleep(0.05)
        assert get_cached_value("k1", db_path=cache_db) is None

    def test_non_expired_entry_returns_value(self, cache_db: Path):
        set_cached_value("k1", "val", ttl_seconds=3600, db_path=cache_db)
        assert get_cached_value("k1", db_path=cache_db) == "val"

    def test_clear_expired(self, cache_db: Path):
        set_cached_value("k1", "v1", ttl_seconds=0, db_path=cache_db)
        set_cached_value("k2", "v2", ttl_seconds=3600, db_path=cache_db)
        time.sleep(0.05)
        deleted = clear_expired_cache(db_path=cache_db)
        assert deleted == 1
        assert get_cached_value("k2", db_path=cache_db) == "v2"


# ---------------------------------------------------------------------------
# Graceful fallback
# ---------------------------------------------------------------------------

class TestGracefulFallback:
    def test_get_on_corrupt_path(self, tmp_path: Path):
        bad_path = tmp_path / "no_such_dir" / "deep" / "cache.db"
        # Should not crash — just return None
        result = get_cached_value("key", db_path=bad_path)
        # The dir will be created by _get_conn, so this should work fine
        assert result is None

    def test_set_does_not_crash_on_unserializable(self, cache_db: Path):
        # set with non-JSON-serializable should not crash (default=str handles it)
        set_cached_value("k1", {"func": lambda x: x}, db_path=cache_db)
        # value may not round-trip perfectly but shouldn't crash
