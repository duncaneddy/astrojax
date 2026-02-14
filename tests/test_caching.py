"""Tests for astrojax.utils.caching module."""

import hashlib
import os
import time
from pathlib import Path

import pytest

from astrojax.utils.caching import (
    file_age_days,
    file_age_seconds,
    file_hash,
    get_cache_dir,
    get_datasets_cache_dir,
    get_eop_cache_dir,
    is_file_stale,
)

# ---------------------------------------------------------------------------
# get_cache_dir
# ---------------------------------------------------------------------------


class TestGetCacheDir:
    """Tests for get_cache_dir()."""

    def test_default_path(self, monkeypatch, tmp_path):
        """Default cache lives under ~/.cache/astrojax."""
        monkeypatch.delenv("ASTROJAX_CACHE", raising=False)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        result = get_cache_dir()
        assert result == tmp_path / ".cache" / "astrojax"
        assert result.is_dir()

    def test_env_override(self, monkeypatch, tmp_path):
        """ASTROJAX_CACHE env var overrides the default."""
        custom = tmp_path / "custom_cache"
        monkeypatch.setenv("ASTROJAX_CACHE", str(custom))
        result = get_cache_dir()
        assert result == custom
        assert result.is_dir()

    def test_subdirectory_creation(self, monkeypatch, tmp_path):
        """Subdirectory is appended and created."""
        monkeypatch.setenv("ASTROJAX_CACHE", str(tmp_path / "cache"))
        result = get_cache_dir("my_sub")
        assert result == tmp_path / "cache" / "my_sub"
        assert result.is_dir()

    def test_idempotent(self, monkeypatch, tmp_path):
        """Calling twice returns the same path without error."""
        monkeypatch.setenv("ASTROJAX_CACHE", str(tmp_path / "cache"))
        first = get_cache_dir("sub")
        second = get_cache_dir("sub")
        assert first == second

    def test_nested_subdirectory(self, monkeypatch, tmp_path):
        """Nested subdirectory paths are created."""
        monkeypatch.setenv("ASTROJAX_CACHE", str(tmp_path / "cache"))
        result = get_cache_dir("a/b/c")
        assert result == tmp_path / "cache" / "a" / "b" / "c"
        assert result.is_dir()


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


class TestConvenienceWrappers:
    """Tests for get_eop_cache_dir and get_datasets_cache_dir."""

    def test_eop_cache_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ASTROJAX_CACHE", str(tmp_path))
        result = get_eop_cache_dir()
        assert result == tmp_path / "eop"
        assert result.is_dir()

    def test_datasets_cache_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ASTROJAX_CACHE", str(tmp_path))
        result = get_datasets_cache_dir()
        assert result == tmp_path / "datasets"
        assert result.is_dir()

    def test_env_override_propagates(self, monkeypatch, tmp_path):
        """Env override flows through convenience wrappers."""
        custom = tmp_path / "my_cache"
        monkeypatch.setenv("ASTROJAX_CACHE", str(custom))
        eop = get_eop_cache_dir()
        datasets = get_datasets_cache_dir()
        assert eop.parent == custom
        assert datasets.parent == custom


# ---------------------------------------------------------------------------
# file_age_seconds / file_age_days
# ---------------------------------------------------------------------------


class TestFileAge:
    """Tests for file_age_seconds and file_age_days."""

    def test_recent_file(self, tmp_path):
        """A just-created file should be very young."""
        f = tmp_path / "recent.txt"
        f.write_text("hello")
        age = file_age_seconds(f)
        assert 0 <= age < 5

    def test_old_file(self, tmp_path):
        """Backdating mtime makes the file appear old."""
        f = tmp_path / "old.txt"
        f.write_text("hello")
        old_time = time.time() - 3600  # 1 hour ago
        os.utime(f, (old_time, old_time))
        age = file_age_seconds(f)
        assert 3590 < age < 3700

    def test_age_days(self, tmp_path):
        """file_age_days returns age in days."""
        f = tmp_path / "day_old.txt"
        f.write_text("hello")
        old_time = time.time() - 86400  # 1 day ago
        os.utime(f, (old_time, old_time))
        days = file_age_days(f)
        assert 0.99 < days < 1.1

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            file_age_seconds(tmp_path / "nope.txt")

    def test_string_path(self, tmp_path):
        """Accepts string paths as well as Path objects."""
        f = tmp_path / "str_path.txt"
        f.write_text("hello")
        age = file_age_seconds(str(f))
        assert age >= 0


# ---------------------------------------------------------------------------
# is_file_stale
# ---------------------------------------------------------------------------


class TestIsFileStale:
    """Tests for is_file_stale."""

    def test_missing_file_is_stale(self, tmp_path):
        """A nonexistent file is always stale."""
        assert is_file_stale(tmp_path / "missing.txt", max_age_seconds=9999) is True

    def test_old_file_is_stale(self, tmp_path):
        """A file older than max_age is stale."""
        f = tmp_path / "old.txt"
        f.write_text("data")
        old_time = time.time() - 7200
        os.utime(f, (old_time, old_time))
        assert is_file_stale(f, max_age_seconds=3600) is True

    def test_fresh_file_not_stale(self, tmp_path):
        """A recently created file is not stale."""
        f = tmp_path / "fresh.txt"
        f.write_text("data")
        assert is_file_stale(f, max_age_seconds=3600) is False


# ---------------------------------------------------------------------------
# file_hash
# ---------------------------------------------------------------------------


class TestFileHash:
    """Tests for file_hash."""

    def test_sha256_known_digest(self, tmp_path):
        """SHA-256 of known content matches expected digest."""
        f = tmp_path / "known.txt"
        content = b"hello world"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert file_hash(f) == expected

    def test_md5_known_digest(self, tmp_path):
        """MD5 of known content matches expected digest."""
        f = tmp_path / "known_md5.txt"
        content = b"test data"
        f.write_bytes(content)
        expected = hashlib.md5(content).hexdigest()
        assert file_hash(f, algorithm="md5") == expected

    def test_empty_file(self, tmp_path):
        """Hash of an empty file is the algorithm's null digest."""
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert file_hash(f) == expected

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            file_hash(tmp_path / "nope.bin")

    def test_unsupported_algorithm_raises(self, tmp_path):
        """ValueError for unknown algorithm name."""
        f = tmp_path / "algo.txt"
        f.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            file_hash(f, algorithm="not_a_real_algo")

    def test_chunk_size_invariance(self, tmp_path):
        """Hash is the same regardless of chunk size."""
        f = tmp_path / "chunks.bin"
        data = os.urandom(1024 * 100)  # 100 KiB
        f.write_bytes(data)
        h1 = file_hash(f, chunk_size=128)
        h2 = file_hash(f, chunk_size=65536)
        h3 = file_hash(f, chunk_size=1024 * 1024)
        assert h1 == h2 == h3

    def test_large_file_chunked(self, tmp_path):
        """A file larger than chunk_size still hashes correctly."""
        f = tmp_path / "large.bin"
        data = os.urandom(256 * 1024)  # 256 KiB
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert file_hash(f, chunk_size=1024) == expected
