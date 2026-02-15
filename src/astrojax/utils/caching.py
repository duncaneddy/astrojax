"""Filesystem cache directory management and file utilities.

Provides helpers for locating and managing the astrojax cache directory,
checking file freshness, and computing file hashes.  These are pure-Python
utilities with no JAX dependency.

The cache root is determined by the ``ASTROJAX_CACHE`` environment variable.
If unset, it defaults to ``~/.cache/astrojax``.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

_ENV_VAR = "ASTROJAX_CACHE"
_DEFAULT_SUBDIR = ".cache/astrojax"


def get_cache_dir(subdirectory: str | None = None) -> Path:
    """Return the astrojax cache directory, creating it if needed.

    The root is ``$ASTROJAX_CACHE`` if set, otherwise ``~/.cache/astrojax``.
    An optional *subdirectory* is appended and also created.

    Args:
        subdirectory: Optional subdirectory to append (e.g. ``"eop"``).

    Returns:
        Resolved :class:`~pathlib.Path` to the cache directory.
    """
    env = os.environ.get(_ENV_VAR)
    if env is not None:
        root = Path(env)
    else:
        root = Path.home() / _DEFAULT_SUBDIR

    if subdirectory is not None:
        root = root / subdirectory

    root.mkdir(parents=True, exist_ok=True)
    return root


def get_eop_cache_dir() -> Path:
    """Return the EOP cache directory (``<cache>/eop``).

    Returns:
        Path to the EOP cache directory.
    """
    return get_cache_dir("eop")


def get_sw_cache_dir() -> Path:
    """Return the space weather cache directory (``<cache>/space_weather``).

    Returns:
        Path to the space weather cache directory.
    """
    return get_cache_dir("space_weather")


def get_datasets_cache_dir() -> Path:
    """Return the datasets cache directory (``<cache>/datasets``).

    Returns:
        Path to the datasets cache directory.
    """
    return get_cache_dir("datasets")


def file_age_seconds(filepath: str | Path) -> float:
    """Return the age of *filepath* in seconds since last modification.

    Args:
        filepath: Path to the file.

    Returns:
        Seconds elapsed since the file was last modified.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No such file: '{filepath}'")
    return max(0.0, time.time() - filepath.stat().st_mtime)


def file_age_days(filepath: str | Path) -> float:
    """Return the age of *filepath* in days since last modification.

    Args:
        filepath: Path to the file.

    Returns:
        Days elapsed since the file was last modified.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    return file_age_seconds(filepath) / 86400.0


def is_file_stale(filepath: str | Path, max_age_seconds: float) -> bool:
    """Check whether *filepath* is missing or older than *max_age_seconds*.

    Returns ``True`` when the file should be refreshed — either because it
    does not exist or because it was last modified more than
    *max_age_seconds* ago.

    Args:
        filepath: Path to the file.
        max_age_seconds: Maximum acceptable age in seconds.

    Returns:
        ``True`` if the file is missing or stale.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return True
    return file_age_seconds(filepath) > max_age_seconds


def file_hash(
    filepath: str | Path,
    algorithm: str = "sha256",
    chunk_size: int = 65536,
) -> str:
    """Compute the hex digest of *filepath* using the given hash algorithm.

    Reads in chunks so that arbitrarily large files can be hashed without
    loading them entirely into memory.

    Args:
        filepath: Path to the file.
        algorithm: Hash algorithm name accepted by :mod:`hashlib`
            (e.g. ``"sha256"``, ``"md5"``).
        chunk_size: Number of bytes per read chunk.

    Returns:
        Lowercase hex digest string.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If *algorithm* is not supported by :mod:`hashlib`.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No such file: '{filepath}'")

    try:
        h = hashlib.new(algorithm)
    except ValueError as err:
        raise ValueError(
            f"Unsupported hash algorithm: '{algorithm}'. "
            f"Available: {sorted(hashlib.algorithms_available)}"
        ) from err

    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
