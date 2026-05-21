"""Download the JPL Small Body Database (SBDB) asteroid dataset.

Provides a helper to fetch the SBDB asteroid data CSV file from a
GitHub LFS repository.  Network errors are propagated to the caller so
that higher-level code (e.g. :func:`load_sbdb_asteroids`) can decide on
fallback behaviour.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

SBDB_URL: str = (
    "https://github.com/duncaneddy/lfs/raw/main/asteroid_data/sbdb_asteroid_data_2026_02_15.csv"
)
"""Default URL for the JPL SBDB asteroid data CSV (GitHub LFS snapshot)."""

_FILENAME: str = "sbdb_asteroid_data.csv"
"""Canonical filename used for cached SBDB data."""

_DEFAULT_TIMEOUT: float = 120.0
"""Default HTTP timeout in seconds."""


def download_sbdb_file(
    filepath: str | Path,
    *,
    url: str = SBDB_URL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Path:
    """Download the JPL SBDB asteroid dataset to *filepath*.

    Creates parent directories if they do not exist.  On success the
    downloaded binary data is written to *filepath* and the resolved
    path is returned.

    Args:
        filepath: Destination path for the downloaded file.
        url: URL to fetch.  Defaults to :data:`SBDB_URL`.
        timeout: HTTP timeout in seconds.  Defaults to 120.

    Returns:
        Resolved :class:`~pathlib.Path` to the written file.

    Raises:
        httpx.HTTPStatusError: If the server returns a non-2xx status.
        httpx.TransportError: On network-level failures (DNS, timeout, etc.).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading SBDB asteroid dataset from %s", url)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    filepath.write_bytes(response.content)
    logger.info("SBDB asteroid data written to %s", filepath)
    return filepath.resolve()
