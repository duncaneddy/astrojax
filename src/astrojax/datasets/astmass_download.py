"""Download the SBN Archive asteroid masses compilation dataset.

Provides a helper to fetch the ``compil.ast.masses.zip`` file from the
SBN Archive.  Network errors are propagated to the caller so that
higher-level code (e.g. :func:`load_asteroid_masses`) can decide on
fallback behaviour.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

ASTMASS_URL: str = "https://sbnarchive.psi.edu/pds4/non_mission/compil.ast.masses.zip"
"""Default URL for the SBN Archive asteroid masses compilation (ZIP)."""

_FILENAME: str = "compil.ast.masses.zip"
"""Canonical filename used for cached asteroid masses data."""

_DEFAULT_TIMEOUT: float = 120.0
"""Default HTTP timeout in seconds."""


def download_astmass_file(
    filepath: str | Path,
    *,
    url: str = ASTMASS_URL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Path:
    """Download the SBN Archive asteroid masses dataset to *filepath*.

    Creates parent directories if they do not exist.  On success the
    downloaded binary data is written to *filepath* and the resolved
    path is returned.

    Args:
        filepath: Destination path for the downloaded file.
        url: URL to fetch.  Defaults to :data:`ASTMASS_URL`.
        timeout: HTTP timeout in seconds.  Defaults to 120.

    Returns:
        Resolved :class:`~pathlib.Path` to the written file.

    Raises:
        httpx.HTTPStatusError: If the server returns a non-2xx status.
        httpx.TransportError: On network-level failures (DNS, timeout, etc.).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading asteroid masses dataset from %s", url)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    filepath.write_bytes(response.content)
    logger.info("Asteroid masses data written to %s", filepath)
    return filepath.resolve()
