"""Download CSSI Space Weather files from CelesTrak.

Provides a helper to fetch the latest space weather data file.
Network errors are propagated to the caller so that higher-level code
(e.g. :func:`load_cached_sw`) can decide on fallback behaviour.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

CELESTRAK_SW_URL: str = "https://celestrak.org/SpaceData/SW-All.txt"
"""Default URL for the CelesTrak CSSI Space Weather file (full history)."""

_SW_FILENAME: str = "sw19571001.txt"
"""Canonical filename used for cached space weather data."""

_DEFAULT_TIMEOUT: float = 120.0
"""Default HTTP timeout in seconds."""


def download_sw_file(
    filepath: str | Path,
    *,
    url: str = CELESTRAK_SW_URL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Path:
    """Download a CSSI space weather file to *filepath*.

    Creates parent directories if they do not exist. On success the
    downloaded text is written to *filepath* and the resolved path is
    returned.

    Args:
        filepath: Destination path for the downloaded file.
        url: URL to fetch. Defaults to :data:`CELESTRAK_SW_URL`.
        timeout: HTTP timeout in seconds. Defaults to 120.

    Returns:
        Resolved :class:`~pathlib.Path` to the written file.

    Raises:
        httpx.HTTPStatusError: If the server returns a non-2xx status.
        httpx.TransportError: On network-level failures (DNS, timeout, etc.).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading space weather data from %s", url)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    filepath.write_text(response.text, encoding="utf-8")
    logger.info("Space weather data written to %s", filepath)
    return filepath.resolve()
