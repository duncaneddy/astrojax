"""Download IERS Earth Orientation Parameter files.

Provides a helper to fetch the latest ``finals.all.iau2000.txt`` from
the IERS data centre.  Network errors are propagated to the caller so
that higher-level code (e.g. :func:`load_cached_eop`) can decide on
fallback behaviour.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

IERS_STANDARD_URL: str = (
    "https://datacenter.iers.org/data/latestVersion/finals.all.iau2000.txt"
)
"""Default URL for the IERS Standard Bulletin A finals file."""

_STANDARD_FILENAME: str = "finals.all.iau2000.txt"
"""Canonical filename used for cached EOP data."""

_DEFAULT_TIMEOUT: float = 120.0
"""Default HTTP timeout in seconds."""


def download_standard_eop_file(
    filepath: str | Path,
    *,
    url: str = IERS_STANDARD_URL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Path:
    """Download an IERS standard EOP file to *filepath*.

    Creates parent directories if they do not exist.  On success the
    downloaded text is written to *filepath* and the resolved path is
    returned.

    Args:
        filepath: Destination path for the downloaded file.
        url: URL to fetch.  Defaults to :data:`IERS_STANDARD_URL`.
        timeout: HTTP timeout in seconds.  Defaults to 120.

    Returns:
        Resolved :class:`~pathlib.Path` to the written file.

    Raises:
        httpx.HTTPStatusError: If the server returns a non-2xx status.
        httpx.TransportError: On network-level failures (DNS, timeout, etc.).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading EOP data from %s", url)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    filepath.write_text(response.text, encoding="utf-8")
    logger.info("EOP data written to %s", filepath)
    return filepath.resolve()
