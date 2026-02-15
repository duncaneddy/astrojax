"""Download Minor Planet Center (MPC) asteroid orbit catalog.

Provides a helper to fetch the ``mpcorb_extended.json.gz`` file from the
MPC data centre.  Network errors are propagated to the caller so that
higher-level code (e.g. :func:`load_mpc_asteroids`) can decide on
fallback behaviour.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

MPC_URL: str = "https://minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz"
"""Default URL for the MPC extended orbit catalog (gzipped JSON)."""

_FILENAME: str = "mpcorb_extended.json.gz"
"""Canonical filename used for cached MPC data."""

_DEFAULT_TIMEOUT: float = 300.0
"""Default HTTP timeout in seconds (large file, so allow longer)."""


def download_mpc_file(
    filepath: str | Path,
    *,
    url: str = MPC_URL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Path:
    """Download the MPC extended orbit catalog to *filepath*.

    Creates parent directories if they do not exist.  On success the
    downloaded binary data is written to *filepath* and the resolved
    path is returned.

    Args:
        filepath: Destination path for the downloaded file.
        url: URL to fetch.  Defaults to :data:`MPC_URL`.
        timeout: HTTP timeout in seconds.  Defaults to 300.

    Returns:
        Resolved :class:`~pathlib.Path` to the written file.

    Raises:
        httpx.HTTPStatusError: If the server returns a non-2xx status.
        httpx.TransportError: On network-level failures (DNS, timeout, etc.).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MPC orbit catalog from %s", url)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()

    filepath.write_bytes(response.content)
    logger.info("MPC data written to %s", filepath)
    return filepath.resolve()
