"""Download the DAMIT complete export archive.

Provides a helper to stream-download the large (~1.3 GB) DAMIT
complete export tar.gz file.  Uses streaming with 1 MB chunks to
avoid holding the entire file in memory, and writes to a temporary
file with atomic rename on completion.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DAMIT_URL: str = "https://damit.cuni.cz/projects/damit/exports/complete/latest"
"""Default URL for the DAMIT complete export (tar.gz)."""

_FILENAME: str = "damit_complete.tar.gz"
"""Canonical filename used for cached DAMIT data."""

_DEFAULT_TIMEOUT: float = 600.0
"""Default HTTP timeout in seconds (10 minutes for the large archive)."""

_CHUNK_SIZE: int = 1_048_576
"""Streaming chunk size in bytes (1 MB)."""


def download_damit_file(
    filepath: str | Path,
    *,
    url: str = DAMIT_URL,
    timeout: float = _DEFAULT_TIMEOUT,
) -> Path:
    """Stream-download the DAMIT complete export to *filepath*.

    Creates parent directories if they do not exist.  The file is first
    written to a ``.tmp`` sibling and atomically renamed on successful
    completion, preventing partial downloads from corrupting the cache.

    Args:
        filepath: Destination path for the downloaded archive.
        url: URL to fetch.  Defaults to :data:`DAMIT_URL`.
        timeout: HTTP timeout in seconds.  Defaults to 600 (10 minutes).

    Returns:
        Resolved :class:`~pathlib.Path` to the written file.

    Raises:
        httpx.HTTPStatusError: If the server returns a non-2xx status.
        httpx.TransportError: On network-level failures (DNS, timeout, etc.).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(".tmp")

    logger.info("Downloading DAMIT complete export from %s", url)

    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        verify=False,
    ) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()

            total = 0
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                    f.write(chunk)
                    total += len(chunk)

    # Atomic rename on success
    tmp_path.rename(filepath)
    logger.info(
        "DAMIT archive written to %s (%.1f MB)",
        filepath,
        total / 1_048_576,
    )
    return filepath.resolve()
