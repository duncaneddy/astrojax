"""Factory functions for loading the MPC asteroid orbit catalog.

Provides convenience functions for loading the Minor Planet Center
extended orbit catalog as a Polars DataFrame:

- :func:`load_mpc_asteroids`: Load from cache, downloading fresh data
  when stale.
- :func:`load_mpc_from_file`: Load from an arbitrary file path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from astrojax.datasets._mpc_download import _FILENAME, download_mpc_file
from astrojax.datasets._mpc_parsers import load_mpc_json_to_dataframe
from astrojax.utils.caching import get_datasets_cache_dir, is_file_stale

logger = logging.getLogger(__name__)

_DEFAULT_MAX_AGE_DAYS: float = 7.0
"""Default maximum age for cached MPC data in days."""

_DEFAULT_MAX_AGE_SECONDS: float = _DEFAULT_MAX_AGE_DAYS * 86400.0
"""Default maximum age for cached MPC data in seconds."""


def load_mpc_from_file(filepath: str | Path) -> pl.DataFrame:
    """Load MPC asteroid data from a local file.

    Args:
        filepath: Path to an ``mpcorb_extended.json.gz`` file.

    Returns:
        Polars DataFrame with asteroid orbital elements.

    Raises:
        FileNotFoundError: If the file does not exist.

    Examples:
        ```python
        from astrojax.datasets import load_mpc_from_file
        df = load_mpc_from_file("/path/to/mpcorb_extended.json.gz")
        print(df.shape)
        ```
    """
    return load_mpc_json_to_dataframe(filepath)


def load_mpc_asteroids(
    filepath: str | Path | None = None,
    *,
    max_age_days: float = _DEFAULT_MAX_AGE_DAYS,
) -> pl.DataFrame:
    """Load the MPC asteroid catalog from a local cache, downloading when stale.

    Checks whether the cached file at *filepath* exists and is younger than
    *max_age_days*.  If the file is missing or stale, a fresh copy of
    ``mpcorb_extended.json.gz`` is downloaded from the MPC.

    Args:
        filepath: Path to the cached MPC file.  When ``None`` (the default),
            uses ``<cache_dir>/datasets/mpc/mpcorb_extended.json.gz``.
        max_age_days: Maximum acceptable age of the cached file in days.
            Defaults to 7.

    Returns:
        Polars DataFrame with asteroid orbital elements.

    Raises:
        RuntimeError: If the download fails and no cached file exists.

    Examples:
        ```python
        from astrojax.datasets import load_mpc_asteroids
        df = load_mpc_asteroids()
        print(df.shape)
        print(df.head())
        ```
    """
    if filepath is None:
        filepath = get_datasets_cache_dir() / "mpc" / _FILENAME
    else:
        filepath = Path(filepath)

    max_age_seconds = max_age_days * 86400.0

    if is_file_stale(filepath, max_age_seconds):
        try:
            download_mpc_file(filepath)
        except Exception as exc:
            if filepath.exists():
                logger.warning(
                    "Failed to download fresh MPC data; using existing cache.",
                    exc_info=True,
                )
            else:
                logger.error(
                    "Failed to download MPC data and no cached file exists.",
                    exc_info=True,
                )
                raise RuntimeError(
                    "Failed to download MPC data and no cached file exists at "
                    f"{filepath}. Check your network connection."
                ) from exc

    return load_mpc_json_to_dataframe(filepath)
