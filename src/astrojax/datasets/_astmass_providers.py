"""Factory functions for loading the SBN Archive asteroid masses dataset.

Provides convenience functions for loading the asteroid masses compilation
as a Polars DataFrame:

- :func:`load_asteroid_masses`: Load from cache, downloading fresh data
  when stale.
- :func:`load_astmass_from_file`: Load from an arbitrary file path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from astrojax.datasets._astmass_download import _FILENAME, download_astmass_file
from astrojax.datasets._astmass_parsers import load_astmass_tab_to_dataframe
from astrojax.utils.caching import get_datasets_cache_dir, is_file_stale

logger = logging.getLogger(__name__)

_DEFAULT_MAX_AGE_DAYS: float = 30.0
"""Default maximum age for cached asteroid masses data in days."""

_DEFAULT_MAX_AGE_SECONDS: float = _DEFAULT_MAX_AGE_DAYS * 86400.0
"""Default maximum age for cached asteroid masses data in seconds."""


def load_astmass_from_file(filepath: str | Path) -> pl.DataFrame:
    """Load asteroid masses data from a local file.

    Args:
        filepath: Path to a ``compil.ast.masses.zip`` file.

    Returns:
        Polars DataFrame with asteroid mass, density, and shape data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the ZIP does not contain ``astmass12.tab``.

    Examples:
        ```python
        from astrojax.datasets import load_astmass_from_file
        df = load_astmass_from_file("/path/to/compil.ast.masses.zip")
        print(df.shape)
        ```
    """
    return load_astmass_tab_to_dataframe(filepath)


def load_asteroid_masses(
    filepath: str | Path | None = None,
    *,
    max_age_days: float = _DEFAULT_MAX_AGE_DAYS,
) -> pl.DataFrame:
    """Load asteroid masses from a local cache, downloading when stale.

    Checks whether the cached file at *filepath* exists and is younger than
    *max_age_days*.  If the file is missing or stale, a fresh copy of
    ``compil.ast.masses.zip`` is downloaded from the SBN Archive.

    Args:
        filepath: Path to the cached ZIP file.  When ``None`` (the default),
            uses ``<cache_dir>/datasets/astmass/compil.ast.masses.zip``.
        max_age_days: Maximum acceptable age of the cached file in days.
            Defaults to 30.

    Returns:
        Polars DataFrame with asteroid mass, density, and shape data.

    Raises:
        RuntimeError: If the download fails and no cached file exists.

    Examples:
        ```python
        from astrojax.datasets import load_asteroid_masses
        df = load_asteroid_masses()
        print(df.shape)
        print(df.head())
        ```
    """
    if filepath is None:
        filepath = get_datasets_cache_dir() / "astmass" / _FILENAME
    else:
        filepath = Path(filepath)

    max_age_seconds = max_age_days * 86400.0

    if is_file_stale(filepath, max_age_seconds):
        try:
            download_astmass_file(filepath)
        except Exception as exc:
            if filepath.exists():
                logger.warning(
                    "Failed to download fresh asteroid masses data; using existing cache.",
                    exc_info=True,
                )
            else:
                logger.error(
                    "Failed to download asteroid masses data and no cached file exists.",
                    exc_info=True,
                )
                raise RuntimeError(
                    "Failed to download asteroid masses data and no cached file exists at "
                    f"{filepath}. Check your network connection."
                ) from exc

    return load_astmass_tab_to_dataframe(filepath)
