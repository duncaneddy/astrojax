"""Factory functions for loading the JPL SBDB asteroid dataset.

Provides convenience functions for loading the SBDB asteroid data as a
Polars DataFrame:

- :func:`load_sbdb_asteroids`: Load from cache, downloading fresh data
  when stale.
- :func:`load_sbdb_from_file`: Load from an arbitrary file path.
- :func:`get_sbdb_asteroid_ephemeris`: Look up a single asteroid's
  orbital elements and physical properties.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from astrojax.datasets.sbdb_download import _FILENAME, download_sbdb_file
from astrojax.datasets.sbdb_parsers import load_sbdb_csv_to_dataframe
from astrojax.utils.caching import get_datasets_cache_dir, is_file_stale

logger = logging.getLogger(__name__)

_DEFAULT_MAX_AGE_DAYS: float = 365.0
"""Default maximum age for cached SBDB data in days."""

_DEFAULT_MAX_AGE_SECONDS: float = _DEFAULT_MAX_AGE_DAYS * 86400.0
"""Default maximum age for cached SBDB data in seconds."""


def load_sbdb_from_file(filepath: str | Path) -> pl.DataFrame:
    """Load SBDB asteroid data from a local file.

    Args:
        filepath: Path to a ``sbdb_asteroid_data.csv`` file.

    Returns:
        Polars DataFrame with SBDB asteroid orbital elements and
        physical properties.

    Raises:
        FileNotFoundError: If the file does not exist.

    Examples:
        ```python
        from astrojax.datasets import load_sbdb_from_file
        df = load_sbdb_from_file("/path/to/sbdb_asteroid_data.csv")
        print(df.shape)
        ```
    """
    return load_sbdb_csv_to_dataframe(filepath)


def load_sbdb_asteroids(
    filepath: str | Path | None = None,
    *,
    max_age_days: float = _DEFAULT_MAX_AGE_DAYS,
) -> pl.DataFrame:
    """Load SBDB asteroid data from a local cache, downloading when stale.

    Checks whether the cached file at *filepath* exists and is younger than
    *max_age_days*.  If the file is missing or stale, a fresh copy is
    downloaded from GitHub LFS.

    Args:
        filepath: Path to the cached CSV file.  When ``None`` (the default),
            uses ``<cache_dir>/datasets/sbdb/sbdb_asteroid_data.csv``.
        max_age_days: Maximum acceptable age of the cached file in days.
            Defaults to 365.

    Returns:
        Polars DataFrame with SBDB asteroid orbital elements and
        physical properties.

    Raises:
        RuntimeError: If the download fails and no cached file exists.

    Examples:
        ```python
        from astrojax.datasets import load_sbdb_asteroids
        df = load_sbdb_asteroids()
        print(df.shape)
        print(df.head())
        ```
    """
    if filepath is None:
        filepath = get_datasets_cache_dir() / "sbdb" / _FILENAME
    else:
        filepath = Path(filepath)

    max_age_seconds = max_age_days * 86400.0

    if is_file_stale(filepath, max_age_seconds):
        try:
            download_sbdb_file(filepath)
        except Exception as exc:
            if filepath.exists():
                logger.warning(
                    "Failed to download fresh SBDB data; using existing cache.",
                    exc_info=True,
                )
            else:
                logger.error(
                    "Failed to download SBDB data and no cached file exists.",
                    exc_info=True,
                )
                raise RuntimeError(
                    "Failed to download SBDB data and no cached file exists at "
                    f"{filepath}. Check your network connection."
                ) from exc

    return load_sbdb_csv_to_dataframe(filepath)


def get_sbdb_asteroid_ephemeris(df: pl.DataFrame, identifier: int | str) -> dict:
    """Look up an asteroid's orbital elements from the SBDB DataFrame.

    Searches by number (if *identifier* is an ``int`` or numeric string)
    or by name (if *identifier* is a non-numeric string).

    The returned dictionary is compatible with
    :func:`~astrojax.datasets.asteroid_state_ecliptic`:

    .. code-block:: python

        eph = get_sbdb_asteroid_ephemeris(df, 1)
        oe = jnp.array([eph["a"], eph["e"], eph["i"],
                         eph["node"], eph["peri"], eph["M"]])
        state = asteroid_state_ecliptic(eph["epoch_jd"], oe, target_jd)

    Args:
        df: Polars DataFrame as returned by :func:`load_sbdb_asteroids`.
        identifier: Asteroid number (int or numeric str) or name (str).

    Returns:
        Dictionary with keys: ``name``, ``full_name``, ``number``,
        ``epoch_jd``, ``a``, ``e``, ``i``, ``node``, ``peri``, ``M``,
        ``n``, ``diameter``, ``GM``.

    Raises:
        KeyError: If the asteroid is not found.

    Examples:
        ```python
        from astrojax.datasets import load_sbdb_asteroids, get_sbdb_asteroid_ephemeris
        df = load_sbdb_asteroids()
        ceres = get_sbdb_asteroid_ephemeris(df, 1)
        print(ceres["name"])  # "Ceres"
        ```
    """
    if isinstance(identifier, int) or (
        isinstance(identifier, str) and identifier.strip().isdigit()
    ):
        num_str = str(int(identifier)).strip()
        result = df.filter(pl.col("number").str.strip_chars() == num_str)
    else:
        result = df.filter(pl.col("name").str.strip_chars() == str(identifier).strip())

    if result.is_empty():
        raise KeyError(f"Asteroid not found: {identifier!r}")

    row = result.row(0, named=True)

    return {
        "name": row["name"],
        "full_name": row["full_name"],
        "number": row["number"],
        "epoch_jd": row["epoch_jd"],
        "a": row["a"],
        "e": row["e"],
        "i": row["i"],
        "node": row["node"],
        "peri": row["peri"],
        "M": row["M"],
        "n": row["n"],
        "diameter": row["diameter"],
        "GM": row["GM"],
    }
