"""Factory functions for loading the DAMIT asteroid shape model dataset.

Provides convenience functions for loading DAMIT data as Polars
DataFrames and JAX arrays:

- :func:`load_damit_asteroids`: Load asteroid identity table from cache.
- :func:`load_damit_models`: Load model/spin parameter table from cache.
- :func:`get_damit_spin`: Look up spin parameters for a specific asteroid.
- :func:`get_damit_shape`: Load shape mesh for a specific model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from jax import Array

from astrojax.datasets.damit_download import (
    _EXTRACTED_MARKER,
    _FILENAME,
    download_damit_file,
    extract_damit_archive,
)
from astrojax.datasets.damit_parsers import (
    _find_extracted_prefix,
    _shape_index,
    load_shape_for_model,
    parse_damit_asteroids_table,
    parse_damit_models_table,
)
from astrojax.utils.caching import get_datasets_cache_dir, is_file_stale

logger = logging.getLogger(__name__)

_DEFAULT_MAX_AGE_DAYS: float = 30.0
"""Default maximum age for cached DAMIT data in days."""

_DEFAULT_MAX_AGE_SECONDS: float = _DEFAULT_MAX_AGE_DAYS * 86400.0
"""Default maximum age for cached DAMIT data in seconds."""

_EXTRACTED_DIR_NAME: str = "extracted"
"""Name of the subdirectory used for extracted archive contents."""


def _is_extraction_stale(tar_path: Path, extract_dir: Path) -> bool:
    """Check whether the extraction directory is missing or stale.

    Compares the ``tar.gz`` mtime against the ``.extracted`` marker
    file mtime.  Returns ``True`` if extraction is needed.

    Args:
        tar_path: Path to the DAMIT tar.gz archive.
        extract_dir: Path to the extraction directory.

    Returns:
        ``True`` if the extraction directory is missing, the marker
        file is absent, or the tar.gz is newer than the marker.
    """
    marker = extract_dir / _EXTRACTED_MARKER
    if not marker.exists():
        return True
    return tar_path.stat().st_mtime > marker.stat().st_mtime


def _ensure_damit_data(
    filepath: Path,
    max_age_days: float,
) -> tuple[Path, Path]:
    """Ensure the DAMIT tar.gz and extracted directory are ready.

    Downloads the archive if it is missing or stale, then extracts it
    to a sibling ``extracted/`` directory if the extraction is missing
    or the archive has been updated.

    Args:
        filepath: Path to the cached tar.gz file.
        max_age_days: Maximum acceptable age in days.

    Returns:
        Tuple of ``(tar_path, extracted_dir)`` where *extracted_dir*
        is the root of the extracted archive contents.

    Raises:
        RuntimeError: If the download fails and no cached file exists.
    """
    max_age_seconds = max_age_days * 86400.0
    freshly_downloaded = False

    if is_file_stale(filepath, max_age_seconds):
        try:
            download_damit_file(filepath)
            freshly_downloaded = True
        except Exception as exc:
            if filepath.exists():
                logger.warning(
                    "Failed to download fresh DAMIT data; using existing cache.",
                    exc_info=True,
                )
            else:
                logger.error(
                    "Failed to download DAMIT data and no cached file exists.",
                    exc_info=True,
                )
                raise RuntimeError(
                    "Failed to download DAMIT data and no cached file exists at "
                    f"{filepath}. Check your network connection."
                ) from exc

    # Determine extraction directory (sibling to the tar.gz)
    extract_dir = filepath.parent / _EXTRACTED_DIR_NAME

    # Extract if needed: freshly downloaded, or extraction is stale/missing
    if freshly_downloaded or _is_extraction_stale(filepath, extract_dir):
        extract_damit_archive(filepath, extract_dir)
        # Invalidate cached lookups so they reload from the fresh extraction
        _find_extracted_prefix.cache_clear()
        _shape_index.invalidate()

    return filepath, extract_dir


def _resolve_filepath(filepath: str | Path | None) -> Path:
    """Resolve the DAMIT archive filepath, using defaults if None.

    Args:
        filepath: User-provided path, or ``None`` for default cache.

    Returns:
        Resolved :class:`~pathlib.Path`.
    """
    if filepath is None:
        return get_datasets_cache_dir() / "damit" / _FILENAME
    return Path(filepath)


def load_damit_asteroids(
    filepath: str | Path | None = None,
    *,
    max_age_days: float = _DEFAULT_MAX_AGE_DAYS,
) -> pl.DataFrame:
    """Load the DAMIT asteroids table from a local cache, downloading when stale.

    Checks whether the cached tar.gz at *filepath* exists and is younger
    than *max_age_days*.  If the file is missing or stale, a fresh copy
    is downloaded from the DAMIT server.

    Args:
        filepath: Path to the cached tar.gz file.  When ``None`` (the
            default), uses ``<cache_dir>/datasets/damit/damit_complete.tar.gz``.
        max_age_days: Maximum acceptable age of the cached file in days.
            Defaults to 30.

    Returns:
        Polars DataFrame with columns: ``id``, ``number``, ``name``,
        ``designation``, ``comment``.

    Raises:
        RuntimeError: If the download fails and no cached file exists.

    Examples:
        ```python
        from astrojax.datasets import load_damit_asteroids
        df = load_damit_asteroids()
        print(df.shape)
        ```
    """
    filepath = _resolve_filepath(filepath)
    tar_path, extracted_dir = _ensure_damit_data(filepath, max_age_days)
    return parse_damit_asteroids_table(tar_path, extracted_dir=extracted_dir)


def load_damit_models(
    filepath: str | Path | None = None,
    *,
    max_age_days: float = _DEFAULT_MAX_AGE_DAYS,
) -> pl.DataFrame:
    """Load the DAMIT asteroid models table from a local cache, downloading when stale.

    The models table contains spin parameters (lambda, beta, period,
    yorp, jd0, phi0) and physical properties (diameter, albedo, thermal
    inertia) for all principal-axis rotator models in DAMIT.

    Args:
        filepath: Path to the cached tar.gz file.  When ``None`` (the
            default), uses ``<cache_dir>/datasets/damit/damit_complete.tar.gz``.
        max_age_days: Maximum acceptable age of the cached file in days.
            Defaults to 30.

    Returns:
        Polars DataFrame with spin parameter and physical property
        columns for all DAMIT models.

    Raises:
        RuntimeError: If the download fails and no cached file exists.

    Examples:
        ```python
        from astrojax.datasets import load_damit_models
        df = load_damit_models()
        print(df.columns)
        ```
    """
    filepath = _resolve_filepath(filepath)
    tar_path, extracted_dir = _ensure_damit_data(filepath, max_age_days)
    return parse_damit_models_table(tar_path, extracted_dir=extracted_dir)


def get_damit_spin(
    models_df: pl.DataFrame,
    asteroid_id: int,
    *,
    model_index: int = 0,
) -> dict:
    """Look up spin parameters for an asteroid from a pre-loaded models DataFrame.

    Filters the models table by ``asteroid_id`` and returns the row at
    *model_index* as a dictionary.  Multiple models may exist for a
    single asteroid; use *model_index* to select among them.

    Args:
        models_df: Polars DataFrame as returned by :func:`load_damit_models`.
        asteroid_id: DAMIT asteroid ID (from the ``asteroid_id`` column).
        model_index: Zero-based index among models for this asteroid.
            Defaults to 0 (first/best model).

    Returns:
        Dictionary with all columns from the matched row, including
        spin parameters ``lambda``, ``beta``, ``period``, ``yorp``,
        ``jd0``, ``phi0``.

    Raises:
        KeyError: If no models exist for *asteroid_id* or *model_index*
            is out of range.

    Examples:
        ```python
        from astrojax.datasets import load_damit_models, get_damit_spin
        models = load_damit_models()
        spin = get_damit_spin(models, asteroid_id=1)
        print(spin["lambda"], spin["beta"], spin["period"])
        ```
    """
    subset = models_df.filter(pl.col("asteroid_id") == asteroid_id)

    if len(subset) == 0:
        raise KeyError(f"No DAMIT models found for asteroid_id={asteroid_id}")

    if model_index < 0 or model_index >= len(subset):
        raise KeyError(
            f"model_index={model_index} out of range for asteroid_id={asteroid_id} "
            f"({len(subset)} model(s) available)"
        )

    return subset.row(model_index, named=True)


def get_damit_shape(
    filepath: str | Path | None = None,
    *,
    model_id: int,
    max_age_days: float = _DEFAULT_MAX_AGE_DAYS,
) -> tuple[Array, Array]:
    """Load the shape mesh for a specific DAMIT model.

    Downloads the DAMIT archive if needed, extracts it, then reads
    ``model_{model_id}/shape.txt`` directly from disk.

    Args:
        filepath: Path to the cached tar.gz file.  When ``None``,
            uses the default cache location.
        model_id: DAMIT model ID (from the ``id`` column of the
            models table).
        max_age_days: Maximum acceptable age of the cached file in days.
            Defaults to 30.

    Returns:
        Tuple of ``(vertices, facets)`` where *vertices* is a
        ``(N, 3)`` float32 JAX array and *facets* is a ``(M, 3)``
        int32 JAX array with 0-indexed vertex references.

    Raises:
        RuntimeError: If the download fails and no cached file exists.
        KeyError: If no shape file exists for *model_id*.

    Examples:
        ```python
        from astrojax.datasets import get_damit_shape
        vertices, facets = get_damit_shape(model_id=42)
        print(vertices.shape, facets.shape)
        ```
    """
    filepath = _resolve_filepath(filepath)
    tar_path, extracted_dir = _ensure_damit_data(filepath, max_age_days)
    return load_shape_for_model(tar_path, model_id, extracted_dir=extracted_dir)
