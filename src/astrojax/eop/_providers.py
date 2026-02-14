"""Factory functions for creating EOPData instances.

Provides convenience constructors for common EOP configurations:

- :func:`static_eop`: Constant EOP values (useful for testing or when
  specific values are known).
- :func:`zero_eop`: All-zero EOP (equivalent to ignoring Earth orientation
  corrections).
- :func:`load_eop_from_file`: Load from IERS standard format file.
- :func:`load_eop_from_standard_file`: Alias for :func:`load_eop_from_file`.
- :func:`load_default_eop`: Load bundled ``finals.all.iau2000.txt`` data.
- :func:`load_cached_eop`: Load from a local cache, downloading fresh data
  from IERS when stale.
"""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path

import jax.numpy as jnp

from astrojax.config import get_dtype
from astrojax.eop._download import _STANDARD_FILENAME, download_standard_eop_file
from astrojax.eop._parsers import parse_standard_file
from astrojax.eop._types import EOPData
from astrojax.utils.caching import get_eop_cache_dir, is_file_stale

logger = logging.getLogger(__name__)

_DEFAULT_MAX_AGE_DAYS: float = 7.0
"""Default maximum age for cached EOP data in days."""

_DEFAULT_MAX_AGE_SECONDS: float = _DEFAULT_MAX_AGE_DAYS * 86400.0
"""Default maximum age for cached EOP data in seconds."""


def static_eop(
    pm_x: float = 0.0,
    pm_y: float = 0.0,
    ut1_utc: float = 0.0,
    dX: float = 0.0,
    dY: float = 0.0,
    lod: float = 0.0,
    mjd_min: float = 0.0,
    mjd_max: float = 99999.0,
) -> EOPData:
    """Create an EOPData with constant values across the full MJD range.

    Useful for testing or when specific constant EOP values are known.
    The resulting dataset contains two points (at mjd_min and mjd_max)
    with identical values, so interpolation returns the constant everywhere.

    Args:
        pm_x: Polar motion x-component [rad]. Default: 0.0.
        pm_y: Polar motion y-component [rad]. Default: 0.0.
        ut1_utc: UT1-UTC offset [seconds]. Default: 0.0.
        dX: Celestial pole offset X [rad]. Default: 0.0.
        dY: Celestial pole offset Y [rad]. Default: 0.0.
        lod: Length of day excess [seconds]. Default: 0.0.
        mjd_min: Start of the valid MJD range. Default: 0.0.
        mjd_max: End of the valid MJD range. Default: 99999.0.

    Returns:
        EOPData with constant values.

    Examples:
        ```python
        from astrojax.eop import static_eop, get_ut1_utc
        eop = static_eop(ut1_utc=0.1)
        val = get_ut1_utc(eop, 59569.0)  # returns ~0.1
        ```
    """
    dtype = get_dtype()
    mjd_arr = jnp.array([mjd_min, mjd_max], dtype=dtype)
    return EOPData(
        mjd=mjd_arr,
        pm_x=jnp.array([pm_x, pm_x], dtype=dtype),
        pm_y=jnp.array([pm_y, pm_y], dtype=dtype),
        ut1_utc=jnp.array([ut1_utc, ut1_utc], dtype=dtype),
        dX=jnp.array([dX, dX], dtype=dtype),
        dY=jnp.array([dY, dY], dtype=dtype),
        lod=jnp.array([lod, lod], dtype=dtype),
        mjd_min=jnp.array(mjd_min, dtype=dtype),
        mjd_max=jnp.array(mjd_max, dtype=dtype),
        mjd_last_lod=jnp.array(mjd_max, dtype=dtype),
        mjd_last_dxdy=jnp.array(mjd_max, dtype=dtype),
    )


def zero_eop() -> EOPData:
    """Create an EOPData with all-zero values.

    Equivalent to ignoring Earth orientation corrections entirely.
    This is the simplest provider for cases where EOP corrections
    are not needed.

    Returns:
        EOPData with all values set to zero.

    Examples:
        ```python
        from astrojax.eop import zero_eop, get_ut1_utc
        eop = zero_eop()
        val = get_ut1_utc(eop, 59569.0)  # returns 0.0
        ```
    """
    return static_eop()


def load_eop_from_file(filepath: str | Path) -> EOPData:
    """Load EOP data from an IERS standard format file.

    Parses the file, converts lists to JAX arrays, and computes metadata
    scalars (mjd_min, mjd_max, last valid LOD/dXdY MJDs).

    Args:
        filepath: Path to an IERS standard format file
            (e.g. ``finals.all.iau2000.txt``).

    Returns:
        EOPData ready for JIT-compatible lookups.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no valid EOP data is found.

    Examples:
        ```python
        from astrojax.eop import load_eop_from_file, get_ut1_utc
        eop = load_eop_from_file("path/to/finals.all.iau2000.txt")
        val = get_ut1_utc(eop, 59569.0)
        ```
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"EOP file not found: {filepath}")

    mjds, pm_xs, pm_ys, ut1_utcs, lods, dXs, dYs = parse_standard_file(str(filepath))

    dtype = jnp.float64

    # Convert to JAX arrays for array construction
    mjd_np = jnp.array(mjds, dtype=dtype)
    lod_np = jnp.array(lods, dtype=dtype)
    dX_np = jnp.array(dXs, dtype=dtype)
    dY_np = jnp.array(dYs, dtype=dtype)

    # Find last valid MJD for optional fields
    lod_valid = ~jnp.isnan(lod_np)
    dxdy_valid = ~jnp.isnan(dX_np) & ~jnp.isnan(dY_np)

    mjd_last_lod = float(mjd_np[lod_valid][-1]) if jnp.any(lod_valid) else float(mjd_np[0])
    mjd_last_dxdy = float(mjd_np[dxdy_valid][-1]) if jnp.any(dxdy_valid) else float(mjd_np[0])

    jdtype = get_dtype()
    return EOPData(
        mjd=jnp.array(mjd_np, dtype=jdtype),
        pm_x=jnp.array(pm_xs, dtype=jdtype),
        pm_y=jnp.array(pm_ys, dtype=jdtype),
        ut1_utc=jnp.array(ut1_utcs, dtype=jdtype),
        dX=jnp.array(dX_np, dtype=jdtype),
        dY=jnp.array(dY_np, dtype=jdtype),
        lod=jnp.array(lod_np, dtype=jdtype),
        mjd_min=jnp.array(mjd_np[0], dtype=jdtype),
        mjd_max=jnp.array(mjd_np[-1], dtype=jdtype),
        mjd_last_lod=jnp.array(mjd_last_lod, dtype=jdtype),
        mjd_last_dxdy=jnp.array(mjd_last_dxdy, dtype=jdtype),
    )


load_eop_from_standard_file = load_eop_from_file
"""Alias for :func:`load_eop_from_file`."""


def load_default_eop() -> EOPData:
    """Load the bundled default EOP data (``finals.all.iau2000.txt``).

    Uses ``importlib.resources`` to locate the data file bundled with the
    package, following the same pattern as ``GravityModel.from_type()``.

    Returns:
        EOPData loaded from the bundled ``finals.all.iau2000.txt``.

    Examples:
        ```python
        from astrojax.eop import load_default_eop, get_ut1_utc
        eop = load_default_eop()
        val = get_ut1_utc(eop, 59569.0)
        ```
    """
    data_pkg = importlib.resources.files("astrojax.data.eop")
    resource = data_pkg.joinpath("finals.all.iau2000.txt")
    with importlib.resources.as_file(resource) as path:
        return load_eop_from_file(path)


def load_cached_eop(
    filepath: str | Path | None = None,
    *,
    max_age_days: float = _DEFAULT_MAX_AGE_DAYS,
) -> EOPData:
    """Load EOP data from a local cache, downloading fresh data when stale.

    Checks whether the cached file at *filepath* exists and is younger than
    *max_age_days*.  If the file is missing or stale, a fresh copy of
    ``finals.all.iau2000.txt`` is downloaded from IERS.  If the download
    fails or the file cannot be parsed, the bundled default data is returned
    so this function never raises on network issues.

    Args:
        filepath: Path to the cached EOP file.  When ``None`` (the default),
            uses ``<cache_dir>/eop/finals.all.iau2000.txt``.
        max_age_days: Maximum acceptable age of the cached file in days.
            Defaults to 7.

    Returns:
        EOPData loaded from the cached (or freshly downloaded) file, or
        the bundled default data as a fallback.

    Examples:
        ```python
        from astrojax.eop import load_cached_eop, get_ut1_utc

        # Uses default cache location and 7-day refresh
        eop = load_cached_eop()
        val = get_ut1_utc(eop, 59569.0)

        # Custom path and 1-day refresh
        eop = load_cached_eop("/tmp/eop_cache/finals.txt", max_age_days=1.0)
        ```
    """
    if filepath is None:
        filepath = get_eop_cache_dir() / _STANDARD_FILENAME
    else:
        filepath = Path(filepath)

    max_age_seconds = max_age_days * 86400.0

    if is_file_stale(filepath, max_age_seconds):
        try:
            download_standard_eop_file(filepath)
        except Exception:
            logger.warning(
                "Failed to download EOP data; falling back to bundled data.",
                exc_info=True,
            )
            return load_default_eop()

    try:
        return load_eop_from_file(filepath)
    except Exception:
        logger.warning(
            "Failed to parse cached EOP file %s; falling back to bundled data.",
            filepath,
            exc_info=True,
        )
        return load_default_eop()
