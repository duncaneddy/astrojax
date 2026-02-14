"""Factory functions for creating SpaceWeatherData instances.

Provides convenience constructors for common space weather configurations:

- :func:`static_space_weather`: Constant space weather values (useful for
  testing or when specific values are known).
- :func:`zero_space_weather`: All-zero space weather.
- :func:`load_sw_from_file`: Load from CSSI format file.
- :func:`load_default_sw`: Load bundled ``sw19571001.txt`` data.
- :func:`load_cached_sw`: Load from a local cache, downloading fresh data
  from CelesTrak when stale.
"""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path

import jax.numpy as jnp

from astrojax.config import get_dtype
from astrojax.space_weather._download import _SW_FILENAME, download_sw_file
from astrojax.space_weather._parsers import parse_cssi_file
from astrojax.space_weather._types import SpaceWeatherData
from astrojax.utils.caching import get_sw_cache_dir, is_file_stale

logger = logging.getLogger(__name__)

_DEFAULT_MAX_AGE_DAYS: float = 7.0
"""Default maximum age for cached space weather data in days."""


def static_space_weather(
    ap: float = 4.0,
    f107: float = 150.0,
    f107a: float = 150.0,
    kp: float = 1.0,
    mjd_min: float = 0.0,
    mjd_max: float = 99999.0,
) -> SpaceWeatherData:
    """Create a SpaceWeatherData with constant values.

    Useful for testing or when specific constant space weather values are
    known. The resulting dataset contains two points (at mjd_min and mjd_max)
    with identical values, so lookups return the constant everywhere.

    Args:
        ap: Constant Ap index. Default: 4.0.
        f107: Constant F10.7 flux [sfu]. Default: 150.0.
        f107a: Constant 81-day average F10.7 flux [sfu]. Default: 150.0.
        kp: Constant Kp index. Default: 1.0.
        mjd_min: Start of the valid MJD range. Default: 0.0.
        mjd_max: End of the valid MJD range. Default: 99999.0.

    Returns:
        SpaceWeatherData with constant values.

    Examples:
        ```python
        from astrojax.space_weather import static_space_weather, get_sw_f107_obs
        sw = static_space_weather(f107=200.0)
        val = get_sw_f107_obs(sw, 59569.0)  # returns ~200.0
        ```
    """
    dtype = get_dtype()
    mjd_arr = jnp.array([mjd_min, mjd_max], dtype=dtype)
    kp_arr = jnp.full((2, 8), kp, dtype=dtype)
    ap_arr = jnp.full((2, 8), ap, dtype=dtype)
    return SpaceWeatherData(
        mjd=mjd_arr,
        kp=kp_arr,
        ap=ap_arr,
        ap_daily=jnp.array([ap, ap], dtype=dtype),
        f107_obs=jnp.array([f107, f107], dtype=dtype),
        f107_adj=jnp.array([f107, f107], dtype=dtype),
        f107_obs_ctr81=jnp.array([f107a, f107a], dtype=dtype),
        f107_obs_lst81=jnp.array([f107a, f107a], dtype=dtype),
        f107_adj_ctr81=jnp.array([f107a, f107a], dtype=dtype),
        f107_adj_lst81=jnp.array([f107a, f107a], dtype=dtype),
        mjd_min=jnp.array(mjd_min, dtype=dtype),
        mjd_max=jnp.array(mjd_max, dtype=dtype),
    )


def zero_space_weather() -> SpaceWeatherData:
    """Create a SpaceWeatherData with all-zero values.

    Returns:
        SpaceWeatherData with all values set to zero.

    Examples:
        ```python
        from astrojax.space_weather import zero_space_weather, get_sw_f107_obs
        sw = zero_space_weather()
        val = get_sw_f107_obs(sw, 59569.0)  # returns 0.0
        ```
    """
    return static_space_weather(ap=0.0, f107=0.0, f107a=0.0, kp=0.0)


def load_sw_from_file(filepath: str | Path) -> SpaceWeatherData:
    """Load space weather data from a CSSI format file.

    Parses the file, converts lists to JAX arrays, and computes metadata
    scalars (mjd_min, mjd_max).

    Args:
        filepath: Path to a CSSI space weather file
            (e.g. ``sw19571001.txt``).

    Returns:
        SpaceWeatherData ready for JIT-compatible lookups.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no valid data is found.

    Examples:
        ```python
        from astrojax.space_weather import load_sw_from_file, get_sw_f107_obs
        sw = load_sw_from_file("path/to/sw19571001.txt")
        val = get_sw_f107_obs(sw, 59569.0)
        ```
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Space weather file not found: {filepath}")

    (
        mjds,
        kps,
        aps,
        ap_dailys,
        f107_obss,
        f107_adjs,
        f107_obs_ctr81s,
        f107_obs_lst81s,
        f107_adj_ctr81s,
        f107_adj_lst81s,
    ) = parse_cssi_file(str(filepath))

    dtype = get_dtype()

    return SpaceWeatherData(
        mjd=jnp.array(mjds, dtype=dtype),
        kp=jnp.array(kps, dtype=dtype),
        ap=jnp.array(aps, dtype=dtype),
        ap_daily=jnp.array(ap_dailys, dtype=dtype),
        f107_obs=jnp.array(f107_obss, dtype=dtype),
        f107_adj=jnp.array(f107_adjs, dtype=dtype),
        f107_obs_ctr81=jnp.array(f107_obs_ctr81s, dtype=dtype),
        f107_obs_lst81=jnp.array(f107_obs_lst81s, dtype=dtype),
        f107_adj_ctr81=jnp.array(f107_adj_ctr81s, dtype=dtype),
        f107_adj_lst81=jnp.array(f107_adj_lst81s, dtype=dtype),
        mjd_min=jnp.array(mjds[0], dtype=dtype),
        mjd_max=jnp.array(mjds[-1], dtype=dtype),
    )


def load_default_sw() -> SpaceWeatherData:
    """Load the bundled default space weather data (``sw19571001.txt``).

    Uses ``importlib.resources`` to locate the data file bundled with the
    package.

    Returns:
        SpaceWeatherData loaded from the bundled ``sw19571001.txt``.

    Examples:
        ```python
        from astrojax.space_weather import load_default_sw, get_sw_f107_obs
        sw = load_default_sw()
        val = get_sw_f107_obs(sw, 59569.0)
        ```
    """
    data_pkg = importlib.resources.files("astrojax.data.space_weather")
    resource = data_pkg.joinpath("sw19571001.txt")
    with importlib.resources.as_file(resource) as path:
        return load_sw_from_file(path)


def load_cached_sw(
    filepath: str | Path | None = None,
    *,
    max_age_days: float = _DEFAULT_MAX_AGE_DAYS,
) -> SpaceWeatherData:
    """Load space weather data from a local cache, downloading when stale.

    Checks whether the cached file at *filepath* exists and is younger than
    *max_age_days*. If the file is missing or stale, a fresh copy is
    downloaded from CelesTrak. If the download fails or the file cannot be
    parsed, the bundled default data is returned so this function never
    raises on network issues.

    Args:
        filepath: Path to the cached SW file. When ``None`` (the default),
            uses ``<cache_dir>/space_weather/sw19571001.txt``.
        max_age_days: Maximum acceptable age of the cached file in days.
            Defaults to 7.

    Returns:
        SpaceWeatherData loaded from the cached (or freshly downloaded) file,
        or the bundled default data as a fallback.

    Examples:
        ```python
        from astrojax.space_weather import load_cached_sw, get_sw_f107_obs
        sw = load_cached_sw()
        val = get_sw_f107_obs(sw, 59569.0)
        ```
    """
    if filepath is None:
        filepath = get_sw_cache_dir() / _SW_FILENAME
    else:
        filepath = Path(filepath)

    max_age_seconds = max_age_days * 86400.0

    if is_file_stale(filepath, max_age_seconds):
        try:
            download_sw_file(filepath)
        except Exception:
            logger.warning(
                "Failed to download space weather data; falling back to bundled data.",
                exc_info=True,
            )
            return load_default_sw()

    try:
        return load_sw_from_file(filepath)
    except Exception:
        logger.warning(
            "Failed to parse cached space weather file %s; falling back to bundled data.",
            filepath,
            exc_info=True,
        )
        return load_default_sw()
