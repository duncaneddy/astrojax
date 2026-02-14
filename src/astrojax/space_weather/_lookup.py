"""JIT-compatible space weather lookup functions.

All functions use only JAX primitives (``jnp.searchsorted``, array indexing,
``jnp.where``) and are fully compatible with ``jax.jit``, ``jax.vmap``,
and ``jax.grad``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.space_weather._types import SpaceWeatherData


def _day_index(sw: SpaceWeatherData, mjd: Array) -> Array:
    """Find the day index for a given MJD using searchsorted.

    Uses floor(MJD) to find the daily entry. Clamps to valid range.

    Args:
        sw: Space weather dataset.
        mjd: Scalar MJD to query.

    Returns:
        Index into sw.mjd for the requested day.
    """
    mjd_floor = jnp.floor(mjd)
    idx = jnp.searchsorted(sw.mjd, mjd_floor, side="right") - 1
    return jnp.clip(idx, 0, sw.mjd.shape[0] - 1)


def _interval_index(mjd: Array) -> Array:
    """Get the 3-hour interval index (0-7) from the fractional MJD.

    Args:
        mjd: Scalar MJD.

    Returns:
        Integer interval index (0-7).
    """
    fraction = mjd - jnp.floor(mjd)
    hours = fraction * 24.0
    index = jnp.floor(hours / 3.0).astype(jnp.int32)
    return jnp.clip(index, 0, 7)


def get_sw_kp(sw: SpaceWeatherData, mjd: ArrayLike) -> Array:
    """Query 3-hourly Kp index at the given MJD.

    Args:
        sw: Space weather dataset.
        mjd: Modified Julian Date to query.

    Returns:
        Kp index (0.0-9.0) for the 3-hour interval containing the MJD.
    """
    mjd = jnp.asarray(mjd, dtype=sw.mjd.dtype)
    idx = _day_index(sw, mjd)
    interval = _interval_index(mjd)
    return sw.kp[idx, interval]


def get_sw_ap(sw: SpaceWeatherData, mjd: ArrayLike) -> Array:
    """Query 3-hourly Ap index at the given MJD.

    Args:
        sw: Space weather dataset.
        mjd: Modified Julian Date to query.

    Returns:
        Ap index for the 3-hour interval containing the MJD.
    """
    mjd = jnp.asarray(mjd, dtype=sw.mjd.dtype)
    idx = _day_index(sw, mjd)
    interval = _interval_index(mjd)
    return sw.ap[idx, interval]


def get_sw_ap_daily(sw: SpaceWeatherData, mjd: ArrayLike) -> Array:
    """Query daily average Ap index at the given MJD.

    Args:
        sw: Space weather dataset.
        mjd: Modified Julian Date to query.

    Returns:
        Daily average Ap index.
    """
    mjd = jnp.asarray(mjd, dtype=sw.mjd.dtype)
    idx = _day_index(sw, mjd)
    return sw.ap_daily[idx]


def get_sw_f107_obs(sw: SpaceWeatherData, mjd: ArrayLike) -> Array:
    """Query observed F10.7 solar flux at the given MJD.

    Args:
        sw: Space weather dataset.
        mjd: Modified Julian Date to query.

    Returns:
        Observed F10.7 flux [sfu].
    """
    mjd = jnp.asarray(mjd, dtype=sw.mjd.dtype)
    idx = _day_index(sw, mjd)
    return sw.f107_obs[idx]


def get_sw_f107_adj(sw: SpaceWeatherData, mjd: ArrayLike) -> Array:
    """Query adjusted F10.7 solar flux at the given MJD.

    Args:
        sw: Space weather dataset.
        mjd: Modified Julian Date to query.

    Returns:
        Adjusted F10.7 flux [sfu].
    """
    mjd = jnp.asarray(mjd, dtype=sw.mjd.dtype)
    idx = _day_index(sw, mjd)
    return sw.f107_adj[idx]


def get_sw_f107_obs_ctr81(sw: SpaceWeatherData, mjd: ArrayLike) -> Array:
    """Query 81-day centered average observed F10.7 at the given MJD.

    Args:
        sw: Space weather dataset.
        mjd: Modified Julian Date to query.

    Returns:
        81-day centered average observed F10.7 flux [sfu].
    """
    mjd = jnp.asarray(mjd, dtype=sw.mjd.dtype)
    idx = _day_index(sw, mjd)
    return sw.f107_obs_ctr81[idx]


def get_sw_f107_obs_lst81(sw: SpaceWeatherData, mjd: ArrayLike) -> Array:
    """Query 81-day last average observed F10.7 at the given MJD.

    Args:
        sw: Space weather dataset.
        mjd: Modified Julian Date to query.

    Returns:
        81-day last average observed F10.7 flux [sfu].
    """
    mjd = jnp.asarray(mjd, dtype=sw.mjd.dtype)
    idx = _day_index(sw, mjd)
    return sw.f107_obs_lst81[idx]


def _get_ap_at_offset(sw: SpaceWeatherData, mjd: Array, hours_offset: float) -> Array:
    """Get the 3-hourly Ap at a given hour offset from the reference MJD.

    Args:
        sw: Space weather dataset.
        mjd: Reference MJD.
        hours_offset: Hour offset (negative = past).

    Returns:
        Ap index at the offset time.
    """
    offset_mjd = mjd + hours_offset / 24.0
    idx = _day_index(sw, offset_mjd)
    interval = _interval_index(offset_mjd)
    return sw.ap[idx, interval]


def _avg_ap_range(sw: SpaceWeatherData, mjd: Array, start_hours: float, n_intervals: int) -> Array:
    """Average N consecutive 3-hourly Ap values starting at a given offset.

    Args:
        sw: Space weather dataset.
        mjd: Reference MJD.
        start_hours: Start hour offset (negative = past).
        n_intervals: Number of 3-hour intervals to average.

    Returns:
        Average Ap over the specified range.
    """
    total = jnp.zeros((), dtype=sw.mjd.dtype)
    for i in range(n_intervals):
        offset = start_hours + i * 3.0
        total = total + _get_ap_at_offset(sw, mjd, offset)
    return total / n_intervals


def get_sw_ap_array(sw: SpaceWeatherData, mjd: ArrayLike) -> Array:
    """Build the 7-element NRLMSISE-00 AP array.

    Constructs the magnetic activity array required by the NRLMSISE-00 model:

    - ``[0]``: Daily Ap
    - ``[1]``: Current 3-hour Ap
    - ``[2]``: 3-hour Ap at -3h
    - ``[3]``: 3-hour Ap at -6h
    - ``[4]``: 3-hour Ap at -9h
    - ``[5]``: Average of eight 3-hour Ap from 12-33h prior
    - ``[6]``: Average of eight 3-hour Ap from 36-57h prior

    All lookups use compile-time known offsets (unrolled at trace time).

    Args:
        sw: Space weather dataset.
        mjd: Modified Julian Date to query.

    Returns:
        Array of shape ``(7,)`` with the NRLMSISE-00 AP array.
    """
    mjd = jnp.asarray(mjd, dtype=sw.mjd.dtype)

    ap_daily = get_sw_ap_daily(sw, mjd)
    ap_current = _get_ap_at_offset(sw, mjd, 0.0)
    ap_m3 = _get_ap_at_offset(sw, mjd, -3.0)
    ap_m6 = _get_ap_at_offset(sw, mjd, -6.0)
    ap_m9 = _get_ap_at_offset(sw, mjd, -9.0)

    # Average of 8 three-hour Ap from 12-33h prior
    ap_avg_12_33 = _avg_ap_range(sw, mjd, -33.0, 8)

    # Average of 8 three-hour Ap from 36-57h prior
    ap_avg_36_57 = _avg_ap_range(sw, mjd, -57.0, 8)

    return jnp.array(
        [
            ap_daily,
            ap_current,
            ap_m3,
            ap_m6,
            ap_m9,
            ap_avg_12_33,
            ap_avg_36_57,
        ]
    )
