from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .constants import JD_MJD_OFFSET

def caldate_to_mjd(year: ArrayLike, month: ArrayLike, day: ArrayLike, hour: ArrayLike = 0, minute: ArrayLike = 0, second: ArrayLike = 0.0) -> jax.Array:
    """Convert a calendar date to Modified Julian Date. Algorithm is only valid from year 1583 onward.

    Args:
        year (ArrayLike): Year of the calendar date.
        month (ArrayLike): Month of the calendar date.
        day (ArrayLike): Day of the calendar date.
        hour (ArrayLike): Hour of the calendar date. Default: ``0``
        minute (ArrayLike): Minute of the calendar date. Default: ``0``
        second (ArrayLike): Second of the calendar date. Default: ``0.0``

    Returns:
        jax.Array: Modified Julian Date (float32).

    References:

        1. Montenbruck, O., & Gill, E. (2012). *Satellite Orbits: Models, Methods and Applications*. Springer Science & Business Media.
    """

    is_jan_or_feb = month <= 2
    year = jnp.where(is_jan_or_feb, year - 1, year)
    month = jnp.where(is_jan_or_feb, month + 12, month)

    B = jnp.floor(year / 400) - jnp.floor(year / 100) + jnp.floor(year / 4)

    mjd = 365*year - 679004 + B + jnp.floor(30.6001 * (month + 1)) + day

    frac_day = (hour + (minute + second / 60.0) / 60.0) / 24.0

    return jnp.float32(jnp.floor(mjd).astype(jnp.int32)) + frac_day

def caldate_to_jd(year: ArrayLike, month: ArrayLike, day: ArrayLike, hour: ArrayLike = 0, minute: ArrayLike = 0, second: ArrayLike = 0.0) -> jax.Array:
    """Convert a calendar date to Julian Date.

    Args:
        year (ArrayLike): Year of the calendar date.
        month (ArrayLike): Month of the calendar date.
        day (ArrayLike): Day of the calendar date.
        hour (ArrayLike): Hour of the calendar date. Default: ``0``
        minute (ArrayLike): Minute of the calendar date. Default: ``0``
        second (ArrayLike): Second of the calendar date. Default: ``0.0``

    Returns:
        jax.Array: Julian Date (float32).

    References:

        1. Montenbruck, O., & Gill, E. (2012). *Satellite Orbits: Models, Methods and Applications*. Springer Science & Business Media.
    """

    mjd = caldate_to_mjd(year, month, day, hour, minute, second)

    return mjd + JD_MJD_OFFSET

def jd_to_mjd(jd: ArrayLike) -> jax.Array:
    """Convert Julian Date to Modified Julian Date.

    Args:
        jd (ArrayLike): Julian Date.

    Returns:
        jax.Array: Modified Julian Date.

    References:

        1. Montenbruck, O., & Gill, E. (2012). *Satellite Orbits: Models, Methods and Applications*. Springer Science & Business Media.
    """

    return jd - JD_MJD_OFFSET

def mjd_to_jd(mjd: ArrayLike) -> jax.Array:
    """Convert Modified Julian Date to Julian Date.

    Args:
        mjd (ArrayLike): Modified Julian Date.

    Returns:
        jax.Array: Julian Date.

    References:

        1. Montenbruck, O., & Gill, E. (2012). *Satellite Orbits: Models, Methods and Applications*. Springer Science & Business Media.
    """

    return mjd + JD_MJD_OFFSET


def jd_to_caldate(jd: ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Convert Julian Date to calendar date.

    Uses the algorithm from Montenbruck & Gill for Gregorian calendar dates.

    Args:
        jd (ArrayLike): Julian Date.

    Returns:
        tuple[jax.Array, ...]: (year, month, day, hour, minute, second) where
            year/month/day/hour/minute are int32 and second is float32.

    References:

        1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
           Applications*, 2012, p. 322.
    """
    jd_shifted = jd + 0.5
    z = jnp.floor(jd_shifted).astype(jnp.int32)
    f = jd_shifted - z

    # Julian/Gregorian calendar switchover at JD 2299161.
    # Use scaled integer arithmetic: (z - 1867216.25)/36524.25
    # = (100*z - 186721625) / 3652425 to avoid float32 precision loss.
    alpha = (100 * z - 186721625) // 3652425
    a_gregorian = z + 1 + alpha - alpha // 4
    a = jnp.where(z < 2299161, z, a_gregorian)

    b = a + 1524
    # Scaled: (b - 122.1)/365.25 = (100*b - 12210)/36525
    c = (100 * b - 12210) // 36525
    # Scaled: 365.25*c = 36525*c/100
    d = (36525 * c) // 100
    # Scaled: (b - d)/30.6001 = (b - d)*10000/306001
    e = ((b - d) * 10000) // 306001

    # Scaled: 30.6001*e = 306001*e/10000
    day_with_frac = b - d - (306001 * e) // 10000 + f
    day = jnp.floor(day_with_frac).astype(jnp.int32)
    frac_of_day = day_with_frac - day

    month = jnp.where(e < 14, e - 1, e - 13)
    year = jnp.where(month > 2, c - 4716, c - 4715)

    # Decompose fractional day via integer milliseconds to avoid
    # truncation artifacts from floating-point precision limits
    total_ms = jnp.round(frac_of_day * 86400000.0).astype(jnp.int32)
    hour = total_ms // 3600000
    total_ms = total_ms - hour * 3600000
    minute = total_ms // 60000
    total_ms = total_ms - minute * 60000
    second = jnp.float32(total_ms) / 1000.0

    return year, month, day, hour, minute, second


def mjd_to_caldate(mjd: ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Convert Modified Julian Date to calendar date.

    Args:
        mjd (ArrayLike): Modified Julian Date.

    Returns:
        tuple[jax.Array, ...]: (year, month, day, hour, minute, second) where
            year/month/day/hour/minute are int32 and second is float32.

    References:

        1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
           Applications*, 2012.
    """
    return jd_to_caldate(mjd + JD_MJD_OFFSET)
