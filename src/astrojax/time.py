from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .config import get_dtype
from .constants import JD_MJD_OFFSET

# TT - TAI offset in seconds (constant by definition)
TT_TAI: float = 32.184

# Leap second table: (MJD of introduction, TAI-UTC in seconds)
# Each entry marks the MJD at which TAI-UTC steps to the given value.
# Source: IERS Bulletin C / USNO leap second table (1972-01-01 through 2017-01-01).
_LEAP_SECOND_TABLE: tuple[tuple[float, float], ...] = (
    (41317.0, 10.0),  # 1972-01-01
    (41499.0, 11.0),  # 1972-07-01
    (41683.0, 12.0),  # 1973-01-01
    (42048.0, 13.0),  # 1974-01-01
    (42413.0, 14.0),  # 1975-01-01
    (42778.0, 15.0),  # 1976-01-01
    (43144.0, 16.0),  # 1977-01-01
    (43509.0, 17.0),  # 1978-01-01
    (43874.0, 18.0),  # 1979-01-01
    (44239.0, 19.0),  # 1980-01-01
    (44786.0, 20.0),  # 1981-07-01
    (45151.0, 21.0),  # 1982-01-01
    (45516.0, 22.0),  # 1983-07-01
    (46247.0, 23.0),  # 1985-07-01
    (47161.0, 24.0),  # 1988-01-01
    (47892.0, 25.0),  # 1990-01-01
    (48257.0, 26.0),  # 1991-01-01
    (48804.0, 27.0),  # 1992-07-01
    (49169.0, 28.0),  # 1993-07-01
    (49534.0, 29.0),  # 1994-07-01
    (50083.0, 30.0),  # 1996-01-01
    (50630.0, 31.0),  # 1997-07-01
    (51179.0, 32.0),  # 1999-01-01
    (53736.0, 33.0),  # 2006-01-01
    (54832.0, 34.0),  # 2009-01-01
    (56109.0, 35.0),  # 2012-07-01
    (57204.0, 36.0),  # 2015-07-01
    (57754.0, 37.0),  # 2017-01-01
)


def leap_seconds_tai_utc(mjd: ArrayLike) -> jax.Array:
    """Return TAI-UTC (cumulative leap seconds) for a given MJD.

    Uses a hardcoded step-function lookup table covering 1972-01-01 through
    2017-01-01. For dates before 1972, returns 10.0; for dates after the last
    entry, returns the most recent value (37.0).

    JIT-compatible: uses ``jnp.searchsorted`` for O(log n) lookup.

    Args:
        mjd: Modified Julian Date (UTC), scalar or array.

    Returns:
        TAI-UTC in seconds.
    """
    mjd = jnp.asarray(mjd, dtype=get_dtype())
    mjd_breaks = jnp.array([m for m, _ in _LEAP_SECOND_TABLE], dtype=get_dtype())
    tai_utc_vals = jnp.array([v for _, v in _LEAP_SECOND_TABLE], dtype=get_dtype())

    # searchsorted(side='right') returns the index of the first entry > mjd,
    # so idx-1 is the last entry <= mjd.
    idx = jnp.searchsorted(mjd_breaks, mjd, side="right")

    # Before the first entry (idx=0): return 10.0
    # Otherwise: return tai_utc_vals[idx-1]
    return jnp.where(idx == 0, get_dtype()(10.0), tai_utc_vals[idx - 1])


def caldate_to_mjd(
    year: ArrayLike,
    month: ArrayLike,
    day: ArrayLike,
    hour: ArrayLike = 0,
    minute: ArrayLike = 0,
    second: ArrayLike = 0.0,
) -> jax.Array:
    """Convert a calendar date to Modified Julian Date. Algorithm is only valid from year 1583 onward.

    Args:
        year (ArrayLike): Year of the calendar date.
        month (ArrayLike): Month of the calendar date.
        day (ArrayLike): Day of the calendar date.
        hour (ArrayLike): Hour of the calendar date. Default: ``0``
        minute (ArrayLike): Minute of the calendar date. Default: ``0``
        second (ArrayLike): Second of the calendar date. Default: ``0.0``

    Returns:
        Modified Julian Date.

    References:

        1. Montenbruck, O., & Gill, E. (2012). *Satellite Orbits: Models, Methods and Applications*. Springer Science & Business Media.
    """

    is_jan_or_feb = month <= 2
    year = jnp.where(is_jan_or_feb, year - 1, year)
    month = jnp.where(is_jan_or_feb, month + 12, month)

    B = jnp.floor(year / 400) - jnp.floor(year / 100) + jnp.floor(year / 4)

    mjd = 365 * year - 679004 + B + jnp.floor(30.6001 * (month + 1)) + day

    frac_day = (hour + (minute + second / 60.0) / 60.0) / 24.0

    return get_dtype()(jnp.floor(mjd).astype(jnp.int32)) + frac_day


def caldate_to_jd(
    year: ArrayLike,
    month: ArrayLike,
    day: ArrayLike,
    hour: ArrayLike = 0,
    minute: ArrayLike = 0,
    second: ArrayLike = 0.0,
) -> jax.Array:
    """Convert a calendar date to Julian Date.

    Args:
        year (ArrayLike): Year of the calendar date.
        month (ArrayLike): Month of the calendar date.
        day (ArrayLike): Day of the calendar date.
        hour (ArrayLike): Hour of the calendar date. Default: ``0``
        minute (ArrayLike): Minute of the calendar date. Default: ``0``
        second (ArrayLike): Second of the calendar date. Default: ``0.0``

    Returns:
        Julian Date.

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
        Modified Julian Date.

    References:

        1. Montenbruck, O., & Gill, E. (2012). *Satellite Orbits: Models, Methods and Applications*. Springer Science & Business Media.
    """

    return jd - JD_MJD_OFFSET


def mjd_to_jd(mjd: ArrayLike) -> jax.Array:
    """Convert Modified Julian Date to Julian Date.

    Args:
        mjd (ArrayLike): Modified Julian Date.

    Returns:
        Julian Date.

    References:

        1. Montenbruck, O., & Gill, E. (2012). *Satellite Orbits: Models, Methods and Applications*. Springer Science & Business Media.
    """

    return mjd + JD_MJD_OFFSET


def jd_to_caldate(
    jd: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Convert Julian Date to calendar date.

    Uses the algorithm from Montenbruck & Gill for Gregorian calendar dates.

    Args:
        jd (ArrayLike): Julian Date.

    Returns:
        tuple[jax.Array, ...]: (year, month, day, hour, minute, second) where
            year/month/day/hour/minute are int32 and second is configurable float dtype.

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
    second = get_dtype()(total_ms) / 1000.0

    return year, month, day, hour, minute, second


def mjd_to_caldate(
    mjd: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Convert Modified Julian Date to calendar date.

    Args:
        mjd (ArrayLike): Modified Julian Date.

    Returns:
        tuple[jax.Array, ...]: (year, month, day, hour, minute, second) where
            year/month/day/hour/minute are int32 and second is configurable float dtype.

    References:

        1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
           Applications*, 2012.
    """
    return jd_to_caldate(mjd + JD_MJD_OFFSET)
