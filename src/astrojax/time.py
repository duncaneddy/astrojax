import jax
import jax.numpy as jnp

from .constants import JD_MJD_OFFSET

def caldate_to_mjd(year:int, month:int, day:int, hour:int=0, minute:int=0, second:float=0.0) -> float:
    """Convert a calendar date to Julian Date. Algorithm is only valid from year 1583 onward.

    Args:
        year (int): Year of the calendar date.
        month (int): Month of the calendar date.
        day (int): Day of the calendar date.
        hour (int): Hour of the calendar date. Default: ``0``
        minute (int): Minute of the calendar date. Default: ``0``
        second (float): Second of the calendar date. Default: ``0.0``

    Returns:
        jd (float): Julian Date

    References:

        1. Montenbruck, O., & Gill, E. (2012). *Satellite Orbits: Models, Methods and Applications*. Springer Science & Business Media.
    """

    if month <= 2:
        year -= 1
        month += 12

    B = jnp.floor(year / 400) - jnp.floor(year / 100) + jnp.floor(year / 4)

    mjd = 365*year - 679004 + B + jnp.floor(30.6001 * (month + 1)) + day

    frac_day = (hour + (minute + second / 60.0) / 60.0) / 24.0

    return mjd + frac_day

def caldate_to_jd(year:int, month:int, day:int, hour:int=0, minute:int=0, second:float=0.0) -> float:
    """Convert a calendar date to Modified Julian Date.

    Args:
        year (int): Year of the calendar date.
        month (int): Month of the calendar date.
        day (int): Day of the calendar date.
        hour (int): Hour of the calendar date. Default: ``0``
        minute (int): Minute of the calendar date. Default: ``0``
        second (float): Second of the calendar date. Default: ``0.0``

    Returns:
        mjd (float): Modified Julian Date

    References:

        1. Montenbruck, O., & Gill, E. (2012). *Satellite Orbits: Models, Methods and Applications*. Springer Science & Business Media.
    """

    mjd = caldate_to_mjd(year, month, day, hour, minute, second)

    return mjd + JD_MJD_OFFSET

def jd_to_mjd(jd:float) -> float:
    """Convert Julian Date to Modified Julian Date.

    Args:
        jd (float): Julian Date

    Returns:
        mjd (float): Modified Julian Date

    References:

        1. Montenbruck, O., & Gill, E. (2012). *Satellite Orbits: Models, Methods and Applications*. Springer Science & Business Media.
    """

    return jd - JD_MJD_OFFSET

def mjd_to_jd(mjd:float) -> float:
    """Convert Modified Julian Date to Julian Date.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        jd (float): Julian Date

    References:

        1. Montenbruck, O., & Gill, E. (2012). *Satellite Orbits: Models, Methods and Applications*. Springer Science & Business Media.
    """

    return mjd + JD_MJD_OFFSET
