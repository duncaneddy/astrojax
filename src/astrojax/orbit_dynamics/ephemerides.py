"""Low-precision analytical ephemerides for the Sun and Moon.

Provides position vectors in the EME2000 (ECI) inertial frame using
the analytical models from Montenbruck & Gill.  These are suitable for
perturbation force modelling where ~0.1 deg accuracy is acceptable.

All positions are in SI base units (metres).

.. note::

    Time system: UTC is assumed to approximate TT for computing Julian
    centuries from J2000.  The error (~69 s as of 2024) introduces a
    negligible position offset for low-precision ephemeris work.

References:
    1. O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods
       and Applications*, 2012.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from astrojax.config import get_dtype
from astrojax.constants import AS2RAD, DEG2RAD
from astrojax.attitude_representations import Rx
from astrojax.epoch import Epoch

# Julian Date of J2000.0
_JD_J2000 = 2451545
_SECONDS_PER_DAY = 86400.0

# Obliquity of the J2000 ecliptic [rad]
_EPSILON = 23.43929111 * DEG2RAD


def _julian_centuries_from_j2000(epc: Epoch) -> jax.Array:
    """Compute Julian centuries from J2000 using split representation.

    Uses the same high-precision approach as ``Epoch.gmst()`` to avoid
    the lossy single-float JD path.

    Args:
        epc: Epoch at which to evaluate.

    Returns:
        Julian centuries (T) from J2000.0.
    """
    _float = get_dtype()
    days_from_j2000 = _float(epc._jd - jnp.int32(_JD_J2000))
    frac_day = epc._compensated_seconds() / _float(_SECONDS_PER_DAY)
    return (days_from_j2000 + frac_day) / _float(36525.0)


def _frac(x):
    """Fractional part of x: ``x - floor(x)``."""
    return x - jnp.floor(x)


def sun_position(epc: Epoch) -> Array:
    """Position of the Sun in the ECI (EME2000) frame.

    Uses the low-precision analytical model from Montenbruck & Gill.

    Args:
        epc: Epoch at which to compute the Sun's position.

    Returns:
        3-element Sun position vector in metres.

    Examples:
        ```python
        from astrojax import Epoch
        from astrojax.orbit_dynamics import sun_position
        epc = Epoch(2024, 2, 25)
        r_sun = sun_position(epc)
        float(jnp.linalg.norm(r_sun))  # ~1 AU
        ```
    """
    _float = get_dtype()
    pi2 = _float(2.0) * jnp.pi

    T = _julian_centuries_from_j2000(epc)

    # Mean anomaly [rad]
    M = pi2 * _frac(_float(0.9931267) + _float(99.9973583) * T)

    # Ecliptic longitude [rad]
    L = pi2 * _frac(
        _float(0.7859444)
        + M / pi2
        + (_float(6892.0) * jnp.sin(M) + _float(72.0) * jnp.sin(_float(2.0) * M))
        / _float(1296.0e3)
    )

    # Distance [m]
    r = (
        _float(149.619e9)
        - _float(2.499e9) * jnp.cos(M)
        - _float(0.021e9) * jnp.cos(_float(2.0) * M)
    )

    # Position in ecliptic coordinates
    r_ecliptic = jnp.array([r * jnp.cos(L), r * jnp.sin(L), _float(0.0)])

    # Rotate from ecliptic to equatorial (EME2000) via Rx(-epsilon)
    R = Rx(-_EPSILON)
    return R @ r_ecliptic


def moon_position(epc: Epoch) -> Array:
    """Position of the Moon in the ECI (EME2000) frame.

    Uses the low-precision analytical model from Montenbruck & Gill.

    Args:
        epc: Epoch at which to compute the Moon's position.

    Returns:
        3-element Moon position vector in metres.

    Examples:
        ```python
        from astrojax import Epoch
        from astrojax.orbit_dynamics import moon_position
        epc = Epoch(2024, 2, 25)
        r_moon = moon_position(epc)
        float(jnp.linalg.norm(r_moon))  # ~384,000 km
        ```
    """
    _float = get_dtype()
    pi2 = _float(2.0) * jnp.pi

    T = _julian_centuries_from_j2000(epc)

    # Mean elements of the lunar orbit
    L_0 = _frac(_float(0.606433) + _float(1336.851344) * T)     # Mean longitude [rev]
    l_m = pi2 * _frac(_float(0.374897) + _float(1325.552410) * T)  # Moon mean anomaly [rad]
    lp = pi2 * _frac(_float(0.993133) + _float(99.997361) * T)   # Sun mean anomaly [rad]
    D = pi2 * _frac(_float(0.827361) + _float(1236.853086) * T)  # Diff longitude Moon-Sun [rad]
    F = pi2 * _frac(_float(0.259086) + _float(1342.227825) * T)  # Argument of latitude [rad]

    # Ecliptic longitude perturbation [arcsec]
    dL = (
        _float(22640.0) * jnp.sin(l_m)
        - _float(4586.0) * jnp.sin(l_m- _float(2.0) * D)
        + _float(2370.0) * jnp.sin(_float(2.0) * D)
        + _float(769.0) * jnp.sin(_float(2.0) * l_m)
        - _float(668.0) * jnp.sin(lp)
        - _float(412.0) * jnp.sin(_float(2.0) * F)
        - _float(212.0) * jnp.sin(_float(2.0) * l_m - _float(2.0) * D)
        - _float(206.0) * jnp.sin(l_m+ lp - _float(2.0) * D)
        + _float(192.0) * jnp.sin(l_m+ _float(2.0) * D)
        - _float(165.0) * jnp.sin(lp - _float(2.0) * D)
        - _float(125.0) * jnp.sin(D)
        - _float(110.0) * jnp.sin(l_m+ lp)
        + _float(148.0) * jnp.sin(l_m- lp)
        - _float(55.0) * jnp.sin(_float(2.0) * F - _float(2.0) * D)
    )

    # Ecliptic longitude [rad]
    L = pi2 * _frac(L_0 + dL / _float(1296.0e3))

    # Ecliptic latitude [rad]
    S = F + (dL + _float(412.0) * jnp.sin(_float(2.0) * F) + _float(541.0) * jnp.sin(lp)) * _float(AS2RAD)
    h = F - _float(2.0) * D
    N = (
        -_float(526.0) * jnp.sin(h)
        + _float(44.0) * jnp.sin(l_m+ h)
        - _float(31.0) * jnp.sin(-l_m+ h)
        - _float(23.0) * jnp.sin(lp + h)
        + _float(11.0) * jnp.sin(-lp + h)
        - _float(25.0) * jnp.sin(-_float(2.0) * l_m+ F)
        + _float(21.0) * jnp.sin(-l_m+ F)
    )
    B = (_float(18520.0) * jnp.sin(S) + N) * _float(AS2RAD)

    # Distance [m]
    r = (
        _float(385000e3)
        - _float(20905e3) * jnp.cos(l_m)
        - _float(3699e3) * jnp.cos(_float(2.0) * D - l_m)
        - _float(2956e3) * jnp.cos(_float(2.0) * D)
        - _float(570e3) * jnp.cos(_float(2.0) * l_m)
        + _float(246e3) * jnp.cos(_float(2.0) * l_m - _float(2.0) * D)
        - _float(205e3) * jnp.cos(lp - _float(2.0) * D)
        - _float(171e3) * jnp.cos(l_m + _float(2.0) * D)
        - _float(152e3) * jnp.cos(l_m + lp - _float(2.0) * D)
    )

    # Position in ecliptic coordinates
    r_ecliptic = jnp.array([
        r * jnp.cos(L) * jnp.cos(B),
        r * jnp.sin(L) * jnp.cos(B),
        r * jnp.sin(B),
    ])

    # Rotate from ecliptic to equatorial (EME2000) via Rx(-epsilon)
    R = Rx(-_EPSILON)
    return R @ r_ecliptic
