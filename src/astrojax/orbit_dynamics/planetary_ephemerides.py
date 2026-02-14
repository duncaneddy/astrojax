"""Approximate heliocentric planetary positions using JPL Keplerian elements.

Provides heliocentric position vectors for all eight major planets in the
EME2000 (J2000 equatorial) frame.  Positions are computed from time-varying
Keplerian elements (JPL Table 1, valid 1800-2050 AD) and are returned in
metres.

Accuracy is approximately 1 arcminute for inner planets and up to 10
arcminutes for outer planets over the valid date range.

Note:
    All returned positions are **heliocentric** (origin at the Sun),
    unlike ``sun_position()`` and ``moon_position()`` which are geocentric.

References:
    E.M. Standish & J.G. Williams, "Keplerian Elements for
    Approximate Positions of the Major Planets",
    https://ssd.jpl.nasa.gov/planets/approx_pos.html
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from astrojax.attitude_representations import Rx, Rz
from astrojax.constants import AU
from astrojax.epoch import Epoch
from astrojax.orbit_dynamics._jpl_planetary_coefficients import (
    EMB_ID,
    JUPITER_ID,
    MARS_ID,
    MERCURY_ID,
    NEPTUNE_ID,
    SATURN_ID,
    TABLE1_ELEMENTS,
    TABLE1_OBLIQUITY,
    URANUS_ID,
    VENUS_ID,
)
from astrojax.orbit_dynamics.ephemerides import _julian_centuries_from_j2000
from astrojax.orbits import anomaly_mean_to_eccentric


def _planet_position_heliocentric_ecliptic(planet_id: int, epc: Epoch) -> Array:
    """Compute heliocentric ecliptic position for a planet.

    Implements the JPL algorithm: propagate Keplerian elements to the
    given epoch, solve Kepler's equation, and rotate from the orbital
    plane to the ecliptic frame.

    Args:
        planet_id: Planet index (0=Mercury ... 7=Neptune).
        epc: Epoch at which to evaluate.

    Returns:
        Heliocentric position in the ecliptic frame, in metres. Shape ``(3,)``.
    """
    T = _julian_centuries_from_j2000(epc)

    # Extract coefficients for this planet: shape (6, 2)
    coeffs = TABLE1_ELEMENTS[planet_id]

    # Propagate elements: element = element_0 + element_dot * T
    a = coeffs[0, 0] + coeffs[0, 1] * T          # semi-major axis (AU)
    e = coeffs[1, 0] + coeffs[1, 1] * T          # eccentricity
    incl = coeffs[2, 0] + coeffs[2, 1] * T        # inclination (deg)
    L = coeffs[3, 0] + coeffs[3, 1] * T          # mean longitude (deg)
    lon_peri = coeffs[4, 0] + coeffs[4, 1] * T   # longitude of perihelion (deg)
    lon_node = coeffs[5, 0] + coeffs[5, 1] * T   # longitude of ascending node (deg)

    # Argument of perihelion and mean anomaly (degrees)
    omega = lon_peri - lon_node
    M = L - lon_peri

    # Wrap M to [-180, 180] degrees
    M = M % 360.0
    M = jnp.where(M > 180.0, M - 360.0, M)

    # Solve Kepler's equation for eccentric anomaly (degrees)
    E = anomaly_mean_to_eccentric(M, e, use_degrees=True)

    # Orbital plane coordinates (AU)
    E_rad = E * jnp.pi / 180.0
    x_prime = a * (jnp.cos(E_rad) - e)
    y_prime = a * jnp.sqrt(1.0 - e * e) * jnp.sin(E_rad)

    # Rotate from orbital plane to ecliptic: Rz(-lon_node) @ Rx(-incl) @ Rz(-omega)
    r_orbital = jnp.array([x_prime, y_prime, 0.0])
    r_ecliptic = Rz(-lon_node, use_degrees=True) @ (
        Rx(-incl, use_degrees=True) @ (Rz(-omega, use_degrees=True) @ r_orbital)
    )

    # Convert AU to metres
    return r_ecliptic * AU


def _ecliptic_to_equatorial(r_ecl: Array) -> Array:
    """Rotate a vector from the ecliptic to J2000 equatorial frame.

    Uses the JPL-specified obliquity of 23.43928 degrees.

    Args:
        r_ecl: Position vector in the ecliptic frame. Shape ``(3,)``.

    Returns:
        Position vector in the EME2000 equatorial frame. Shape ``(3,)``.
    """
    return Rx(-TABLE1_OBLIQUITY, use_degrees=True) @ r_ecl


def mercury_position_jpl_approx(epc: Epoch) -> Array:
    """Heliocentric position of Mercury in the EME2000 frame.

    Uses JPL approximate Keplerian elements (Table 1, 1800-2050 AD).

    Args:
        epc: Epoch at which to compute the position.

    Returns:
        Heliocentric position vector in metres. Shape ``(3,)``.

    Examples:
        ```python
        from astrojax import Epoch
        from astrojax.orbit_dynamics import mercury_position_jpl_approx
        epc = Epoch(2024, 6, 15)
        r = mercury_position_jpl_approx(epc)
        ```
    """
    return _ecliptic_to_equatorial(_planet_position_heliocentric_ecliptic(MERCURY_ID, epc))


def venus_position_jpl_approx(epc: Epoch) -> Array:
    """Heliocentric position of Venus in the EME2000 frame.

    Uses JPL approximate Keplerian elements (Table 1, 1800-2050 AD).

    Args:
        epc: Epoch at which to compute the position.

    Returns:
        Heliocentric position vector in metres. Shape ``(3,)``.

    Examples:
        ```python
        from astrojax import Epoch
        from astrojax.orbit_dynamics import venus_position_jpl_approx
        epc = Epoch(2024, 6, 15)
        r = venus_position_jpl_approx(epc)
        ```
    """
    return _ecliptic_to_equatorial(_planet_position_heliocentric_ecliptic(VENUS_ID, epc))


def emb_position_jpl_approx(epc: Epoch) -> Array:
    """Heliocentric position of the Earth-Moon Barycenter in the EME2000 frame.

    Uses JPL approximate Keplerian elements (Table 1, 1800-2050 AD).

    Args:
        epc: Epoch at which to compute the position.

    Returns:
        Heliocentric position vector in metres. Shape ``(3,)``.

    Examples:
        ```python
        from astrojax import Epoch
        from astrojax.orbit_dynamics import emb_position_jpl_approx
        epc = Epoch(2024, 6, 15)
        r = emb_position_jpl_approx(epc)
        ```
    """
    return _ecliptic_to_equatorial(_planet_position_heliocentric_ecliptic(EMB_ID, epc))


def mars_position_jpl_approx(epc: Epoch) -> Array:
    """Heliocentric position of Mars in the EME2000 frame.

    Uses JPL approximate Keplerian elements (Table 1, 1800-2050 AD).

    Args:
        epc: Epoch at which to compute the position.

    Returns:
        Heliocentric position vector in metres. Shape ``(3,)``.

    Examples:
        ```python
        from astrojax import Epoch
        from astrojax.orbit_dynamics import mars_position_jpl_approx
        epc = Epoch(2024, 6, 15)
        r = mars_position_jpl_approx(epc)
        ```
    """
    return _ecliptic_to_equatorial(_planet_position_heliocentric_ecliptic(MARS_ID, epc))


def jupiter_position_jpl_approx(epc: Epoch) -> Array:
    """Heliocentric position of Jupiter in the EME2000 frame.

    Uses JPL approximate Keplerian elements (Table 1, 1800-2050 AD).

    Args:
        epc: Epoch at which to compute the position.

    Returns:
        Heliocentric position vector in metres. Shape ``(3,)``.

    Examples:
        ```python
        from astrojax import Epoch
        from astrojax.orbit_dynamics import jupiter_position_jpl_approx
        epc = Epoch(2024, 6, 15)
        r = jupiter_position_jpl_approx(epc)
        ```
    """
    return _ecliptic_to_equatorial(_planet_position_heliocentric_ecliptic(JUPITER_ID, epc))


def saturn_position_jpl_approx(epc: Epoch) -> Array:
    """Heliocentric position of Saturn in the EME2000 frame.

    Uses JPL approximate Keplerian elements (Table 1, 1800-2050 AD).

    Args:
        epc: Epoch at which to compute the position.

    Returns:
        Heliocentric position vector in metres. Shape ``(3,)``.

    Examples:
        ```python
        from astrojax import Epoch
        from astrojax.orbit_dynamics import saturn_position_jpl_approx
        epc = Epoch(2024, 6, 15)
        r = saturn_position_jpl_approx(epc)
        ```
    """
    return _ecliptic_to_equatorial(_planet_position_heliocentric_ecliptic(SATURN_ID, epc))


def uranus_position_jpl_approx(epc: Epoch) -> Array:
    """Heliocentric position of Uranus in the EME2000 frame.

    Uses JPL approximate Keplerian elements (Table 1, 1800-2050 AD).

    Args:
        epc: Epoch at which to compute the position.

    Returns:
        Heliocentric position vector in metres. Shape ``(3,)``.

    Examples:
        ```python
        from astrojax import Epoch
        from astrojax.orbit_dynamics import uranus_position_jpl_approx
        epc = Epoch(2024, 6, 15)
        r = uranus_position_jpl_approx(epc)
        ```
    """
    return _ecliptic_to_equatorial(_planet_position_heliocentric_ecliptic(URANUS_ID, epc))


def neptune_position_jpl_approx(epc: Epoch) -> Array:
    """Heliocentric position of Neptune in the EME2000 frame.

    Uses JPL approximate Keplerian elements (Table 1, 1800-2050 AD).

    Args:
        epc: Epoch at which to compute the position.

    Returns:
        Heliocentric position vector in metres. Shape ``(3,)``.

    Examples:
        ```python
        from astrojax import Epoch
        from astrojax.orbit_dynamics import neptune_position_jpl_approx
        epc = Epoch(2024, 6, 15)
        r = neptune_position_jpl_approx(epc)
        ```
    """
    return _ecliptic_to_equatorial(_planet_position_heliocentric_ecliptic(NEPTUNE_ID, epc))


def planet_position_jpl_approx(planet_id: int, epc: Epoch) -> Array:
    """Heliocentric position of a planet in the EME2000 frame.

    General dispatcher that accepts a planet ID. Supports JAX tracing
    for the ``planet_id`` argument via dynamic array indexing.

    Uses JPL approximate Keplerian elements (Table 1, 1800-2050 AD).

    Args:
        planet_id: Planet index. Use the module constants:
            ``MERCURY_ID=0``, ``VENUS_ID=1``, ``EMB_ID=2``, ``MARS_ID=3``,
            ``JUPITER_ID=4``, ``SATURN_ID=5``, ``URANUS_ID=6``, ``NEPTUNE_ID=7``.
        epc: Epoch at which to compute the position.

    Returns:
        Heliocentric position vector in metres. Shape ``(3,)``.

    Examples:
        ```python
        from astrojax import Epoch
        from astrojax.orbit_dynamics import planet_position_jpl_approx, MARS_ID
        epc = Epoch(2024, 6, 15)
        r = planet_position_jpl_approx(MARS_ID, epc)
        ```
    """
    return _ecliptic_to_equatorial(_planet_position_heliocentric_ecliptic(planet_id, epc))
