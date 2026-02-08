"""Keplerian orbital mechanics functions for Earth-centric orbits.

This module provides functions for computing orbital parameters from
Keplerian elements, including orbital period, mean motion, semi-major axis,
velocities at apsides, distances, altitudes, anomaly conversions, and
special orbits (sun-synchronous, geostationary).

All functions use JAX operations and are compatible with ``jax.jit``,
``jax.vmap``, and ``jax.grad``. Inputs are coerced to the configured
float dtype (see :func:`astrojax.config.set_dtype`).

The anomaly conversion functions include a Newton-Raphson Kepler equation
solver implemented with ``jax.lax.fori_loop`` for JAX traceability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.constants import GM_EARTH, J2_EARTH, OMEGA_EARTH, R_EARTH
from astrojax.utils import from_radians, to_radians

# ──────────────────────────────────────────────
# Orbital period and semi-major axis
# ──────────────────────────────────────────────


def orbital_period(a: ArrayLike) -> Array:
    """Compute the orbital period of an object around Earth.

    Args:
        a: Semi-major axis. Units: *m*

    Returns:
        Orbital period. Units: *s*

    Examples:
        ```python
        from astrojax.constants import R_EARTH
        from astrojax.orbits import orbital_period
        T = orbital_period(R_EARTH + 500e3)
        ```
    """
    a = jnp.asarray(a, dtype=get_dtype())
    return 2.0 * jnp.pi * jnp.sqrt(a**3 / GM_EARTH)


def orbital_period_from_state(state_eci: ArrayLike) -> Array:
    """Compute orbital period from an ECI state vector using the vis-viva equation.

    Derives the semi-major axis from the state vector's position and velocity
    magnitudes, then computes the corresponding orbital period.

    Args:
        state_eci: ECI state vector ``[x, y, z, vx, vy, vz]``.
            Units: *m* and *m/s*

    Returns:
        Orbital period. Units: *s*

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH, GM_EARTH
        from astrojax.orbits import orbital_period_from_state
        r = R_EARTH + 500e3
        v = jnp.sqrt(GM_EARTH / r)
        state = jnp.array([r, 0.0, 0.0, 0.0, v, 0.0])
        T = orbital_period_from_state(state)
        ```
    """
    state_eci = jnp.asarray(state_eci, dtype=get_dtype())
    r = jnp.linalg.norm(state_eci[:3])
    v_sq = jnp.sum(state_eci[3:6] ** 2)
    a = 1.0 / (2.0 / r - v_sq / GM_EARTH)
    return orbital_period(a)


def semimajor_axis_from_orbital_period(period: ArrayLike) -> Array:
    """Compute semi-major axis from orbital period around Earth.

    Args:
        period: Orbital period. Units: *s*

    Returns:
        Semi-major axis. Units: *m*

    Examples:
        ```python
        from astrojax.orbits import semimajor_axis_from_orbital_period
        a = semimajor_axis_from_orbital_period(5676.977)
        ```
    """
    period = jnp.asarray(period, dtype=get_dtype())
    return (period**2 * GM_EARTH / (4.0 * jnp.pi**2)) ** (1.0 / 3.0)


def semimajor_axis(n: ArrayLike, use_degrees: bool = False) -> Array:
    """Compute semi-major axis from mean motion around Earth.

    Args:
        n: Mean motion. Units: *rad/s* or *deg/s*
        use_degrees: If ``True``, interpret ``n`` as degrees per second.

    Returns:
        Semi-major axis. Units: *m*

    Examples:
        ```python
        from astrojax.orbits import semimajor_axis
        a = semimajor_axis(0.001106784)
        ```
    """
    n = jnp.asarray(n, dtype=get_dtype())
    n_rad = to_radians(n, use_degrees)
    return (GM_EARTH / n_rad**2) ** (1.0 / 3.0)


# ──────────────────────────────────────────────
# Mean motion
# ──────────────────────────────────────────────


def mean_motion(a: ArrayLike, use_degrees: bool = False) -> Array:
    """Compute the mean motion of an object orbiting Earth.

    Args:
        a: Semi-major axis. Units: *m*
        use_degrees: If ``True``, return mean motion in degrees per second.

    Returns:
        Mean motion. Units: *rad/s* or *deg/s*

    Examples:
        ```python
        from astrojax.constants import R_EARTH
        from astrojax.orbits import mean_motion
        n = mean_motion(R_EARTH + 500e3)
        ```
    """
    a = jnp.asarray(a, dtype=get_dtype())
    n = jnp.sqrt(GM_EARTH / a**3)
    return from_radians(n, use_degrees)


# ──────────────────────────────────────────────
# Velocities at apsides
# ──────────────────────────────────────────────


def perigee_velocity(a: ArrayLike, e: ArrayLike) -> Array:
    """Compute velocity at perigee for an Earth orbit.

    Args:
        a: Semi-major axis. Units: *m*
        e: Eccentricity. Dimensionless.

    Returns:
        Perigee velocity magnitude. Units: *m/s*

    Examples:
        ```python
        from astrojax.constants import R_EARTH
        from astrojax.orbits import perigee_velocity
        vp = perigee_velocity(R_EARTH + 500e3, 0.001)
        ```
    """
    a = jnp.asarray(a, dtype=get_dtype())
    e = jnp.asarray(e, dtype=get_dtype())
    return jnp.sqrt(GM_EARTH / a) * jnp.sqrt((1.0 + e) / (1.0 - e))


def apogee_velocity(a: ArrayLike, e: ArrayLike) -> Array:
    """Compute velocity at apogee for an Earth orbit.

    Args:
        a: Semi-major axis. Units: *m*
        e: Eccentricity. Dimensionless.

    Returns:
        Apogee velocity magnitude. Units: *m/s*

    Examples:
        ```python
        from astrojax.constants import R_EARTH
        from astrojax.orbits import apogee_velocity
        va = apogee_velocity(R_EARTH + 500e3, 0.001)
        ```
    """
    a = jnp.asarray(a, dtype=get_dtype())
    e = jnp.asarray(e, dtype=get_dtype())
    return jnp.sqrt(GM_EARTH / a) * jnp.sqrt((1.0 - e) / (1.0 + e))


# ──────────────────────────────────────────────
# Distances and altitudes
# ──────────────────────────────────────────────


def periapsis_distance(a: ArrayLike, e: ArrayLike) -> Array:
    """Compute the distance at periapsis.

    Args:
        a: Semi-major axis. Units: *m*
        e: Eccentricity. Dimensionless.

    Returns:
        Periapsis distance. Units: *m*

    Examples:
        ```python
        from astrojax.orbits import periapsis_distance
        rp = periapsis_distance(500e3, 0.1)
        ```
    """
    a = jnp.asarray(a, dtype=get_dtype())
    e = jnp.asarray(e, dtype=get_dtype())
    return a * (1.0 - e)


def apoapsis_distance(a: ArrayLike, e: ArrayLike) -> Array:
    """Compute the distance at apoapsis.

    Args:
        a: Semi-major axis. Units: *m*
        e: Eccentricity. Dimensionless.

    Returns:
        Apoapsis distance. Units: *m*

    Examples:
        ```python
        from astrojax.orbits import apoapsis_distance
        ra = apoapsis_distance(500e3, 0.1)
        ```
    """
    a = jnp.asarray(a, dtype=get_dtype())
    e = jnp.asarray(e, dtype=get_dtype())
    return a * (1.0 + e)


def perigee_altitude(a: ArrayLike, e: ArrayLike) -> Array:
    """Compute altitude above Earth's surface at perigee.

    Args:
        a: Semi-major axis. Units: *m*
        e: Eccentricity. Dimensionless.

    Returns:
        Perigee altitude. Units: *m*

    Examples:
        ```python
        from astrojax.constants import R_EARTH
        from astrojax.orbits import perigee_altitude
        alt = perigee_altitude(R_EARTH + 500e3, 0.01)
        ```
    """
    return periapsis_distance(a, e) - R_EARTH


def apogee_altitude(a: ArrayLike, e: ArrayLike) -> Array:
    """Compute altitude above Earth's surface at apogee.

    Args:
        a: Semi-major axis. Units: *m*
        e: Eccentricity. Dimensionless.

    Returns:
        Apogee altitude. Units: *m*

    Examples:
        ```python
        from astrojax.constants import R_EARTH
        from astrojax.orbits import apogee_altitude
        alt = apogee_altitude(R_EARTH + 500e3, 0.01)
        ```
    """
    return apoapsis_distance(a, e) - R_EARTH


# ──────────────────────────────────────────────
# Special orbits
# ──────────────────────────────────────────────


def sun_synchronous_inclination(a: ArrayLike, e: ArrayLike, use_degrees: bool = False) -> Array:
    """Compute the inclination for a Sun-synchronous orbit around Earth.

    Uses the J2 gravitational perturbation to compute the inclination
    required for the RAAN precession rate to match Earth's mean motion
    around the Sun (one revolution per year).

    Args:
        a: Semi-major axis. Units: *m*
        e: Eccentricity. Dimensionless.
        use_degrees: If ``True``, return inclination in degrees.

    Returns:
        Sun-synchronous inclination. Units: *rad* or *deg*

    Examples:
        ```python
        from astrojax.constants import R_EARTH
        from astrojax.orbits import sun_synchronous_inclination
        inc = sun_synchronous_inclination(R_EARTH + 500e3, 0.001, use_degrees=True)
        ```
    """
    a = jnp.asarray(a, dtype=get_dtype())
    e = jnp.asarray(e, dtype=get_dtype())

    # Required RAAN precession rate for sun-synchronous orbit
    omega_dot_ss = 2.0 * jnp.pi / 365.2421897 / 86400.0

    i = jnp.arccos(
        -2.0 * a**3.5 * omega_dot_ss * (1.0 - e**2) ** 2
        / (3.0 * R_EARTH**2 * J2_EARTH * jnp.sqrt(GM_EARTH))
    )
    return from_radians(i, use_degrees)


def geo_sma() -> Array:
    """Compute the semi-major axis for a geostationary orbit around Earth.

    Returns:
        Geostationary semi-major axis. Units: *m*

    Examples:
        ```python
        from astrojax.orbits import geo_sma
        a_geo = geo_sma()
        ```
    """
    return semimajor_axis_from_orbital_period(2.0 * jnp.pi / OMEGA_EARTH)


# ──────────────────────────────────────────────
# Anomaly conversions
# ──────────────────────────────────────────────


def anomaly_eccentric_to_mean(anm_ecc: ArrayLike, e: ArrayLike, use_degrees: bool = False) -> Array:
    """Convert eccentric anomaly to mean anomaly.

    Applies Kepler's equation: ``M = E - e * sin(E)``.

    Args:
        anm_ecc: Eccentric anomaly. Units: *rad* or *deg*
        e: Eccentricity. Dimensionless.
        use_degrees: If ``True``, input and output are in degrees.

    Returns:
        Mean anomaly. Units: *rad* or *deg*

    References:
        O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
        Applications*, 2012. Eq. 2.65.

    Examples:
        ```python
        from astrojax.orbits import anomaly_eccentric_to_mean
        M = anomaly_eccentric_to_mean(90.0, 0.1, use_degrees=True)
        ```
    """
    anm_ecc = jnp.asarray(anm_ecc, dtype=get_dtype())
    e = jnp.asarray(e, dtype=get_dtype())

    E = to_radians(anm_ecc, use_degrees)
    M = E - e * jnp.sin(E)
    return from_radians(M, use_degrees)


def anomaly_mean_to_eccentric(anm_mean: ArrayLike, e: ArrayLike, use_degrees: bool = False) -> Array:
    """Convert mean anomaly to eccentric anomaly.

    Solves Kepler's equation ``M = E - e * sin(E)`` for ``E`` using
    Newton-Raphson iteration implemented with ``jax.lax.fori_loop``
    for JAX traceability.

    Args:
        anm_mean: Mean anomaly. Units: *rad* or *deg*
        e: Eccentricity. Dimensionless.
        use_degrees: If ``True``, input and output are in degrees.

    Returns:
        Eccentric anomaly. Units: *rad* or *deg*

    Examples:
        ```python
        from astrojax.orbits import anomaly_mean_to_eccentric
        E = anomaly_mean_to_eccentric(84.27, 0.1, use_degrees=True)
        ```
    """
    anm_mean = jnp.asarray(anm_mean, dtype=get_dtype())
    e = jnp.asarray(e, dtype=get_dtype())

    M = to_radians(anm_mean, use_degrees)
    M = M % (2.0 * jnp.pi)

    # Initial guess: M for low eccentricity, pi for high eccentricity
    E0 = jnp.where(e < 0.8, M, jnp.pi)

    def newton_step(_, E):
        f = E - e * jnp.sin(E) - M
        E = E - f / (1.0 - e * jnp.cos(E))
        return E

    E = jax.lax.fori_loop(0, 10, newton_step, E0)
    return from_radians(E, use_degrees)


def anomaly_true_to_eccentric(anm_true: ArrayLike, e: ArrayLike, use_degrees: bool = False) -> Array:
    """Convert true anomaly to eccentric anomaly.

    Args:
        anm_true: True anomaly. Units: *rad* or *deg*
        e: Eccentricity. Dimensionless.
        use_degrees: If ``True``, input and output are in degrees.

    Returns:
        Eccentric anomaly. Units: *rad* or *deg*

    References:
        D. Vallado, *Fundamentals of Astrodynamics and Applications
        (4th Ed.)*, pp. 47, eq. 2-9, 2010.

    Examples:
        ```python
        from astrojax.orbits import anomaly_true_to_eccentric
        E = anomaly_true_to_eccentric(90.0, 0.1, use_degrees=True)
        ```
    """
    anm_true = jnp.asarray(anm_true, dtype=get_dtype())
    e = jnp.asarray(e, dtype=get_dtype())

    nu = to_radians(anm_true, use_degrees)
    E = jnp.arctan2(jnp.sin(nu) * jnp.sqrt(1.0 - e**2), jnp.cos(nu) + e)
    return from_radians(E, use_degrees)


def anomaly_eccentric_to_true(anm_ecc: ArrayLike, e: ArrayLike, use_degrees: bool = False) -> Array:
    """Convert eccentric anomaly to true anomaly.

    Args:
        anm_ecc: Eccentric anomaly. Units: *rad* or *deg*
        e: Eccentricity. Dimensionless.
        use_degrees: If ``True``, input and output are in degrees.

    Returns:
        True anomaly. Units: *rad* or *deg*

    References:
        D. Vallado, *Fundamentals of Astrodynamics and Applications
        (4th Ed.)*, pp. 47, eq. 2-9, 2010.

    Examples:
        ```python
        from astrojax.orbits import anomaly_eccentric_to_true
        nu = anomaly_eccentric_to_true(90.0, 0.1, use_degrees=True)
        ```
    """
    anm_ecc = jnp.asarray(anm_ecc, dtype=get_dtype())
    e = jnp.asarray(e, dtype=get_dtype())

    E = to_radians(anm_ecc, use_degrees)
    nu = jnp.arctan2(jnp.sin(E) * jnp.sqrt(1.0 - e**2), jnp.cos(E) - e)
    return from_radians(nu, use_degrees)


def anomaly_true_to_mean(anm_true: ArrayLike, e: ArrayLike, use_degrees: bool = False) -> Array:
    """Convert true anomaly to mean anomaly.

    Composite conversion: true -> eccentric -> mean.

    Args:
        anm_true: True anomaly. Units: *rad* or *deg*
        e: Eccentricity. Dimensionless.
        use_degrees: If ``True``, input and output are in degrees.

    Returns:
        Mean anomaly. Units: *rad* or *deg*

    Examples:
        ```python
        from astrojax.orbits import anomaly_true_to_mean
        M = anomaly_true_to_mean(90.0, 0.1, use_degrees=True)
        ```
    """
    return anomaly_eccentric_to_mean(
        anomaly_true_to_eccentric(anm_true, e, use_degrees),
        e,
        use_degrees,
    )


def anomaly_mean_to_true(anm_mean: ArrayLike, e: ArrayLike, use_degrees: bool = False) -> Array:
    """Convert mean anomaly to true anomaly.

    Composite conversion: mean -> eccentric -> true.

    Args:
        anm_mean: Mean anomaly. Units: *rad* or *deg*
        e: Eccentricity. Dimensionless.
        use_degrees: If ``True``, input and output are in degrees.

    Returns:
        True anomaly. Units: *rad* or *deg*

    Examples:
        ```python
        from astrojax.orbits import anomaly_mean_to_true
        nu = anomaly_mean_to_true(90.0, 0.1, use_degrees=True)
        ```
    """
    return anomaly_eccentric_to_true(
        anomaly_mean_to_eccentric(anm_mean, e, use_degrees),
        e,
        use_degrees,
    )
