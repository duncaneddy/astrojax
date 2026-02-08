"""Third-body gravitational perturbations from the Sun and Moon.

Computes the gravitational acceleration on a spacecraft due to the Sun
and Moon using low-precision analytical ephemerides and the point-mass
gravity model.

All inputs and outputs use SI base units (metres, metres/second squared).

References:
    1. O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods
       and Applications*, 2012.
"""

from __future__ import annotations

from jax import Array
from jax.typing import ArrayLike

from astrojax.constants import GM_MOON, GM_SUN
from astrojax.epoch import Epoch
from astrojax.orbit_dynamics.ephemerides import moon_position, sun_position
from astrojax.orbit_dynamics.gravity import accel_point_mass


def accel_third_body_sun(epc: Epoch, r_object: ArrayLike) -> Array:
    """Acceleration due to the Sun's gravity on a near-Earth object.

    Computes the Sun's position at *epc* using the low-precision
    analytical ephemeris and applies the point-mass gravity model.

    Args:
        epc: Epoch at which to evaluate.
        r_object: Position of the object in ECI [m].  Shape ``(3,)``
            or ``(6,)`` (only first 3 elements used).

    Returns:
        Acceleration vector [m/s^2], shape ``(3,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax import Epoch
        from astrojax.orbit_dynamics import accel_third_body_sun
        epc = Epoch(2024, 2, 25)
        r = jnp.array([6878e3, 0.0, 0.0])
        a = accel_third_body_sun(epc, r)
        ```
    """
    r_sun = sun_position(epc)
    return accel_point_mass(r_object, r_sun, GM_SUN)


def accel_third_body_moon(epc: Epoch, r_object: ArrayLike) -> Array:
    """Acceleration due to the Moon's gravity on a near-Earth object.

    Computes the Moon's position at *epc* using the low-precision
    analytical ephemeris and applies the point-mass gravity model.

    Args:
        epc: Epoch at which to evaluate.
        r_object: Position of the object in ECI [m].  Shape ``(3,)``
            or ``(6,)`` (only first 3 elements used).

    Returns:
        Acceleration vector [m/s^2], shape ``(3,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax import Epoch
        from astrojax.orbit_dynamics import accel_third_body_moon
        epc = Epoch(2024, 2, 25)
        r = jnp.array([6878e3, 0.0, 0.0])
        a = accel_third_body_moon(epc, r)
        ```
    """
    r_moon = moon_position(epc)
    return accel_point_mass(r_object, r_moon, GM_MOON)
