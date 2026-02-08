"""Point-mass gravity force models.

Provides the gravitational acceleration due to a point-mass central body.
All inputs and outputs use SI base units (metres, metres/second squared).

References:
    1. O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods
       and Applications*, 2012.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.constants import GM_EARTH


def accel_point_mass(
    r_object: ArrayLike,
    r_body: ArrayLike,
    gm: float,
) -> Array:
    """Acceleration due to point-mass gravity.

    Computes the gravitational acceleration on *r_object* due to a body
    at *r_body* with gravitational parameter *gm*.  When the central body
    is at the origin (``r_body = [0, 0, 0]``), the standard two-body
    expression ``-gm * r / |r|^3`` is used.  Otherwise the indirect
    (third-body) form is applied.

    Args:
        r_object: Position of the object [m].  Shape ``(3,)`` or ``(6,)``
            (only first 3 elements used).
        r_body: Position of the attracting body [m].  Shape ``(3,)``.
        gm: Gravitational parameter of the attracting body [m^3/s^2].

    Returns:
        jax.Array: Acceleration vector [m/s^2], shape ``(3,)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax.constants import R_EARTH, GM_EARTH
        >>> from astrojax.orbit_dynamics import accel_point_mass
        >>> r = jnp.array([R_EARTH, 0.0, 0.0])
        >>> a = accel_point_mass(r, jnp.zeros(3), GM_EARTH)
    """
    _float = get_dtype()
    r_obj = jnp.asarray(r_object, dtype=_float)[:3]
    r_cb = jnp.asarray(r_body, dtype=_float)

    d = r_obj - r_cb
    d_norm = jnp.linalg.norm(d)
    r_cb_norm = jnp.linalg.norm(r_cb)

    # Third-body form (r_body != 0): -gm * (d/|d|^3 + r_body/|r_body|^3)
    # Central-body form (r_body = 0): -gm * d/|d|^3
    a_third = -gm * (d / d_norm**3 + r_cb / r_cb_norm**3)
    a_central = -gm * d / d_norm**3

    return jnp.where(r_cb_norm > _float(0.0), a_third, a_central)


def accel_gravity(r_object: ArrayLike) -> Array:
    """Acceleration due to Earth's point-mass gravity.

    Convenience wrapper for :func:`accel_point_mass` with Earth's
    gravitational parameter and the central body at the origin.

    Args:
        r_object: Position of the object in ECI [m].  Shape ``(3,)`` or
            ``(6,)`` (only first 3 elements used).

    Returns:
        jax.Array: Acceleration vector [m/s^2], shape ``(3,)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax.constants import R_EARTH
        >>> from astrojax.orbit_dynamics import accel_gravity
        >>> r = jnp.array([R_EARTH, 0.0, 0.0])
        >>> a = accel_gravity(r)
    """
    return accel_point_mass(r_object, jnp.zeros(3, dtype=get_dtype()), GM_EARTH)
