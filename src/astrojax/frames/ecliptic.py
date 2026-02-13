"""Ecliptic-ICRF frame transformations.

Provides rotation matrices and state-vector transformations between the
ecliptic coordinate frame and the International Celestial Reference Frame
(ICRF), using the J2000 mean obliquity of the ecliptic.

Since both frames are inertial (neither rotates), the transformation is a
fixed rotation about the x-axis by the mean obliquity angle. No epoch,
EOP data, or Coriolis velocity correction is needed.

The obliquity angle used is the IAU 2006 value at J2000:
84381.406 arcseconds (approximately 23.439°).

All inputs and outputs use SI base units (metres, metres/second).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.attitude_representations import Rx
from astrojax.config import get_dtype
from astrojax.constants import AS2RAD, OBLIQUITY_J2000

# Mean obliquity of the ecliptic at J2000 in radians
_OBLIQUITY_RAD = OBLIQUITY_J2000 * AS2RAD


def rotation_ecliptic_to_icrf() -> Array:
    """Compute the 3x3 rotation matrix from ecliptic to ICRF.

    Returns the matrix ``Rx(-ε)`` where ε is the J2000 mean obliquity.

    Returns:
        3x3 rotation matrix (ecliptic -> ICRF).

    Examples:
        ```python
        from astrojax.frames import rotation_ecliptic_to_icrf
        R = rotation_ecliptic_to_icrf()
        R.shape
        ```
    """
    return Rx(-_OBLIQUITY_RAD)


def rotation_icrf_to_ecliptic() -> Array:
    """Compute the 3x3 rotation matrix from ICRF to ecliptic.

    Returns the matrix ``Rx(ε)`` where ε is the J2000 mean obliquity.
    This is the transpose of :func:`rotation_ecliptic_to_icrf`.

    Returns:
        3x3 rotation matrix (ICRF -> ecliptic).

    Examples:
        ```python
        from astrojax.frames import rotation_icrf_to_ecliptic
        R = rotation_icrf_to_ecliptic()
        R.shape
        ```
    """
    return Rx(_OBLIQUITY_RAD)


def state_ecliptic_to_icrf(x_ecl: ArrayLike) -> Array:
    """Transform a 6-element state vector from ecliptic to ICRF.

    Both position and velocity are rotated identically (no Coriolis
    correction) since both frames are inertial.

    Args:
        x_ecl: 6-element ecliptic state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        6-element ICRF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.frames import state_ecliptic_to_icrf
        x_ecl = jnp.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        x_icrf = state_ecliptic_to_icrf(x_ecl)
        ```
    """
    x_ecl = jnp.asarray(x_ecl, dtype=get_dtype())
    R = rotation_ecliptic_to_icrf()

    r_icrf = R @ x_ecl[:3]
    v_icrf = R @ x_ecl[3:6]

    return jnp.concatenate([r_icrf, v_icrf])


def state_icrf_to_ecliptic(x_icrf: ArrayLike) -> Array:
    """Transform a 6-element state vector from ICRF to ecliptic.

    Both position and velocity are rotated identically (no Coriolis
    correction) since both frames are inertial.

    Args:
        x_icrf: 6-element ICRF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        6-element ecliptic state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.frames import state_icrf_to_ecliptic
        x_icrf = jnp.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        x_ecl = state_icrf_to_ecliptic(x_icrf)
        ```
    """
    x_icrf = jnp.asarray(x_icrf, dtype=get_dtype())
    R = rotation_icrf_to_ecliptic()

    r_ecl = R @ x_icrf[:3]
    v_ecl = R @ x_icrf[3:6]

    return jnp.concatenate([r_ecl, v_ecl])
