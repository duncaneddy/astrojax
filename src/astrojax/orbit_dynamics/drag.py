"""Atmospheric drag acceleration model.

Computes the non-conservative acceleration due to atmospheric drag on
a spacecraft, accounting for the relative velocity between the
spacecraft and the co-rotating atmosphere.

All inputs and outputs use SI base units (metres, metres/second,
metres/second squared, kg, kg/m^3).

References:
    1. O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods
       and Applications*, 2012.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.constants import OMEGA_EARTH


def accel_drag(
    x: ArrayLike,
    density: float,
    mass: float,
    area: float,
    cd: float,
    T: ArrayLike,
) -> Array:
    """Acceleration due to atmospheric drag.

    Transforms the state to the ECEF (ITRF) frame via *T*,
    computes the velocity relative to the co-rotating atmosphere, and
    returns the drag acceleration in the inertial (ECI) frame.

    Args:
        x: 6-element inertial state ``[r, v]`` [m; m/s].
        density: Atmospheric density [kg/m^3].
        mass: Spacecraft mass [kg].
        area: Wind-facing cross-sectional area [m^2].
        cd: Coefficient of drag [dimensionless].
        T: 3x3 rotation matrix from ECI to ECEF (ITRF).

    Returns:
        Drag acceleration in ECI [m/s^2], shape ``(3,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.orbit_dynamics import accel_drag
        x = jnp.array([6878e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
        a = accel_drag(x, 1e-12, 1000.0, 1.0, 2.0, jnp.eye(3))
        ```
    """
    _float = get_dtype()
    x = jnp.asarray(x, dtype=_float)
    T = jnp.asarray(T, dtype=_float)

    r_eci = x[:3]
    v_eci = x[3:6]

    # Position and velocity in true-of-date frame
    r_tod = T @ r_eci
    v_tod = T @ v_eci

    # Earth rotation vector
    omega = jnp.array([_float(0.0), _float(0.0), _float(OMEGA_EARTH)])

    # Velocity relative to co-rotating atmosphere
    v_rel = v_tod - jnp.cross(omega, r_tod)
    v_abs = jnp.linalg.norm(v_rel)

    # Drag acceleration in TOD frame
    a_tod = (
        _float(-0.5) * _float(cd) * (_float(area) / _float(mass)) * _float(density) * v_abs * v_rel
    )

    # Transform back to ECI
    return T.T @ a_tod
