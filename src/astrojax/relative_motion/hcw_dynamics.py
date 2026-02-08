"""Hill-Clohessy-Wiltshire (HCW) relative motion dynamics.

Provides the state derivative for linearised relative motion about a
circular reference orbit.  The HCW equations describe a deputy
satellite's motion relative to a chief on a circular Keplerian orbit
in the chief's RTN frame.

The unforced equations of motion are:

.. math::

    \\ddot{x} &=  3n^2 x + 2n\\dot{y} \\\\
    \\ddot{y} &= -2n\\dot{x} \\\\
    \\ddot{z} &= -n^2 z

where *n* is the mean motion of the chief's circular orbit and
*(x, y, z)* are the RTN components of relative position.

All inputs and outputs use SI base units (metres, metres/second,
radians/second).

References:
    1. W. H. Clohessy and R. S. Wiltshire, "Terminal Guidance System
       for Satellite Rendezvous", *Journal of the Aerospace Sciences*,
       vol. 27, no. 9, pp. 653-658, 1960.
    2. D. Vallado, *Fundamentals of Astrodynamics and Applications*
       (4th Ed.), Microcosm Press, 2013, sec. 6.8.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype


def hcw_derivative(state: ArrayLike, n: ArrayLike) -> Array:
    """Compute the HCW state derivative for unforced relative motion.

    This is a pure function suitable for use with any numerical integrator.
    It is compatible with ``jax.jit``, ``jax.vmap``, and ``jax.grad``.

    Args:
        state: 6-element relative state in RTN
            ``[x, y, z, x_dot, y_dot, z_dot]``. Units: m, m/s.
        n: Mean motion of the chief orbit. Units: rad/s.

    Returns:
        6-element state derivative
            ``[x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]``.
            Units: m/s, m/s^2.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.relative_motion import hcw_derivative
        from astrojax.constants import GM_EARTH, R_EARTH
        sma = R_EARTH + 500e3
        n = jnp.sqrt(GM_EARTH / sma**3)
        state = jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        deriv = hcw_derivative(state, n)
        float(deriv[3])  # x_ddot = 3 n^2 x
        ```
    """
    state = jnp.asarray(state, dtype=get_dtype())
    n = jnp.asarray(n, dtype=get_dtype())

    x = state[0]
    z = state[2]
    x_dot = state[3]
    y_dot = state[4]
    z_dot = state[5]

    n2 = n * n

    x_ddot = 3.0 * n2 * x + 2.0 * n * y_dot
    y_ddot = -2.0 * n * x_dot
    z_ddot = -n2 * z

    return jnp.array([x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot])
