"""Core rigid-body attitude dynamics equations.

Provides pure JAX functions for quaternion kinematics and Euler's
rotational equation of motion:

- :func:`quaternion_derivative` -- quaternion time-derivative from
  angular velocity using the 4x4 Omega matrix form.
- :func:`euler_equation` -- angular acceleration from Euler's equation
  with external torque.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype


def quaternion_derivative(q: ArrayLike, omega: ArrayLike) -> Array:
    """Compute the quaternion time-derivative from angular velocity.

    Uses the 4x4 Omega matrix form::

        q_dot = 0.5 * Omega(omega) @ q

    where the Omega matrix for scalar-first ``[w, x, y, z]`` quaternions is::

            [  0   -wx  -wy  -wz ]
        O = [ wx    0    wz  -wy ]
            [ wy  -wz    0    wx ]
            [ wz   wy  -wx    0  ]

    Args:
        q: Unit quaternion ``[w, x, y, z]`` of shape ``(4,)``.
        omega: Angular velocity in the body frame ``[wx, wy, wz]``
            of shape ``(3,)`` [rad/s].

    Returns:
        Quaternion derivative of shape ``(4,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        q = jnp.array([1.0, 0.0, 0.0, 0.0])  # identity
        omega = jnp.array([0.0, 0.0, 0.1])    # yaw rate
        q_dot = quaternion_derivative(q, omega)
        q_dot.shape
        ```
    """
    _float = get_dtype()
    q = jnp.asarray(q, dtype=_float)
    omega = jnp.asarray(omega, dtype=_float)

    wx, wy, wz = omega[0], omega[1], omega[2]

    Omega = jnp.array([
        [0.0, -wx, -wy, -wz],
        [wx, 0.0, wz, -wy],
        [wy, -wz, 0.0, wx],
        [wz, wy, -wx, 0.0],
    ], dtype=_float)

    return 0.5 * Omega @ q


def euler_equation(
    omega: ArrayLike,
    I: ArrayLike,  # noqa: E741
    tau: ArrayLike,
) -> Array:
    """Compute angular acceleration from Euler's rotational equation.

    Euler's equation for a rigid body::

        I @ omega_dot = -omega x (I @ omega) + tau
        omega_dot = I^{-1} @ (-omega x (I @ omega) + tau)

    Uses ``jnp.linalg.solve`` instead of an explicit inverse for
    numerical stability.

    Args:
        omega: Angular velocity in the body frame ``[wx, wy, wz]``
            of shape ``(3,)`` [rad/s].
        I: Inertia tensor of shape ``(3, 3)`` [kg m^2].
        tau: Total external torque in the body frame ``[tx, ty, tz]``
            of shape ``(3,)`` [N m].

    Returns:
        Angular acceleration ``[dwx, dwy, dwz]``
            of shape ``(3,)`` [rad/s^2].

    Examples:
        ```python
        import jax.numpy as jnp
        omega = jnp.array([0.1, 0.0, 0.0])
        I = jnp.diag(jnp.array([10.0, 20.0, 30.0]))
        tau = jnp.zeros(3)
        omega_dot = euler_equation(omega, I, tau)
        omega_dot.shape
        ```
    """
    _float = get_dtype()
    omega = jnp.asarray(omega, dtype=_float)
    I = jnp.asarray(I, dtype=_float)  # noqa: E741
    tau = jnp.asarray(tau, dtype=_float)

    rhs = -jnp.cross(omega, I @ omega) + tau
    return jnp.linalg.solve(I, rhs)
