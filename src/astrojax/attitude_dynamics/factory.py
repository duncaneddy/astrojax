"""Configurable attitude dynamics factory.

Composes individual torque models into a single
``dynamics(t, state) -> derivative`` closure compatible with all
astrojax integrators.

The factory captures static configuration at Python trace time — boolean
toggles like ``config.gravity_gradient`` become Python ``if`` branches
that are resolved during ``jax.jit`` tracing, producing an optimized
computation graph with no runtime branching overhead.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.attitude_dynamics.config import AttitudeDynamicsConfig
from astrojax.attitude_dynamics.euler_dynamics import (
    euler_equation,
    quaternion_derivative,
)
from astrojax.attitude_dynamics.gravity_gradient import torque_gravity_gradient


def create_attitude_dynamics(
    config: AttitudeDynamicsConfig,
    position_fn: Callable[[ArrayLike], Array],
) -> Callable[[ArrayLike, ArrayLike], Array]:
    """Create a configurable attitude dynamics function.

    Returns a closure ``dynamics(t, state) -> derivative`` that computes
    the time-derivative of a 7-element attitude state vector, composing
    the selected torque models from *config*.

    The returned function is compatible with all astrojax integrators
    (``rk4_step``, ``rkf45_step``, ``dp54_step``).

    Args:
        config: Attitude dynamics configuration.
        position_fn: Callable ``position_fn(t) -> Array(3,)`` returning
            the spacecraft ECI position at time *t* [m].  Required even
            for torque-free motion (not called if unused).

    Returns:
        A callable ``dynamics(t, state) -> derivative`` where:

        - *t*: time (scalar), typically seconds since a reference epoch.
        - *state*: ``[q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z]``,
          shape ``(7,)``.
        - *derivative*: ``[dq_w, dq_x, dq_y, dq_z, domega_x, domega_y,
          domega_z]``, shape ``(7,)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax.attitude_dynamics.config import (
        ...     AttitudeDynamicsConfig, SpacecraftInertia,
        ... )
        >>> from astrojax.attitude_dynamics.factory import create_attitude_dynamics
        >>> from astrojax.integrators import rk4_step
        >>> inertia = SpacecraftInertia.from_principal(10.0, 20.0, 30.0)
        >>> config = AttitudeDynamicsConfig.torque_free(inertia)
        >>> pos_fn = lambda t: jnp.array([7000e3, 0.0, 0.0])
        >>> dynamics = create_attitude_dynamics(config, pos_fn)
        >>> x0 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
        >>> result = rk4_step(dynamics, 0.0, x0, 1.0)
    """
    # Capture static configuration into local variables for the closure.
    _I = config.inertia.I
    _gravity_gradient = config.gravity_gradient
    _mu = config.mu

    def dynamics(t: ArrayLike, state: ArrayLike) -> Array:
        """Attitude dynamics: state derivative.

        Args:
            t: Current time (scalar).
            state: ``[q_w, q_x, q_y, q_z, wx, wy, wz]``, shape ``(7,)``.

        Returns:
            jax.Array: ``[dq_w, dq_x, dq_y, dq_z, dwx, dwy, dwz]``,
                shape ``(7,)``.
        """
        _float = get_dtype()
        state = jnp.asarray(state, dtype=_float)

        q = state[:4]
        omega = state[4:7]

        # --- Quaternion kinematics ---
        q_dot = quaternion_derivative(q, omega)

        # --- External torques ---
        tau = jnp.zeros(3, dtype=_float)

        if _gravity_gradient:
            r_eci = position_fn(t)
            tau = tau + torque_gravity_gradient(q, r_eci, _I, _mu)

        # --- Euler's equation ---
        omega_dot = euler_equation(omega, _I, tau)

        return jnp.concatenate([q_dot, omega_dot])

    return dynamics
