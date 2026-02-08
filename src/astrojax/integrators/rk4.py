"""Classic 4th-order Runge-Kutta integrator (RK4).

Implements the standard four-stage, 4th-order explicit Runge-Kutta method
for numerical integration of ordinary differential equations. This is a
fixed-step method with no adaptive step-size control.

The Butcher tableau for RK4 is:

.. math::

    \\begin{array}{c|cccc}
    0   &     &     &     &   \\\\
    1/2 & 1/2 &     &     &   \\\\
    1/2 &  0  & 1/2 &     &   \\\\
    1   &  0  &  0  &  1  &   \\\\
    \\hline
        & 1/6 & 1/3 & 1/3 & 1/6
    \\end{array}

The method achieves 4th-order accuracy, meaning the local truncation error
is :math:`O(h^5)` and the global error is :math:`O(h^4)`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.integrators._types import StepResult


def rk4_step(
    dynamics: Callable[[ArrayLike, ArrayLike], Array],
    t: ArrayLike,
    state: ArrayLike,
    dt: ArrayLike,
    control: Callable[[ArrayLike, ArrayLike], Array] | None = None,
) -> StepResult:
    """Perform a single RK4 integration step.

    Advances the state from time ``t`` to ``t + dt`` using the classic
    4th-order Runge-Kutta method. Compatible with ``jax.jit`` and
    ``jax.vmap``.

    Args:
        dynamics: ODE right-hand side function ``f(t, x) -> dx/dt``.
        t: Current time.
        state: Current state vector.
        dt: Timestep to take. May be negative for backward integration.
        control: Optional additive control function ``u(t, x) -> force``.
            When provided, the effective derivative is
            ``f(t, x) + u(t, x)``.

    Returns:
        StepResult: Named tuple with fields:
            - ``state``: State at ``t + dt``.
            - ``dt_used``: Always equals ``dt``.
            - ``error_estimate``: Always 0.0 (no error estimate for
              fixed-step methods).
            - ``dt_next``: Always equals ``dt``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.integrators import rk4_step
        def harmonic(t, x):
            return jnp.array([x[1], -x[0]])
        result = rk4_step(harmonic, 0.0, jnp.array([1.0, 0.0]), 0.01)
        result.state  # ~[cos(0.01), -sin(0.01)]
        ```
    """
    dtype = get_dtype()
    t = jnp.asarray(t, dtype=dtype)
    state = jnp.asarray(state, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)

    def f(ti, xi):
        dx = dynamics(ti, xi)
        if control is not None:
            dx = dx + control(ti, xi)
        return dx

    k1 = f(t, state)
    k2 = f(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = f(t + dt, state + dt * k3)

    state_new = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return StepResult(
        state=state_new,
        dt_used=dt,
        error_estimate=jnp.asarray(0.0, dtype=dtype),
        dt_next=dt,
    )
