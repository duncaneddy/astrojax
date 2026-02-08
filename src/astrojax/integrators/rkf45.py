"""Runge-Kutta-Fehlberg 4(5) adaptive integrator (RKF45).

Implements the Fehlberg embedded Runge-Kutta method with a 5th-order solution
for propagation and a 4th-order solution for error estimation. The method uses
6 stages per step.

Adaptive step-size control automatically adjusts the timestep to maintain
the error within specified tolerances. Steps that exceed the tolerance are
rejected and retried with a smaller timestep using ``jax.lax.while_loop``
for JIT compatibility.

The Butcher tableau coefficients are taken from the standard Fehlberg
formulation:

- Nodes (c): [0, 1/4, 3/8, 12/13, 1, 1/2]
- 5th-order weights (b_high): [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
- 4th-order weights (b_low): [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.integrators._adaptive import compute_error_norm, compute_next_step_size
from astrojax.integrators._types import AdaptiveConfig, StepResult

# Butcher tableau coefficients as Python tuples (cast at call time).
# Nodes
_C = (0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0)

# Coupling coefficients (lower-triangular rows)
_A1 = (1.0 / 4.0,)
_A2 = (3.0 / 32.0, 9.0 / 32.0)
_A3 = (1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0)
_A4 = (439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0)
_A5 = (-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0)

# 5th-order weights (primary solution)
_B_HIGH = (16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0)

# 4th-order weights (error estimation)
_B_LOW = (25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0)


def rkf45_step(
    dynamics: Callable[[ArrayLike, ArrayLike], Array],
    t: ArrayLike,
    state: ArrayLike,
    dt: ArrayLike,
    config: AdaptiveConfig | None = None,
    control: Callable[[ArrayLike, ArrayLike], Array] | None = None,
) -> StepResult:
    """Perform a single adaptive RKF45 integration step.

    Advances the state from time ``t`` by up to ``dt`` using the
    Runge-Kutta-Fehlberg 4(5) method with adaptive step-size control.
    If the error exceeds the tolerance, the step is rejected and retried
    with a smaller timestep.

    Compatible with ``jax.jit`` and ``jax.vmap``. Not compatible with
    reverse-mode ``jax.grad`` due to the internal ``lax.while_loop``.

    Args:
        dynamics: ODE right-hand side function ``f(t, x) -> dx/dt``.
        t: Current time.
        state: Current state vector.
        dt: Requested timestep. May be negative for backward integration.
            The actual timestep used may be smaller if the adaptive
            controller rejects the initial attempt.
        config: Adaptive step-size configuration. Uses default
            :class:`AdaptiveConfig` if ``None``.
        control: Optional additive control function ``u(t, x) -> force``.
            When provided, the effective derivative is
            ``f(t, x) + u(t, x)``.

    Returns:
        StepResult: Named tuple with fields:
            - ``state``: State at ``t + dt_used``.
            - ``dt_used``: Actual timestep taken (<= ``|dt|``).
            - ``error_estimate``: Normalized error of the accepted step.
            - ``dt_next``: Suggested timestep for the next step.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.integrators import rkf45_step
        def harmonic(t, x):
            return jnp.array([x[1], -x[0]])
        result = rkf45_step(harmonic, 0.0, jnp.array([1.0, 0.0]), 0.1)
        result.state  # ~[cos(0.1), -sin(0.1)]
        ```
    """
    if config is None:
        config = AdaptiveConfig()

    dtype = get_dtype()
    t = jnp.asarray(t, dtype=dtype)
    state = jnp.asarray(state, dtype=dtype)
    dt = jnp.asarray(dt, dtype=dtype)

    def f(ti, xi):
        dx = dynamics(ti, xi)
        if control is not None:
            dx = dx + control(ti, xi)
        return dx

    def _attempt_step(h):
        """Compute one RKF45 trial step with step size h."""
        k0 = f(t, state)
        k1 = f(t + _C[1] * h, state + h * _A1[0] * k0)
        k2 = f(t + _C[2] * h, state + h * (_A2[0] * k0 + _A2[1] * k1))
        k3 = f(t + _C[3] * h, state + h * (_A3[0] * k0 + _A3[1] * k1 + _A3[2] * k2))
        k4 = f(
            t + _C[4] * h,
            state + h * (_A4[0] * k0 + _A4[1] * k1 + _A4[2] * k2 + _A4[3] * k3),
        )
        k5 = f(
            t + _C[5] * h,
            state + h * (_A5[0] * k0 + _A5[1] * k1 + _A5[2] * k2 + _A5[3] * k3 + _A5[4] * k4),
        )

        # 5th-order solution (primary)
        state_high = state + h * (
            _B_HIGH[0] * k0 + _B_HIGH[2] * k2 + _B_HIGH[3] * k3 + _B_HIGH[4] * k4 + _B_HIGH[5] * k5
        )

        # 4th-order solution (for error estimation)
        state_low = state + h * (_B_LOW[0] * k0 + _B_LOW[2] * k2 + _B_LOW[3] * k3 + _B_LOW[4] * k4)

        error_vec = state_high - state_low
        error = compute_error_norm(error_vec, state_high, state, config.abs_tol, config.rel_tol)
        return state_high, error

    # Adaptive step-rejection loop via lax.while_loop.
    # Carry: (h, attempts, accepted, state_out, error_out)
    def cond_fn(carry):
        _h, attempts, accepted, _state_out, _error_out = carry
        return (~accepted) & (attempts < config.max_step_attempts)

    def body_fn(carry):
        h, attempts, _accepted, _state_out, _error_out = carry
        state_new, error = _attempt_step(h)

        at_min_step = jnp.abs(h) <= config.min_step
        step_accepted = (error <= 1.0) | at_min_step

        # If rejected, shrink step size for next attempt
        h_reduced = compute_next_step_size(
            error,
            h,
            4.0,
            config.safety_factor,
            config.min_scale_factor,
            config.max_scale_factor,
            config.min_step,
            config.max_step,
        )
        h_next = jnp.where(step_accepted, h, h_reduced)

        return (h_next, attempts + 1, step_accepted, state_new, error)

    init_carry = (
        dt,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        state,
        jnp.asarray(jnp.inf, dtype=dtype),
    )

    h_final, _attempts, _accepted, state_out, error_out = jax.lax.while_loop(
        cond_fn, body_fn, init_carry
    )

    # Compute suggested next step size from accepted error
    dt_next = compute_next_step_size(
        error_out,
        h_final,
        4.0,
        config.safety_factor,
        config.min_scale_factor,
        config.max_scale_factor,
        config.min_step,
        config.max_step,
    )

    return StepResult(
        state=state_out,
        dt_used=h_final,
        error_estimate=error_out,
        dt_next=dt_next,
    )
