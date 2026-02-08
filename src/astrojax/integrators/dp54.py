"""Dormand-Prince 5(4) adaptive integrator (DP54).

Implements the Dormand-Prince embedded Runge-Kutta method with a 5th-order
solution for propagation and a 4th-order solution for error estimation. The
method uses 7 stages per step.

The Dormand-Prince method has the First-Same-As-Last (FSAL) property: the
7th stage of step *n* is identical to the 1st stage of step *n+1* when the
step is accepted. This implementation does **not** cache the FSAL stage
internally (to remain purely functional for JAX compatibility), but users
can manage this externally if needed.

Adaptive step-size control automatically adjusts the timestep to maintain
the error within specified tolerances. Steps that exceed the tolerance are
rejected and retried with a smaller timestep using ``jax.lax.while_loop``
for JIT compatibility.

The Butcher tableau coefficients are the standard Dormand-Prince values:

- Nodes (c): [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
- 5th-order weights (b_high): [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
- 4th-order weights (b_low): [5179/57600, 0, 7571/16695, 393/640, -92097/339200,
  187/2100, 1/40]
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.integrators._adaptive import compute_error_norm, compute_next_step_size
from astrojax.integrators._types import AdaptiveConfig, StepResult

# Butcher tableau coefficients as Python tuples (cast at call time).
# Nodes
_C = (0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0)

# Coupling coefficients (lower-triangular rows)
_A1 = (1.0 / 5.0,)
_A2 = (3.0 / 40.0, 9.0 / 40.0)
_A3 = (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0)
_A4 = (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0)
_A5 = (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0)
_A6 = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0)

# 5th-order weights (primary solution) — same as _A6
_B_HIGH = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0)

# 4th-order weights (error estimation)
_B_LOW = (
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
)


def dp54_step(
    dynamics: Callable[[ArrayLike, ArrayLike], Array],
    t: ArrayLike,
    state: ArrayLike,
    dt: ArrayLike,
    config: Optional[AdaptiveConfig] = None,
    control: Optional[Callable[[ArrayLike, ArrayLike], Array]] = None,
) -> StepResult:
    """Perform a single adaptive DP54 integration step.

    Advances the state from time ``t`` by up to ``dt`` using the
    Dormand-Prince 5(4) method with adaptive step-size control. If the
    error exceeds the tolerance, the step is rejected and retried with
    a smaller timestep.

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
        from astrojax.integrators import dp54_step
        def harmonic(t, x):
            return jnp.array([x[1], -x[0]])
        result = dp54_step(harmonic, 0.0, jnp.array([1.0, 0.0]), 0.1)
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
        """Compute one DP54 trial step with step size h."""
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
            state
            + h * (_A5[0] * k0 + _A5[1] * k1 + _A5[2] * k2 + _A5[3] * k3 + _A5[4] * k4),
        )
        k6 = f(
            t + _C[6] * h,
            state
            + h
            * (
                _A6[0] * k0
                + _A6[2] * k2
                + _A6[3] * k3
                + _A6[4] * k4
                + _A6[5] * k5
            ),
        )

        # 5th-order solution (primary) — note _B_HIGH[1] = _B_HIGH[6] = 0
        state_high = state + h * (
            _B_HIGH[0] * k0
            + _B_HIGH[2] * k2
            + _B_HIGH[3] * k3
            + _B_HIGH[4] * k4
            + _B_HIGH[5] * k5
        )

        # 4th-order solution (for error estimation)
        state_low = state + h * (
            _B_LOW[0] * k0
            + _B_LOW[2] * k2
            + _B_LOW[3] * k3
            + _B_LOW[4] * k4
            + _B_LOW[5] * k5
            + _B_LOW[6] * k6
        )

        error_vec = state_high - state_low
        error = compute_error_norm(
            error_vec, state_high, state, config.abs_tol, config.rel_tol
        )
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
            error, h, 4.0, config.safety_factor,
            config.min_scale_factor, config.max_scale_factor,
            config.min_step, config.max_step,
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
        error_out, h_final, 4.0, config.safety_factor,
        config.min_scale_factor, config.max_scale_factor,
        config.min_step, config.max_step,
    )

    return StepResult(
        state=state_out,
        dt_used=h_final,
        error_estimate=error_out,
        dt_next=dt_next,
    )
