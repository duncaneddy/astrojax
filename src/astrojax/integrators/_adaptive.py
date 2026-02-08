"""Adaptive step-size control utilities for embedded Runge-Kutta methods.

Provides shared error-norm computation and step-size adjustment logic used
by both the RKF45 and DP54 integrators. The algorithms follow the standard
embedded Runge-Kutta error control approach:

1. Compute a normalized error using mixed absolute/relative tolerances.
2. Accept the step if the normalized error is <= 1.0.
3. Predict the next step size using the error and the method order.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype


def compute_error_norm(
    error_vec: ArrayLike,
    state_new: ArrayLike,
    state_old: ArrayLike,
    abs_tol: float,
    rel_tol: float,
) -> Array:
    """Compute the normalized error norm for adaptive step-size control.

    Uses a mixed absolute/relative tolerance per component with the infinity
    norm (maximum over components). The step is accepted when the returned
    value is <= 1.0.

    The per-component tolerance is:

    .. math::

        \\text{tol}_i = \\text{abs\\_tol} + \\text{rel\\_tol}
            \\cdot \\max(|y^{\\text{new}}_i|, |y^{\\text{old}}_i|)

    Args:
        error_vec: Difference between high-order and low-order solutions.
        state_new: High-order solution (accepted state).
        state_old: State at the beginning of the step.
        abs_tol: Absolute error tolerance.
        rel_tol: Relative error tolerance.

    Returns:
        jax.Array: Scalar normalized error. Step is accepted if <= 1.0.
    """
    error_vec = jnp.asarray(error_vec, dtype=get_dtype())
    state_new = jnp.asarray(state_new, dtype=get_dtype())
    state_old = jnp.asarray(state_old, dtype=get_dtype())

    scale = abs_tol + rel_tol * jnp.maximum(jnp.abs(state_new), jnp.abs(state_old))
    return jnp.max(jnp.abs(error_vec) / scale)


def compute_next_step_size(
    error: ArrayLike,
    h: ArrayLike,
    order: float,
    safety_factor: float,
    min_scale_factor: float,
    max_scale_factor: float,
    min_step: float,
    max_step: float,
) -> Array:
    """Compute the next step size based on the current error estimate.

    Uses the standard optimal step-size formula:

    .. math::

        h_{\\text{next}} = |h| \\cdot S \\cdot
            \\left(\\frac{1}{\\text{error}}\\right)^{1/(p+1)}

    where *S* is the safety factor and *p* is the method order. The result
    is clamped by scale-factor bounds and absolute step-size bounds, and
    the sign of ``h`` is preserved for backward integration.

    Args:
        error: Normalized error from :func:`compute_error_norm`.
        h: Current step size (may be negative for backward integration).
        order: Order of the error estimator (e.g. 4.0 for RKF45, 4.0 for DP54).
        safety_factor: Multiplicative safety factor (typically 0.9).
        min_scale_factor: Minimum allowed ratio ``|h_next| / |h|``.
        max_scale_factor: Maximum allowed ratio ``|h_next| / |h|``.
        min_step: Absolute minimum step size.
        max_step: Absolute maximum step size.

    Returns:
        jax.Array: Suggested next step size with same sign as ``h``.
    """
    error = jnp.asarray(error, dtype=get_dtype())
    h = jnp.asarray(h, dtype=get_dtype())

    abs_h = jnp.abs(h)
    sign_h = jnp.sign(h)

    exponent = 1.0 / (order + 1.0)
    raw_scale = jnp.where(error > 0.0, jnp.power(1.0 / error, exponent), max_scale_factor)
    scale = safety_factor * raw_scale

    # Clamp scale factor
    scale = jnp.clip(scale, min_scale_factor, max_scale_factor)

    # Compute and clamp absolute step size
    abs_h_next = jnp.clip(abs_h * scale, min_step, max_step)

    return sign_h * abs_h_next
