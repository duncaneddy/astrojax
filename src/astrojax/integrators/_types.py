"""Type definitions for numerical integrators.

Provides the core data types used across all integrator implementations:

- :class:`StepResult`: Output of every step function, containing the new state,
  actual timestep used, error estimate, and suggested next timestep.
- :class:`AdaptiveConfig`: Configuration for adaptive step-size control in
  RKF45 and DP54 integrators.

Both types are :class:`~typing.NamedTuple` instances, which JAX treats as
pytrees automatically. This means they work seamlessly with ``jax.jit``,
``jax.vmap``, and ``jax.lax`` control flow primitives.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class StepResult(NamedTuple):
    """Result of a single integrator step.

    Returned by all step functions (``rk4_step``, ``rkf45_step``, ``dp54_step``).
    For fixed-step methods (RK4), ``error_estimate`` is always 0.0 and
    ``dt_next`` equals ``dt_used``.

    Attributes:
        state: State vector at time ``t + dt_used``.
        dt_used: Actual timestep taken. For adaptive methods, this may be
            smaller than the requested ``dt`` if the step was rejected and
            retried.
        error_estimate: Normalized error estimate. A value <= 1.0 means the
            step met the tolerance. Always 0.0 for RK4.
        dt_next: Suggested timestep for the next step. For adaptive methods,
            this is computed from the error estimate. For RK4, equals
            ``dt_used``.
    """

    state: Array
    dt_used: Array
    error_estimate: Array
    dt_next: Array


class AdaptiveConfig(NamedTuple):
    """Configuration for adaptive step-size control.

    Used by ``rkf45_step`` and ``dp54_step`` to control step acceptance,
    rejection, and step-size adjustment. Default values provide a reasonable
    starting point for orbital mechanics problems.

    Attributes:
        abs_tol: Absolute error tolerance per component. Components with
            magnitude near zero are controlled by this tolerance.
        rel_tol: Relative error tolerance per component. Components with
            large magnitude are controlled by this tolerance.
        safety_factor: Multiplicative safety factor applied to step-size
            predictions. Values < 1.0 produce conservative step sizes.
        min_scale_factor: Minimum allowed ratio ``dt_next / dt_used``.
            Prevents excessively aggressive step-size reduction.
        max_scale_factor: Maximum allowed ratio ``dt_next / dt_used``.
            Prevents excessively aggressive step-size growth.
        min_step: Absolute minimum allowed step size. If the adaptive
            algorithm would reduce below this, the step is accepted
            regardless of error.
        max_step: Absolute maximum allowed step size.
        max_step_attempts: Maximum number of step-rejection retries before
            accepting the step regardless. Prevents infinite loops.
    """

    abs_tol: float = 1e-6
    rel_tol: float = 1e-3
    safety_factor: float = 0.9
    min_scale_factor: float = 0.2
    max_scale_factor: float = 10.0
    min_step: float = 1e-12
    max_step: float = 900.0
    max_step_attempts: int = 10
