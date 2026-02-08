"""Numerical ODE integrators for orbital dynamics propagation.

Provides fixed-step and adaptive Runge-Kutta integrators, all implemented
in JAX for compatibility with ``jax.jit``, ``jax.vmap``, and automatic
differentiation.

Available integrators:

- :func:`rk4_step` -- Classic 4th-order Runge-Kutta (fixed step)
- :func:`rkf45_step` -- Runge-Kutta-Fehlberg 4(5) (adaptive step)
- :func:`dp54_step` -- Dormand-Prince 5(4) (adaptive step)

All step functions share a common interface::

    result = step_fn(dynamics, t, state, dt)

where ``dynamics(t, x) -> dx`` defines the ODE right-hand side, and the
result is a :class:`StepResult` named tuple.
"""

from astrojax.integrators._types import AdaptiveConfig, StepResult
from astrojax.integrators.dp54 import dp54_step
from astrojax.integrators.rk4 import rk4_step
from astrojax.integrators.rkf45 import rkf45_step

__all__ = [
    "AdaptiveConfig",
    "StepResult",
    "rk4_step",
    "rkf45_step",
    "dp54_step",
]
