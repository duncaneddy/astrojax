"""State estimation filters for orbit determination.

Provides Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF)
building blocks for sequential state estimation. Measurement models are
in the :mod:`astrojax.orbit_measurements` module.

Available components:

- :class:`FilterState` -- Filter state (estimate and covariance)
- :class:`UKFConfig` -- UKF sigma point configuration
- :class:`FilterResult` -- Update result with diagnostics
- :func:`ekf_predict` -- EKF state propagation (autodiff STM)
- :func:`ekf_update` -- EKF measurement update (Joseph form)
- :func:`ukf_predict` -- UKF state propagation (sigma point transform)
- :func:`ukf_update` -- UKF measurement update (sigma point transform)

All functions are compatible with ``jax.jit`` and ``jax.lax.scan`` for
efficient sequential filtering.
"""

from astrojax.estimation._types import FilterResult, FilterState, UKFConfig
from astrojax.estimation.ekf import ekf_predict, ekf_update
from astrojax.estimation.ukf import ukf_predict, ukf_update

__all__ = [
    "FilterState",
    "UKFConfig",
    "FilterResult",
    "ekf_predict",
    "ekf_update",
    "ukf_predict",
    "ukf_update",
]
