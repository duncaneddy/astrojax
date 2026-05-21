"""Type definitions for state estimation filters.

Provides the core data types used across all estimation implementations:

- :class:`FilterState`: Current filter state containing the state estimate
  and covariance matrix.
- :class:`UKFConfig`: Configuration for the Unscented Kalman Filter sigma
  point generation.
- :class:`FilterResult`: Output of a filter update step, containing the
  updated state plus diagnostic information for filter tuning.

All types are :class:`~typing.NamedTuple` instances, which JAX treats as
pytrees automatically. This means they work seamlessly with ``jax.jit``,
``jax.vmap``, and ``jax.lax`` control flow primitives.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class FilterState(NamedTuple):
    """State of a Kalman filter.

    Holds the current state estimate and error covariance matrix. Returned
    by ``ekf_predict``, ``ukf_predict``, and available as the ``state``
    field of :class:`FilterResult`.

    Attributes:
        x: State estimate vector of shape ``(n,)``.
        P: Error covariance matrix of shape ``(n, n)``. Must be symmetric
            positive semi-definite.
    """

    x: Array
    P: Array


class UKFConfig(NamedTuple):
    """Configuration for the Unscented Kalman Filter.

    Controls the sigma point spread and weighting using the scaled unscented
    transform (Van der Merwe). Default values ``alpha=1.0``, ``beta=2.0``,
    ``kappa=0.0`` produce unit-spread sigma points with well-conditioned
    weights, robust for float32 across all state dimensions.

    Attributes:
        alpha: Spread of sigma points around the mean. ``alpha=1.0``
            gives unit spread and avoids extreme weights that arise
            with small alpha in float32. Default: 1.0.
        beta: Prior knowledge of the state distribution. ``beta=2.0``
            is optimal for Gaussian distributions. Default: 2.0.
        kappa: Secondary scaling parameter. ``kappa=0.0`` is standard
            for state estimation. Default: 0.0.
    """

    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0


class FilterResult(NamedTuple):
    """Result of a filter measurement update step.

    Returned by ``ekf_update`` and ``ukf_update``. Contains the updated
    filter state along with diagnostic quantities useful for filter
    tuning and health monitoring.

    Attributes:
        state: Updated :class:`FilterState` after incorporating the
            measurement.
        innovation: Measurement residual ``z - z_pred`` of shape ``(m,)``.
            Should be zero-mean and consistent with ``innovation_covariance``
            for a healthy filter.
        innovation_covariance: Innovation covariance ``S`` of shape
            ``(m, m)``. The normalized innovation squared
            ``innovation.T @ S^{-1} @ innovation`` should follow a
            chi-squared distribution with ``m`` degrees of freedom.
        kalman_gain: Kalman gain matrix ``K`` of shape ``(n, m)``.
    """

    state: FilterState
    innovation: Array
    innovation_covariance: Array
    kalman_gain: Array
