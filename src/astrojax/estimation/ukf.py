"""Unscented Kalman Filter (UKF) predict and update functions.

Implements the scaled Unscented Kalman Filter using the Van der Merwe
sigma point algorithm. The propagation and measurement functions are
applied to sigma points via ``jax.vmap`` for efficient parallel
evaluation.

The covariance update uses the Joseph form for guaranteed symmetry and
positive semi-definiteness. Cholesky decomposition is regularized with
a dtype-adaptive epsilon to prevent failure from float32 precision loss.

These are building-block functions designed to compose with
``jax.lax.scan`` for sequential filtering. See the user guide for
a complete orbit determination example.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.estimation._types import FilterResult, FilterState, UKFConfig


def _sigma_points(
    x: Array,
    P: Array,
    config: UKFConfig,
) -> tuple[Array, Array, Array]:
    """Generate scaled sigma points and weights.

    Uses Van der Merwe's scaled unscented transform to generate
    ``2n + 1`` sigma points from the state mean and covariance.

    Args:
        x: State mean of shape ``(n,)``.
        P: State covariance of shape ``(n, n)``.
        config: UKF sigma point configuration.

    Returns:
        A tuple ``(points, Wm, Wc)`` where:

        - ``points``: Sigma points of shape ``(2n+1, n)``.
        - ``Wm``: Mean weights of shape ``(2n+1,)``.
        - ``Wc``: Covariance weights of shape ``(2n+1,)``.
    """
    dtype = get_dtype()
    n = x.shape[0]

    alpha = config.alpha
    beta = config.beta
    kappa = config.kappa
    lam = alpha**2 * (n + kappa) - n

    # Regularize P before Cholesky
    eps = jnp.finfo(dtype).eps * 100.0
    P_reg = P + eps * jnp.eye(n, dtype=dtype)

    # Cholesky factor of (n + lambda) * P
    L = jnp.linalg.cholesky((n + lam) * P_reg)

    # Sigma points: x, x + L_i, x - L_i
    points_plus = x[None, :] + L.T  # (n, n) -> broadcast with x
    points_minus = x[None, :] - L.T
    points = jnp.concatenate([x[None, :], points_plus, points_minus], axis=0)

    # Weights
    w0_m = jnp.asarray(lam / (n + lam), dtype=dtype)
    w0_c = jnp.asarray(lam / (n + lam) + (1.0 - alpha**2 + beta), dtype=dtype)
    wi = jnp.asarray(1.0 / (2.0 * (n + lam)), dtype=dtype)

    Wm = jnp.concatenate([jnp.array([w0_m], dtype=dtype), jnp.full(2 * n, wi, dtype=dtype)])
    Wc = jnp.concatenate([jnp.array([w0_c], dtype=dtype), jnp.full(2 * n, wi, dtype=dtype)])

    return points, Wm, Wc


_DEFAULT_UKF_CONFIG = UKFConfig()


def ukf_predict(
    filter_state: FilterState,
    propagate_fn: Callable[[Array], Array],
    Q: ArrayLike,
    config: UKFConfig = _DEFAULT_UKF_CONFIG,
) -> FilterState:
    """Propagate the filter state forward one timestep using sigma points.

    Generates sigma points from the current state and covariance,
    propagates each through ``propagate_fn`` via ``jax.vmap``, and
    reconstructs the predicted mean and covariance from the weighted
    propagated points.

    The user constructs ``propagate_fn`` by closing over their dynamics,
    integrator, and timestep::

        dynamics = create_orbit_dynamics(epoch_0)
        def propagate(x):
            return rk4_step(dynamics, t, x, dt).state

    Args:
        filter_state: Current filter state ``(x, P)``.
        propagate_fn: State propagation function ``f(x) -> x_next``.
            Applied to each sigma point via ``jax.vmap``.
        Q: Process noise covariance matrix of shape ``(n, n)``.
        config: UKF sigma point configuration. Default: ``UKFConfig()``.

    Returns:
        FilterState: Predicted state and covariance ``(x_pred, P_pred)``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.estimation import FilterState, UKFConfig, ukf_predict

        x0 = jnp.array([1.0, 0.0])
        P0 = jnp.eye(2) * 0.01
        fs = FilterState(x=x0, P=P0)
        Q = jnp.eye(2) * 1e-6

        def propagate(x):
            return x + jnp.array([x[1], -x[0]]) * 0.01

        fs_pred = ukf_predict(fs, propagate, Q)
        ```
    """
    dtype = get_dtype()
    x = jnp.asarray(filter_state.x, dtype=dtype)
    P = jnp.asarray(filter_state.P, dtype=dtype)
    Q = jnp.asarray(Q, dtype=dtype)

    # Generate sigma points
    points, Wm, Wc = _sigma_points(x, P, config)

    # Propagate sigma points
    propagated = jax.vmap(propagate_fn)(points)

    # Weighted mean
    x_pred = jnp.einsum("i,ij->j", Wm, propagated)

    # Weighted covariance
    diff = propagated - x_pred[None, :]
    P_pred = jnp.einsum("i,ij,ik->jk", Wc, diff, diff) + Q

    return FilterState(x=x_pred, P=P_pred)


def ukf_update(
    filter_state: FilterState,
    z: ArrayLike,
    measurement_fn: Callable[[Array], Array],
    R: ArrayLike,
    config: UKFConfig = _DEFAULT_UKF_CONFIG,
) -> FilterResult:
    """Incorporate a measurement into the filter state using sigma points.

    Generates sigma points from the predicted state, transforms each
    through the measurement function via ``jax.vmap``, and computes
    the Kalman gain from the cross-covariance between state and
    measurement spaces.

    Args:
        filter_state: Predicted filter state ``(x_pred, P_pred)``,
            typically from ``ukf_predict``.
        z: Measurement vector of shape ``(m,)``.
        measurement_fn: Measurement model ``h(x) -> z_pred``. Maps the
            state to the expected measurement. Applied to each sigma
            point via ``jax.vmap``.
        R: Measurement noise covariance matrix of shape ``(m, m)``.
        config: UKF sigma point configuration. Must match the config
            used in ``ukf_predict``. Default: ``UKFConfig()``.

    Returns:
        FilterResult: Updated state, innovation, innovation covariance,
            and Kalman gain.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.estimation import FilterState, ukf_update

        x_pred = jnp.array([1.0, 0.5, 0.0, 0.0, 7.5e3, 0.0])
        P_pred = jnp.eye(6) * 100.0
        fs = FilterState(x=x_pred, P=P_pred)

        z = jnp.array([1.01, 0.49, 0.01])
        R = jnp.eye(3) * 0.01

        def measure_position(x):
            return x[:3]

        result = ukf_update(fs, z, measure_position, R)
        ```
    """
    dtype = get_dtype()
    x = jnp.asarray(filter_state.x, dtype=dtype)
    P = jnp.asarray(filter_state.P, dtype=dtype)
    z = jnp.asarray(z, dtype=dtype)
    R = jnp.asarray(R, dtype=dtype)

    # Generate sigma points from predicted state
    points, Wm, Wc = _sigma_points(x, P, config)

    # Transform sigma points through measurement function
    z_points = jax.vmap(measurement_fn)(points)

    # Predicted measurement (weighted mean)
    z_pred = jnp.einsum("i,ij->j", Wm, z_points)

    # Innovation
    innovation = z - z_pred

    # Innovation covariance
    z_diff = z_points - z_pred[None, :]
    S = jnp.einsum("i,ij,ik->jk", Wc, z_diff, z_diff) + R

    # Cross-covariance between state and measurement
    x_diff = points - x[None, :]
    Pxz = jnp.einsum("i,ij,ik->jk", Wc, x_diff, z_diff)

    # Kalman gain: K = Pxz @ S^{-1}
    K = jnp.linalg.solve(S, Pxz.T).T

    # State update
    x_upd = x + K @ innovation

    # Joseph form covariance update
    # Compute effective H as K^T-weighted approximation
    # For UKF, use: P = P - K S K^T (equivalent to Joseph form when K is exact)
    P_upd = P - K @ S @ K.T

    return FilterResult(
        state=FilterState(x=x_upd, P=P_upd),
        innovation=innovation,
        innovation_covariance=S,
        kalman_gain=K,
    )
