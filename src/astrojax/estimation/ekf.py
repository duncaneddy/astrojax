"""Extended Kalman Filter (EKF) predict and update functions.

Implements the standard EKF using automatic differentiation to compute
the state transition matrix (STM) and measurement Jacobian. The user
provides a ``propagate_fn(x) -> x_next`` for prediction and a
``measurement_fn(x) -> z_pred`` for update; JAX computes the required
Jacobians via ``jax.jacfwd``.

The covariance update uses the Joseph form for guaranteed symmetry and
positive semi-definiteness, which is important for float32 stability.

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
from astrojax.estimation._types import FilterResult, FilterState


def ekf_predict(
    filter_state: FilterState,
    propagate_fn: Callable[[Array], Array],
    Q: ArrayLike,
) -> FilterState:
    """Propagate the filter state forward one timestep.

    Advances the state estimate through the nonlinear propagation function
    and updates the covariance using the automatically computed state
    transition matrix (STM).

    The user constructs ``propagate_fn`` by closing over their dynamics,
    integrator, and timestep::

        dynamics = create_orbit_dynamics(epoch_0)
        def propagate(x):
            return rk4_step(dynamics, t, x, dt).state

    Args:
        filter_state: Current filter state ``(x, P)``.
        propagate_fn: State propagation function ``f(x) -> x_next``.
            Must be differentiable by JAX (composed of JAX operations).
        Q: Process noise covariance matrix of shape ``(n, n)``.

    Returns:
        FilterState: Predicted state and covariance ``(x_pred, P_pred)``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.estimation import FilterState, ekf_predict

        x0 = jnp.array([1.0, 0.0])
        P0 = jnp.eye(2) * 0.01
        fs = FilterState(x=x0, P=P0)
        Q = jnp.eye(2) * 1e-6

        def propagate(x):
            return x + jnp.array([x[1], -x[0]]) * 0.01

        fs_pred = ekf_predict(fs, propagate, Q)
        ```
    """
    dtype = get_dtype()
    x = jnp.asarray(filter_state.x, dtype=dtype)
    P = jnp.asarray(filter_state.P, dtype=dtype)
    Q = jnp.asarray(Q, dtype=dtype)

    # Propagate state
    x_pred = propagate_fn(x)

    # State transition matrix via autodiff
    Phi = jax.jacfwd(propagate_fn)(x)

    # Propagate covariance
    P_pred = Phi @ P @ Phi.T + Q

    return FilterState(x=x_pred, P=P_pred)


def ekf_update(
    filter_state: FilterState,
    z: ArrayLike,
    measurement_fn: Callable[[Array], Array],
    R: ArrayLike,
) -> FilterResult:
    """Incorporate a measurement into the filter state.

    Computes the Kalman gain, updates the state estimate, and updates
    the covariance using the Joseph form for numerical stability.

    Args:
        filter_state: Predicted filter state ``(x_pred, P_pred)``,
            typically from ``ekf_predict``.
        z: Measurement vector of shape ``(m,)``.
        measurement_fn: Measurement model ``h(x) -> z_pred``. Maps the
            state to the expected measurement. Must be differentiable
            by JAX.
        R: Measurement noise covariance matrix of shape ``(m, m)``.

    Returns:
        FilterResult: Updated state, innovation, innovation covariance,
            and Kalman gain.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.estimation import FilterState, ekf_update

        x_pred = jnp.array([1.0, 0.5, 0.0, 0.0, 7.5e3, 0.0])
        P_pred = jnp.eye(6) * 100.0
        fs = FilterState(x=x_pred, P=P_pred)

        z = jnp.array([1.01, 0.49, 0.01])  # measured position
        R = jnp.eye(3) * 0.01

        def measure_position(x):
            return x[:3]

        result = ekf_update(fs, z, measure_position, R)
        updated_state = result.state
        ```
    """
    dtype = get_dtype()
    x = jnp.asarray(filter_state.x, dtype=dtype)
    P = jnp.asarray(filter_state.P, dtype=dtype)
    z = jnp.asarray(z, dtype=dtype)
    R = jnp.asarray(R, dtype=dtype)

    n = x.shape[0]

    # Predicted measurement and Jacobian
    z_pred = measurement_fn(x)
    H = jax.jacfwd(measurement_fn)(x)

    # Innovation
    innovation = z - z_pred

    # Innovation covariance
    S = H @ P @ H.T + R

    # Kalman gain: K = P H^T S^{-1}
    # Computed as K^T = S^{-1} (H P^T) = S^{-1} (H P) since P is symmetric
    K = jnp.linalg.solve(S, H @ P).T

    # State update
    x_upd = x + K @ innovation

    # Joseph form covariance update: P = (I-KH) P (I-KH)^T + K R K^T
    IKH = jnp.eye(n, dtype=dtype) - K @ H
    P_upd = IKH @ P @ IKH.T + K @ R @ K.T

    return FilterResult(
        state=FilterState(x=x_upd, P=P_upd),
        innovation=innovation,
        innovation_covariance=S,
        kalman_gain=K,
    )
