"""GNSS measurement models for orbit determination.

Provides measurement functions and noise covariance constructors for
GNSS-based (GPS, Galileo, etc.) orbit determination. These are designed
to be passed directly to ``ekf_update`` or ``ukf_update`` as the
``measurement_fn`` and ``R`` arguments.

Measurement functions extract observable quantities from the full state
vector. Noise covariance constructors build the corresponding ``R``
matrix from sensor noise parameters.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype


def gnss_position_measurement(state: ArrayLike) -> Array:
    """Extract position from an orbital state vector.

    Measurement model for a GNSS receiver that provides position-only
    observations. Extracts the first three components of the state
    vector.

    Args:
        state: State vector of shape ``(n,)`` where ``n >= 3``.
            The first three elements are interpreted as position
            ``[x, y, z]``.

    Returns:
        jax.Array: Position vector of shape ``(3,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.orbit_measurements import gnss_position_measurement

        state = jnp.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
        z = gnss_position_measurement(state)  # [6878e3, 0.0, 0.0]
        ```
    """
    dtype = get_dtype()
    state = jnp.asarray(state, dtype=dtype)
    return state[:3]


def gnss_measurement_noise(sigma_pos: float) -> Array:
    """Construct measurement noise covariance for position-only GNSS.

    Builds a diagonal ``(3, 3)`` noise covariance matrix assuming
    independent, identically distributed position errors.

    Args:
        sigma_pos: Position measurement standard deviation in the same
            units as the state vector (typically meters).

    Returns:
        jax.Array: Diagonal noise covariance matrix of shape ``(3, 3)``
            with ``sigma_pos**2`` on the diagonal.

    Examples:
        ```python
        from astrojax.orbit_measurements import gnss_measurement_noise

        R = gnss_measurement_noise(10.0)  # 10 m 1-sigma noise
        # R = [[100, 0, 0], [0, 100, 0], [0, 0, 100]]
        ```
    """
    dtype = get_dtype()
    return jnp.asarray(sigma_pos**2, dtype=dtype) * jnp.eye(3, dtype=dtype)


def gnss_position_velocity_measurement(state: ArrayLike) -> Array:
    """Extract position and velocity from an orbital state vector.

    Measurement model for a GNSS receiver that provides both position
    and velocity observations. Extracts the first six components of
    the state vector.

    Args:
        state: State vector of shape ``(n,)`` where ``n >= 6``.
            The first six elements are interpreted as
            ``[x, y, z, vx, vy, vz]``.

    Returns:
        jax.Array: Position-velocity vector of shape ``(6,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.orbit_measurements import gnss_position_velocity_measurement

        state = jnp.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
        z = gnss_position_velocity_measurement(state)  # full 6-element vector
        ```
    """
    dtype = get_dtype()
    state = jnp.asarray(state, dtype=dtype)
    return state[:6]


def gnss_position_velocity_noise(
    sigma_pos: float,
    sigma_vel: float,
) -> Array:
    """Construct measurement noise covariance for position-velocity GNSS.

    Builds a diagonal ``(6, 6)`` noise covariance matrix assuming
    independent position and velocity errors.

    Args:
        sigma_pos: Position measurement standard deviation (typically
            meters).
        sigma_vel: Velocity measurement standard deviation (typically
            m/s).

    Returns:
        jax.Array: Diagonal noise covariance matrix of shape ``(6, 6)``
            with ``sigma_pos**2`` for the first three and
            ``sigma_vel**2`` for the last three diagonal elements.

    Examples:
        ```python
        from astrojax.orbit_measurements import gnss_position_velocity_noise

        R = gnss_position_velocity_noise(10.0, 0.1)  # 10 m, 0.1 m/s
        ```
    """
    dtype = get_dtype()
    variances = jnp.array(
        [sigma_pos**2] * 3 + [sigma_vel**2] * 3,
        dtype=dtype,
    )
    return jnp.diag(variances)
