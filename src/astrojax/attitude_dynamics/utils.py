"""Utility functions for attitude dynamics.

Provides helper functions for maintaining state consistency
during numerical integration of attitude dynamics.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype


def normalize_attitude_state(state: ArrayLike) -> Array:
    """Renormalize the quaternion portion of an attitude state vector.

    Numerical integration causes the quaternion norm to drift from
    unity.  This function normalizes the quaternion ``[w, x, y, z]``
    (elements 0--3) while leaving the angular velocity (elements 4--6)
    unchanged.

    Args:
        state: Attitude state ``[q_w, q_x, q_y, q_z, wx, wy, wz]``
            of shape ``(7,)``.

    Returns:
        State with normalized quaternion, shape ``(7,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        state = jnp.array([1.001, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
        normed = normalize_attitude_state(state)
        float(jnp.linalg.norm(normed[:4]))
        ```
    """
    _float = get_dtype()
    state = jnp.asarray(state, dtype=_float)

    q = state[:4]
    omega = state[4:7]

    q_normed = q / jnp.linalg.norm(q)

    return jnp.concatenate([q_normed, omega])
