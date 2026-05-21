"""Hermite cubic interpolation utilities.

Provides JIT-compatible cubic Hermite interpolation for position/velocity
data, enabling O(dt^4) accuracy between trajectory samples.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def hermite_interp(
    t: ArrayLike,
    t0: ArrayLike,
    t1: ArrayLike,
    p0: ArrayLike,
    p1: ArrayLike,
    v0: ArrayLike,
    v1: ArrayLike,
) -> Array:
    """Cubic Hermite interpolation between two points.

    Given positions and velocities at t0 and t1, interpolates at t.
    JIT-compatible, vmappable.

    Args:
        t: Query time (scalar).
        t0, t1: Bracket times (scalars).
        p0, p1: Positions at t0, t1 (arrays, any shape).
        v0, v1: Velocities at t0, t1 (same shape as positions).

    Returns:
        Interpolated position at t (same shape as p0).
    """
    dt = t1 - t0
    s = (t - t0) / dt  # normalized [0, 1]
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * p0 + h10 * (dt * v0) + h01 * p1 + h11 * (dt * v1)


def make_hermite_position_fn(
    times: ArrayLike,
    positions: ArrayLike,
    velocities: ArrayLike,
) -> callable:
    """Build a JIT-compatible interpolated position function using Hermite cubics.

    Args:
        times: 1-D array of sample times, shape (N,).
        positions: Array of positions, shape (N, 3).
        velocities: Array of velocities, shape (N, 3).

    Returns:
        Callable ``f(t) -> Array[3]``, JIT-compatible.
    """
    times = jnp.asarray(times)
    positions = jnp.asarray(positions)
    velocities = jnp.asarray(velocities)
    n = times.shape[0]

    def position_fn(t: ArrayLike) -> Array:
        t = jnp.asarray(t, dtype=times.dtype)
        # Find the bracket index via searchsorted
        idx = jnp.searchsorted(times, t, side="right") - 1
        idx = jnp.clip(idx, 0, n - 2)

        t0 = times[idx]
        t1 = times[idx + 1]
        p0 = positions[idx]
        p1 = positions[idx + 1]
        v0 = velocities[idx]
        v1 = velocities[idx + 1]

        return hermite_interp(t, t0, t1, p0, p1, v0, v1)

    return position_fn
