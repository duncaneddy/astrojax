"""Shared utility functions for angle and unit conversions.

These helpers wrap the ``use_degrees`` convention used throughout
astrojax, providing JAX-traceable degree/radian conversion via
``jnp.where``.
"""

from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike


def to_radians(angle: ArrayLike, use_degrees: bool) -> Array:
    """Convert angle to radians if ``use_degrees`` is True.

    Args:
        angle (ArrayLike): Angle value.
        use_degrees (bool): If ``True``, treat ``angle`` as degrees and convert.

    Returns:
        Angle in radians.
    """
    return jnp.where(use_degrees, jnp.deg2rad(angle), angle)


def from_radians(angle: ArrayLike, use_degrees: bool) -> Array:
    """Convert angle from radians to degrees if ``use_degrees`` is True.

    Args:
        angle (ArrayLike): Angle in radians.
        use_degrees (bool): If ``True``, convert to degrees.

    Returns:
        Angle in radians or degrees.
    """
    return jnp.where(use_degrees, jnp.rad2deg(angle), angle)
