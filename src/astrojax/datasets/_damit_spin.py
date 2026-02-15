"""JAX rotation and shape operations for DAMIT spin parameters.

Computes body-fixed-to-ecliptic rotation matrices from DAMIT spin
parameters, and provides utilities for scaling and rotating asteroid
shape vertices.  All functions use JAX primitives and are compatible
with ``jax.jit``, ``jax.vmap``, and ``jax.grad``.

The rotation model for principal-axis rotators is:

.. math::

    R = R_z(\\lambda) \\cdot R_y(90^\\circ - \\beta) \\cdot R_z(\\phi_0 + \\frac{360}{P} (t - t_0) \\cdot 24 + \\frac{1}{2} \\dot{\\omega} (t - t_0)^2)

where :math:`(t - t_0)` is in days, :math:`P` is in hours, and
:math:`\\dot{\\omega}` (YORP) is in deg/day\u00b2.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.attitude_representations.rotation_matrices import Ry, Rz


def damit_spin_to_rotation(
    spin_params: ArrayLike,
    target_jd: ArrayLike,
) -> Array:
    """Compute the body-to-ecliptic rotation matrix from DAMIT spin parameters.

    Implements the principal-axis rotation model used by DAMIT:

    ``R = Rz(lambda) @ Ry(90 - beta) @ Rz(phi0 + 360/P * (t-t0) * 24 + 0.5 * yorp * (t-t0)^2)``

    where time differences are in days, period P is in hours, and YORP
    acceleration is in deg/day^2.

    Args:
        spin_params: 6-element array ``[lambda_deg, beta_deg, P_hours,
            t0_jd, phi0_deg, yorp]``.  Set ``yorp=0.0`` when absent.
        target_jd: Julian date at which to evaluate the rotation.

    Returns:
        3x3 rotation matrix (body-fixed to ecliptic).

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.datasets import damit_spin_to_rotation

        spin = jnp.array([30.0, 60.0, 5.0, 2460000.5, 0.0, 0.0])
        R = damit_spin_to_rotation(spin, 2460001.5)
        print(R.shape)  # (3, 3)
        ```
    """
    spin_params = jnp.asarray(spin_params)
    target_jd = jnp.asarray(target_jd)

    lam_deg = spin_params[0]
    beta_deg = spin_params[1]
    period_hours = spin_params[2]
    t0_jd = spin_params[3]
    phi0_deg = spin_params[4]
    yorp = spin_params[5]

    dt_days = target_jd - t0_jd

    # Rotation angle about body spin axis (degrees)
    spin_angle_deg = phi0_deg + 360.0 / period_hours * dt_days * 24.0 + 0.5 * yorp * dt_days**2

    # Build rotation: body-fixed -> ecliptic
    r1 = Rz(lam_deg, use_degrees=True)
    r2 = Ry(90.0 - beta_deg, use_degrees=True)
    r3 = Rz(spin_angle_deg, use_degrees=True)

    return r1 @ r2 @ r3


def scale_shape_vertices(
    vertices: ArrayLike,
    max_extent_m: ArrayLike,
) -> Array:
    """Rescale shape vertices so the furthest vertex is at a given distance.

    Normalizes the vertex cloud by its current maximum extent (distance
    from origin to furthest vertex), then scales to *max_extent_m*.

    Args:
        vertices: ``(N, 3)`` array of vertex coordinates.
        max_extent_m: Desired maximum extent in meters (distance from
            origin to the furthest vertex after scaling).

    Returns:
        ``(N, 3)`` array of rescaled vertex coordinates.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.datasets import scale_shape_vertices

        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        scaled = scale_shape_vertices(verts, 100.0)
        # Furthest vertex is now at distance 100.0
        ```
    """
    vertices = jnp.asarray(vertices, dtype=jnp.float32)
    max_extent_m = jnp.asarray(max_extent_m, dtype=jnp.float32)

    distances = jnp.linalg.norm(vertices, axis=1)
    current_max = jnp.max(distances)

    scale = max_extent_m / current_max
    return vertices * scale


def rotate_shape_points(
    spin_params: ArrayLike,
    target_jd: ArrayLike,
    vertices: ArrayLike,
) -> Array:
    """Rotate shape vertices from body-fixed to ecliptic frame.

    Computes the rotation matrix via :func:`damit_spin_to_rotation`
    and applies it to all vertex positions: ``(R @ vertices.T).T``.

    Args:
        spin_params: 6-element spin parameter array (see
            :func:`damit_spin_to_rotation`).
        target_jd: Julian date at which to evaluate the rotation.
        vertices: ``(N, 3)`` array of body-fixed vertex coordinates.

    Returns:
        ``(N, 3)`` array of ecliptic-frame vertex coordinates.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.datasets import rotate_shape_points

        spin = jnp.array([30.0, 60.0, 5.0, 2460000.5, 0.0, 0.0])
        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rotated = rotate_shape_points(spin, 2460001.5, verts)
        ```
    """
    vertices = jnp.asarray(vertices, dtype=jnp.float32)
    rot = damit_spin_to_rotation(spin_params, target_jd)
    return (rot @ vertices.T).T
