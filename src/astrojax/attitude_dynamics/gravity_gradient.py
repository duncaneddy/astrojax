"""Gravity gradient torque model.

The gravity gradient torque arises from the differential gravitational
acceleration across an extended rigid body.  It tends to align the
axis of minimum inertia with the nadir direction.

.. math::

    \\tau_{gg} = \\frac{3\\mu}{r^3} \\left( \\hat{r}_b \\times
    (I \\, \\hat{r}_b) \\right)

where :math:`\\hat{r}_b` is the unit position vector expressed in the
body frame.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.attitude_representations.conversions import (
    quaternion_to_rotation_matrix,
)
from astrojax.config import get_dtype
from astrojax.constants import GM_EARTH


def torque_gravity_gradient(
    q: ArrayLike,
    r_eci: ArrayLike,
    I: ArrayLike,  # noqa: E741
    mu: float = GM_EARTH,
) -> Array:
    """Compute the gravity gradient torque in the body frame.

    Args:
        q: Unit quaternion ``[w, x, y, z]`` of shape ``(4,)``
            representing the body-to-inertial rotation.
        r_eci: Spacecraft position in ECI of shape ``(3,)`` [m].
        I: Inertia tensor of shape ``(3, 3)`` [kg m^2].
        mu: Gravitational parameter [m^3/s^2].

    Returns:
        Gravity gradient torque in the body frame
            of shape ``(3,)`` [N m].

    Examples:
        ```python
        import jax.numpy as jnp
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        r_eci = jnp.array([7000e3, 0.0, 0.0])
        I = jnp.diag(jnp.array([10.0, 20.0, 30.0]))
        tau = torque_gravity_gradient(q, r_eci, I)
        tau.shape
        ```
    """
    _float = get_dtype()
    q = jnp.asarray(q, dtype=_float)
    r_eci = jnp.asarray(r_eci, dtype=_float)
    I = jnp.asarray(I, dtype=_float)  # noqa: E741

    # Body-to-inertial rotation matrix from the quaternion
    R_body_to_eci = quaternion_to_rotation_matrix(q)

    # Inertial-to-body rotation (transpose of body-to-inertial)
    R_eci_to_body = R_body_to_eci.T

    # Position unit vector in body frame
    r_norm = jnp.linalg.norm(r_eci)
    r_hat_eci = r_eci / r_norm
    r_hat_body = R_eci_to_body @ r_hat_eci

    # Gravity gradient torque
    tau = (3.0 * mu / r_norm**3) * jnp.cross(r_hat_body, I @ r_hat_body)

    return tau
