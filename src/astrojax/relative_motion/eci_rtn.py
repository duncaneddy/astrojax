"""ECI to RTN frame transformations for relative satellite motion.

Provides rotation matrices and state-vector transformations between the
Earth-Centered Inertial (ECI) frame and the Radial, Along-Track,
Cross-Track (RTN) frame attached to a chief satellite.

The RTN frame (also called LVLH) is defined as:

- **R** (Radial): from Earth's center toward the satellite position.
- **T** (Along-track): in the orbital plane, completing the right-handed
  triad (N x R).
- **N** (Cross-track / Normal): along the angular-momentum vector,
  perpendicular to the orbital plane.

All inputs and outputs use SI base units (metres, metres/second).

References:
    1. H. Schaub and J. Junkins, *Analytical Mechanics of Space Systems*,
       2nd ed., AIAA, 2009.
    2. K. Alfriend et al., *Spacecraft Formation Flying*, Elsevier, 2010,
       eq. 2.16.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def rotation_rtn_to_eci(x_eci: ArrayLike) -> Array:
    """Compute the 3x3 rotation matrix from the RTN frame to the ECI frame.

    The columns of the returned matrix are the RTN unit vectors expressed in
    ECI coordinates: ``[r_hat | t_hat | n_hat]``.

    Args:
        x_eci: 6-element ECI state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        jax.Array: 3x3 rotation matrix (RTN -> ECI).

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax.relative_motion import rotation_rtn_to_eci
        >>> from astrojax.constants import R_EARTH, GM_EARTH
        >>> sma = R_EARTH + 500e3
        >>> v_circ = jnp.sqrt(GM_EARTH / sma)
        >>> x = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])
        >>> R = rotation_rtn_to_eci(x)
        >>> R.shape
        (3, 3)
    """
    x_eci = jnp.asarray(x_eci, dtype=jnp.float32)

    r = x_eci[:3]
    v = x_eci[3:6]

    r_norm = jnp.linalg.norm(r)
    h = jnp.cross(r, v)
    h_norm = jnp.linalg.norm(h)

    r_hat = r / r_norm
    n_hat = h / h_norm
    t_hat = jnp.cross(n_hat, r_hat)

    return jnp.column_stack([r_hat, t_hat, n_hat])


def rotation_eci_to_rtn(x_eci: ArrayLike) -> Array:
    """Compute the 3x3 rotation matrix from the ECI frame to the RTN frame.

    This is the transpose of :func:`rotation_rtn_to_eci`.

    Args:
        x_eci: 6-element ECI state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        jax.Array: 3x3 rotation matrix (ECI -> RTN).

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax.relative_motion import rotation_eci_to_rtn
        >>> from astrojax.constants import R_EARTH, GM_EARTH
        >>> sma = R_EARTH + 500e3
        >>> v_circ = jnp.sqrt(GM_EARTH / sma)
        >>> x = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])
        >>> R = rotation_eci_to_rtn(x)
        >>> R.shape
        (3, 3)
    """
    return rotation_rtn_to_eci(x_eci).T


def state_eci_to_rtn(x_chief: ArrayLike, x_deputy: ArrayLike) -> Array:
    """Transform absolute ECI states to a relative RTN state.

    Computes the position and velocity of *x_deputy* relative to
    *x_chief* expressed in the chief's rotating RTN frame, accounting
    for the Coriolis effect due to the frame's angular velocity.

    Args:
        x_chief: 6-element ECI state of the chief satellite
            ``[x, y, z, vx, vy, vz]``. Units: m, m/s.
        x_deputy: 6-element ECI state of the deputy satellite
            ``[x, y, z, vx, vy, vz]``. Units: m, m/s.

    Returns:
        jax.Array: 6-element relative state in RTN
            ``[rho_R, rho_T, rho_N, rho_dot_R, rho_dot_T, rho_dot_N]``.
            Units: m, m/s.

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax.relative_motion import state_eci_to_rtn
        >>> from astrojax.constants import R_EARTH, GM_EARTH
        >>> sma = R_EARTH + 500e3
        >>> v_circ = jnp.sqrt(GM_EARTH / sma)
        >>> chief = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])
        >>> deputy = chief + jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        >>> rel = state_eci_to_rtn(chief, deputy)
        >>> float(rel[0])  # ~100 m radial offset
        100.0
    """
    x_chief = jnp.asarray(x_chief, dtype=jnp.float32)
    x_deputy = jnp.asarray(x_deputy, dtype=jnp.float32)

    rc = x_chief[:3]
    vc = x_chief[3:6]

    R_eci2rtn = rotation_eci_to_rtn(x_chief)

    # Relative position / velocity in ECI
    rho_eci = x_deputy[:3] - rc
    rho_dot_eci = x_deputy[3:6] - vc

    # Angular rate of RTN frame (Alfriend eq. 2.16)
    f_dot = jnp.linalg.norm(jnp.cross(rc, vc)) / jnp.linalg.norm(rc) ** 2
    omega = jnp.array([0.0, 0.0, f_dot])

    # Transform to RTN
    rho_rtn = R_eci2rtn @ rho_eci
    rho_dot_rtn = R_eci2rtn @ rho_dot_eci - jnp.cross(omega, rho_rtn)

    return jnp.concatenate([rho_rtn, rho_dot_rtn])


def state_rtn_to_eci(x_chief: ArrayLike, x_rel_rtn: ArrayLike) -> Array:
    """Transform a relative RTN state back to an absolute ECI state.

    Given the chief's ECI state and the deputy's relative state in the
    chief's RTN frame, reconstruct the deputy's absolute ECI state.

    Args:
        x_chief: 6-element ECI state of the chief satellite
            ``[x, y, z, vx, vy, vz]``. Units: m, m/s.
        x_rel_rtn: 6-element relative state in RTN
            ``[rho_R, rho_T, rho_N, rho_dot_R, rho_dot_T, rho_dot_N]``.
            Units: m, m/s.

    Returns:
        jax.Array: 6-element absolute ECI state of the deputy
            ``[x, y, z, vx, vy, vz]``. Units: m, m/s.

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax.relative_motion import state_rtn_to_eci
        >>> from astrojax.constants import R_EARTH, GM_EARTH
        >>> sma = R_EARTH + 500e3
        >>> v_circ = jnp.sqrt(GM_EARTH / sma)
        >>> chief = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])
        >>> rel_rtn = jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        >>> deputy = state_rtn_to_eci(chief, rel_rtn)
        >>> float(deputy[0] - chief[0])  # ~100 m radial offset
        100.0
    """
    x_chief = jnp.asarray(x_chief, dtype=jnp.float32)
    x_rel_rtn = jnp.asarray(x_rel_rtn, dtype=jnp.float32)

    rc = x_chief[:3]
    vc = x_chief[3:6]

    R_rtn2eci = rotation_rtn_to_eci(x_chief)

    rho_rtn = x_rel_rtn[:3]
    rho_dot_rtn = x_rel_rtn[3:6]

    # Angular rate of RTN frame (Alfriend eq. 2.16)
    f_dot = jnp.linalg.norm(jnp.cross(rc, vc)) / jnp.linalg.norm(rc) ** 2
    omega = jnp.array([0.0, 0.0, f_dot])

    r_deputy = rc + R_rtn2eci @ rho_rtn
    v_deputy = R_rtn2eci @ (rho_dot_rtn + jnp.cross(omega, rho_rtn)) + vc

    return jnp.concatenate([r_deputy, v_deputy])
