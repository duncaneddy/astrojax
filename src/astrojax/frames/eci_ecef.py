"""ECI to ECEF frame transformations using Earth rotation.

Provides rotation matrices and state-vector transformations between the
Earth-Centered Inertial (ECI) frame and the Earth-Centered Earth-Fixed
(ECEF) frame.

This module implements a simplified transformation model using only the
Earth rotation component — a single :math:`R_z(\\theta_{\\text{GMST}})`
rotation.  The full IAU 2006/2000A model adds bias-precession-nutation
(Q) and polar motion (W) corrections, which contribute arcsecond-level
terms that are below the default float32 precision floor used by astrojax.

All inputs and outputs use SI base units (metres, metres/second).

References:
    1. D. Vallado, *Fundamentals of Astrodynamics and Applications*,
       4th ed., Microcosm Press, 2013, Sec. 3.7.
    2. O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods
       and Applications*, Springer, 2012, Sec. 5.2.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.attitude_representations import Rz
from astrojax.config import get_dtype
from astrojax.constants import OMEGA_EARTH
from astrojax.epoch import Epoch


def earth_rotation(epc: Epoch) -> Array:
    """Compute the Earth rotation matrix at the given epoch.

    Returns the 3x3 rotation matrix :math:`R_z(\\theta_{\\text{GMST}})` that
    rotates vectors from the ECI frame into the ECEF frame, considering
    only the Earth's axial rotation.

    Args:
        epc: Epoch at which to evaluate the rotation.

    Returns:
        jax.Array: 3x3 Earth rotation matrix.

    Example:
        >>> from astrojax import Epoch
        >>> from astrojax.frames import earth_rotation
        >>> epc = Epoch(2024, 1, 1)
        >>> R = earth_rotation(epc)
        >>> R.shape
        (3, 3)
    """
    return Rz(epc.gmst())


def rotation_eci_to_ecef(epc: Epoch) -> Array:
    """Compute the 3x3 rotation matrix from the ECI frame to the ECEF frame.

    Uses only the Earth rotation component (no precession-nutation or
    polar motion).  Equivalent to :func:`earth_rotation`.

    Args:
        epc: Epoch at which to evaluate the rotation.

    Returns:
        jax.Array: 3x3 rotation matrix (ECI → ECEF).

    Example:
        >>> from astrojax import Epoch
        >>> from astrojax.frames import rotation_eci_to_ecef
        >>> epc = Epoch(2024, 1, 1)
        >>> R = rotation_eci_to_ecef(epc)
        >>> R.shape
        (3, 3)
    """
    return earth_rotation(epc)


def rotation_ecef_to_eci(epc: Epoch) -> Array:
    """Compute the 3x3 rotation matrix from the ECEF frame to the ECI frame.

    This is the transpose of :func:`rotation_eci_to_ecef`.

    Args:
        epc: Epoch at which to evaluate the rotation.

    Returns:
        jax.Array: 3x3 rotation matrix (ECEF → ECI).

    Example:
        >>> from astrojax import Epoch
        >>> from astrojax.frames import rotation_ecef_to_eci
        >>> epc = Epoch(2024, 1, 1)
        >>> R = rotation_ecef_to_eci(epc)
        >>> R.shape
        (3, 3)
    """
    return earth_rotation(epc).T


def state_eci_to_ecef(epc: Epoch, x_eci: ArrayLike) -> Array:
    """Transform a 6-element state vector from ECI to ECEF.

    Rotates position and velocity, and subtracts the velocity contribution
    from Earth's rotation:

    .. math::

        \\mathbf{r}_{\\text{ECEF}} &= R \\, \\mathbf{r}_{\\text{ECI}} \\\\
        \\mathbf{v}_{\\text{ECEF}} &= R \\, \\mathbf{v}_{\\text{ECI}}
            - \\boldsymbol{\\omega} \\times \\mathbf{r}_{\\text{ECEF}}

    where :math:`R = R_z(\\theta_{\\text{GMST}})` and
    :math:`\\boldsymbol{\\omega} = [0, 0, \\omega_\\oplus]^T`.

    Args:
        epc: Epoch at which to evaluate the transformation.
        x_eci: 6-element ECI state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        jax.Array: 6-element ECEF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax import Epoch
        >>> from astrojax.frames import state_eci_to_ecef
        >>> from astrojax.constants import R_EARTH, GM_EARTH
        >>> epc = Epoch(2024, 1, 1)
        >>> sma = R_EARTH + 500e3
        >>> v_circ = jnp.sqrt(GM_EARTH / sma)
        >>> x_eci = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])
        >>> x_ecef = state_eci_to_ecef(epc, x_eci)
        >>> x_ecef.shape
        (6,)
    """
    x_eci = jnp.asarray(x_eci, dtype=get_dtype())

    R = earth_rotation(epc)
    omega = jnp.array([0.0, 0.0, OMEGA_EARTH], dtype=get_dtype())

    r_ecef = R @ x_eci[:3]
    v_ecef = R @ x_eci[3:6] - jnp.cross(omega, r_ecef)

    return jnp.concatenate([r_ecef, v_ecef])


def state_ecef_to_eci(epc: Epoch, x_ecef: ArrayLike) -> Array:
    """Transform a 6-element state vector from ECEF to ECI.

    Applies the inverse of :func:`state_eci_to_ecef`:

    .. math::

        \\mathbf{r}_{\\text{ECI}} &= R^T \\, \\mathbf{r}_{\\text{ECEF}} \\\\
        \\mathbf{v}_{\\text{ECI}} &= R^T \\left(
            \\mathbf{v}_{\\text{ECEF}}
            + \\boldsymbol{\\omega} \\times \\mathbf{r}_{\\text{ECEF}}
        \\right)

    where :math:`R = R_z(\\theta_{\\text{GMST}})` and
    :math:`\\boldsymbol{\\omega} = [0, 0, \\omega_\\oplus]^T`.

    Args:
        epc: Epoch at which to evaluate the transformation.
        x_ecef: 6-element ECEF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        jax.Array: 6-element ECI state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax import Epoch
        >>> from astrojax.frames import state_ecef_to_eci
        >>> from astrojax.constants import R_EARTH
        >>> epc = Epoch(2024, 1, 1)
        >>> x_ecef = jnp.array([R_EARTH, 0.0, 0.0, 0.0, 0.0, 0.0])
        >>> x_eci = state_ecef_to_eci(epc, x_ecef)
        >>> x_eci.shape
        (6,)
    """
    x_ecef = jnp.asarray(x_ecef, dtype=get_dtype())

    R = earth_rotation(epc)
    omega = jnp.array([0.0, 0.0, OMEGA_EARTH], dtype=get_dtype())

    r_ecef = x_ecef[:3]
    v_ecef = x_ecef[3:6]

    r_eci = R.T @ r_ecef
    v_eci = R.T @ (v_ecef + jnp.cross(omega, r_ecef))

    return jnp.concatenate([r_eci, v_eci])
