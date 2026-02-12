"""TEME frame transformations for SGP4 output.

TEME (True Equator, Mean Equinox) is the native output frame of the SGP4/SDP4
propagator.  This module provides transformations between TEME and other
commonly used reference frames:

- **TEME -> PEF**: Rotate by GMST (removes mean Earth rotation)
- **PEF -> ITRF**: Apply polar motion
- **ITRF -> GCRF**: Full IAU 2006/2000A CIO-based transformation

All inputs and outputs use SI base units (metres, metres/second).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.attitude_representations import Rz
from astrojax.config import get_dtype
from astrojax.constants import OMEGA_EARTH
from astrojax.eop._types import EOPData
from astrojax.epoch import Epoch
from astrojax.frames.gcrf_itrf import (
    polar_motion,
    state_gcrf_to_itrf,
    state_itrf_to_gcrf,
)

# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------


def rotation_teme_to_pef(epc: Epoch) -> Array:
    """Compute the 3x3 rotation matrix from TEME to PEF.

    Applies ``Rz(GMST)`` to rotate from the mean equinox frame to the
    pseudo Earth-fixed frame.

    Args:
        epc: Epoch (UTC).

    Returns:
        3x3 rotation matrix (TEME -> PEF).
    """
    return Rz(epc.gmst())


def rotation_pef_to_teme(epc: Epoch) -> Array:
    """Compute the 3x3 rotation matrix from PEF to TEME.

    Transpose of :func:`rotation_teme_to_pef`.

    Args:
        epc: Epoch (UTC).

    Returns:
        3x3 rotation matrix (PEF -> TEME).
    """
    return rotation_teme_to_pef(epc).T


def rotation_teme_to_itrf(eop: EOPData, epc: Epoch) -> Array:
    """Compute the 3x3 rotation matrix from TEME to ITRF.

    Combines GMST rotation and polar motion:
    ``R = PM @ Rz(GMST)``

    Args:
        eop: EOP data providing polar motion parameters.
        epc: Epoch (UTC).

    Returns:
        3x3 rotation matrix (TEME -> ITRF).
    """
    pm = polar_motion(eop, epc)
    r_gmst = Rz(epc.gmst())
    return pm @ r_gmst


def rotation_itrf_to_teme(eop: EOPData, epc: Epoch) -> Array:
    """Compute the 3x3 rotation matrix from ITRF to TEME.

    Transpose of :func:`rotation_teme_to_itrf`.

    Args:
        eop: EOP data.
        epc: Epoch (UTC).

    Returns:
        3x3 rotation matrix (ITRF -> TEME).
    """
    return rotation_teme_to_itrf(eop, epc).T


# ---------------------------------------------------------------------------
# State vector transformations
# ---------------------------------------------------------------------------


def state_teme_to_pef(epc: Epoch, x_teme: ArrayLike) -> Array:
    """Transform a 6-element state vector from TEME to PEF.

    Velocity includes the Earth-rotation correction:
    ``v_pef = R @ v_teme - omega_earth x r_pef``

    Args:
        epc: Epoch (UTC).
        x_teme: 6-element TEME state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        6-element PEF state ``[x, y, z, vx, vy, vz]``. Units: m, m/s.
    """
    dtype = get_dtype()
    x_teme = jnp.asarray(x_teme, dtype=dtype)

    R = rotation_teme_to_pef(epc)
    omega = jnp.array([0.0, 0.0, OMEGA_EARTH], dtype=dtype)

    r_teme = x_teme[:3]
    v_teme = x_teme[3:6]

    r_pef = R @ r_teme
    v_pef = R @ v_teme - jnp.cross(omega, r_pef)

    return jnp.concatenate([r_pef, v_pef])


def state_pef_to_teme(epc: Epoch, x_pef: ArrayLike) -> Array:
    """Transform a 6-element state vector from PEF to TEME.

    Inverse of :func:`state_teme_to_pef`.

    Args:
        epc: Epoch (UTC).
        x_pef: 6-element PEF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        6-element TEME state ``[x, y, z, vx, vy, vz]``. Units: m, m/s.
    """
    dtype = get_dtype()
    x_pef = jnp.asarray(x_pef, dtype=dtype)

    R = rotation_teme_to_pef(epc)
    omega = jnp.array([0.0, 0.0, OMEGA_EARTH], dtype=dtype)

    r_pef = x_pef[:3]
    v_pef = x_pef[3:6]

    r_teme = R.T @ r_pef
    v_teme = R.T @ (v_pef + jnp.cross(omega, r_pef))

    return jnp.concatenate([r_teme, v_teme])


def state_teme_to_itrf(eop: EOPData, epc: Epoch, x_teme: ArrayLike) -> Array:
    """Transform a 6-element state vector from TEME to ITRF.

    Goes via PEF: applies GMST rotation and omega correction, then
    applies polar motion (position only, PM has negligible velocity effect).

    Args:
        eop: EOP data providing polar motion parameters.
        epc: Epoch (UTC).
        x_teme: 6-element TEME state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        6-element ITRF state ``[x, y, z, vx, vy, vz]``. Units: m, m/s.
    """
    dtype = get_dtype()
    x_teme = jnp.asarray(x_teme, dtype=dtype)

    R_gmst = rotation_teme_to_pef(epc)
    pm = polar_motion(eop, epc)
    omega = jnp.array([0.0, 0.0, OMEGA_EARTH], dtype=dtype)

    r_teme = x_teme[:3]
    v_teme = x_teme[3:6]

    # TEME -> PEF
    r_pef = R_gmst @ r_teme
    v_pef = R_gmst @ v_teme - jnp.cross(omega, r_pef)

    # PEF -> ITRF (polar motion)
    r_itrf = pm @ r_pef
    v_itrf = pm @ v_pef

    return jnp.concatenate([r_itrf, v_itrf])


def state_itrf_to_teme(eop: EOPData, epc: Epoch, x_itrf: ArrayLike) -> Array:
    """Transform a 6-element state vector from ITRF to TEME.

    Inverse of :func:`state_teme_to_itrf`.

    Args:
        eop: EOP data.
        epc: Epoch (UTC).
        x_itrf: 6-element ITRF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        6-element TEME state ``[x, y, z, vx, vy, vz]``. Units: m, m/s.
    """
    dtype = get_dtype()
    x_itrf = jnp.asarray(x_itrf, dtype=dtype)

    R_gmst = rotation_teme_to_pef(epc)
    pm = polar_motion(eop, epc)
    omega = jnp.array([0.0, 0.0, OMEGA_EARTH], dtype=dtype)

    r_itrf = x_itrf[:3]
    v_itrf = x_itrf[3:6]

    # ITRF -> PEF (inverse polar motion)
    r_pef = pm.T @ r_itrf
    v_pef = pm.T @ v_itrf

    # PEF -> TEME (inverse GMST rotation + omega correction)
    r_teme = R_gmst.T @ r_pef
    v_teme = R_gmst.T @ (v_pef + jnp.cross(omega, r_pef))

    return jnp.concatenate([r_teme, v_teme])


def state_teme_to_gcrf(eop: EOPData, epc: Epoch, x_teme: ArrayLike) -> Array:
    """Transform a 6-element state vector from TEME to GCRF.

    Chains TEME -> ITRF -> GCRF using the existing ITRF-GCRF
    transformation.

    Args:
        eop: EOP data.
        epc: Epoch (UTC).
        x_teme: 6-element TEME state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        6-element GCRF state ``[x, y, z, vx, vy, vz]``. Units: m, m/s.
    """
    x_itrf = state_teme_to_itrf(eop, epc, x_teme)
    return state_itrf_to_gcrf(eop, epc, x_itrf)


def state_gcrf_to_teme(eop: EOPData, epc: Epoch, x_gcrf: ArrayLike) -> Array:
    """Transform a 6-element state vector from GCRF to TEME.

    Chains GCRF -> ITRF -> TEME.

    Args:
        eop: EOP data.
        epc: Epoch (UTC).
        x_gcrf: 6-element GCRF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        6-element TEME state ``[x, y, z, vx, vy, vz]``. Units: m, m/s.
    """
    x_itrf = state_gcrf_to_itrf(eop, epc, x_gcrf)
    return state_itrf_to_teme(eop, epc, x_itrf)
