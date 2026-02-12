"""GCRF-ITRF frame transformations using the IAU 2006/2000A CIO-based model.

Provides high-fidelity rotation matrices and state-vector transformations
between the Geocentric Celestial Reference Frame (GCRF) and the International
Terrestrial Reference Frame (ITRF), implementing:

- **Bias-precession-nutation** (GCRF -> CIRS) via IAU 2006/2000A
- **Earth rotation** (CIRS -> TIRS) via IAU 2000 Earth Rotation Angle
- **Polar motion** (TIRS -> ITRF)

All functions take an explicit ``eop: EOPData`` parameter (no global state).

ECI/ECEF aliases are provided for backward compatibility:
:func:`rotation_eci_to_ecef` == :func:`rotation_gcrf_to_itrf`, etc.

All inputs and outputs use SI base units (metres, metres/second).

Uses routines and computations derived from software provided by SOFA
under license. Does not itself constitute software provided by and/or
endorsed by SOFA.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.attitude_representations import Rz
from astrojax.config import get_dtype
from astrojax.constants import OMEGA_EARTH
from astrojax.eop._lookup import get_dxdy, get_pm, get_ut1_utc
from astrojax.eop._types import EOPData
from astrojax.epoch import Epoch
from astrojax.sofa import (
    MJD_ZERO,
    c2ixys,
    era00,
    pom00,
    sp00,
    xys06a,
)
from astrojax.time import TT_TAI, leap_seconds_tai_utc


def _mjd_tt(eop: EOPData, epc: Epoch) -> Array:
    """Compute MJD in TT time system from a UTC epoch.

    TT = UTC + (TAI-UTC) + TT_TAI, where TAI-UTC is the leap second count.

    Args:
        eop: EOP data (unused, but kept for API consistency).
        epc: Epoch (UTC).

    Returns:
        MJD in TT.
    """
    mjd_utc = epc.mjd()
    tai_utc = leap_seconds_tai_utc(mjd_utc)
    return mjd_utc + (tai_utc + TT_TAI) / 86400.0


def _mjd_ut1(eop: EOPData, epc: Epoch) -> Array:
    """Compute MJD in UT1 time system from a UTC epoch.

    UT1 = UTC + (UT1-UTC), where UT1-UTC comes from EOP data.

    Args:
        eop: EOP data providing UT1-UTC offset.
        epc: Epoch (UTC).

    Returns:
        MJD in UT1.
    """
    mjd_utc = epc.mjd()
    ut1_utc = get_ut1_utc(eop, mjd_utc)
    return mjd_utc + ut1_utc / 86400.0


# ---------------------------------------------------------------------------
# Core rotation components
# ---------------------------------------------------------------------------


def bias_precession_nutation(eop: EOPData, epc: Epoch) -> Array:
    """Compute the bias-precession-nutation matrix (GCRF -> CIRS).

    Uses the IAU 2006/2000A CIO-based model with dX/dY corrections from
    EOP data applied to the CIP coordinates.

    Args:
        eop: EOP data providing dX, dY celestial pole offsets.
        epc: Epoch (UTC).

    Returns:
        3x3 rotation matrix (GCRF -> CIRS).
    """
    mjd_tt = _mjd_tt(eop, epc)

    # CIP X, Y and CIO locator s from IAU 2006/2000A model
    x, y, s = xys06a(MJD_ZERO, mjd_tt)

    # Apply celestial pole offsets from EOP data (at UTC)
    mjd_utc = epc.mjd()
    dx, dy = get_dxdy(eop, mjd_utc)
    x = x + dx
    y = y + dy

    # Form celestial-to-intermediate matrix
    return c2ixys(x, y, s)


def earth_rotation_angle(eop: EOPData, epc: Epoch) -> Array:
    """Compute the Earth rotation matrix (CIRS -> TIRS).

    Uses the IAU 2000 Earth Rotation Angle with UT1 time from EOP data.

    Args:
        eop: EOP data providing UT1-UTC offset.
        epc: Epoch (UTC).

    Returns:
        3x3 Earth rotation matrix.
    """
    mjd_ut1 = _mjd_ut1(eop, epc)
    era = era00(MJD_ZERO, mjd_ut1)
    return Rz(era)


def polar_motion(eop: EOPData, epc: Epoch) -> Array:
    """Compute the polar motion matrix (TIRS -> ITRF).

    Uses polar motion parameters from EOP data and the TIO locator s'.

    Args:
        eop: EOP data providing pm_x, pm_y polar motion parameters.
        epc: Epoch (UTC).

    Returns:
        3x3 polar motion matrix.
    """
    mjd_tt = _mjd_tt(eop, epc)
    mjd_utc = epc.mjd()

    # Polar motion parameters from EOP (queried at UTC)
    pm_x, pm_y = get_pm(eop, mjd_utc)

    # TIO locator s' (function of TT)
    sp_val = sp00(MJD_ZERO, mjd_tt)

    return pom00(pm_x, pm_y, sp_val)


# ---------------------------------------------------------------------------
# Combined transformations (GCRF/ITRF naming)
# ---------------------------------------------------------------------------


def rotation_gcrf_to_itrf(eop: EOPData, epc: Epoch) -> Array:
    """Compute the full 3x3 rotation matrix from GCRF to ITRF.

    Combines polar motion, Earth rotation, and bias-precession-nutation:
    ``R = PM @ ER @ BPN``

    Args:
        eop: EOP data.
        epc: Epoch (UTC).

    Returns:
        3x3 rotation matrix (GCRF -> ITRF).

    Examples:
        ```python
        from astrojax import Epoch
        from astrojax.eop import zero_eop
        from astrojax.frames import rotation_gcrf_to_itrf
        eop = zero_eop()
        epc = Epoch(2024, 1, 1)
        R = rotation_gcrf_to_itrf(eop, epc)
        R.shape
        ```
    """
    bpn = bias_precession_nutation(eop, epc)
    er = earth_rotation_angle(eop, epc)
    pm = polar_motion(eop, epc)
    return pm @ er @ bpn


def rotation_itrf_to_gcrf(eop: EOPData, epc: Epoch) -> Array:
    """Compute the full 3x3 rotation matrix from ITRF to GCRF.

    This is the transpose of :func:`rotation_gcrf_to_itrf`.

    Args:
        eop: EOP data.
        epc: Epoch (UTC).

    Returns:
        3x3 rotation matrix (ITRF -> GCRF).
    """
    return rotation_gcrf_to_itrf(eop, epc).T


def state_gcrf_to_itrf(eop: EOPData, epc: Epoch, x_gcrf: ArrayLike) -> Array:
    """Transform a 6-element state vector from GCRF to ITRF.

    Position is rotated directly. Velocity includes the correction for
    Earth's rotation (Coriolis effect):

    .. math::

        \\mathbf{r}_{\\text{ITRF}} &= PM \\cdot ER \\cdot BPN \\cdot
            \\mathbf{r}_{\\text{GCRF}} \\\\
        \\mathbf{v}_{\\text{ITRF}} &= PM \\cdot \\left(
            ER \\cdot BPN \\cdot \\mathbf{v}_{\\text{GCRF}}
            - \\boldsymbol{\\omega} \\times (ER \\cdot BPN \\cdot
            \\mathbf{r}_{\\text{GCRF}}) \\right)

    Args:
        eop: EOP data.
        epc: Epoch (UTC).
        x_gcrf: 6-element GCRF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        6-element ITRF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.
    """
    dtype = get_dtype()
    x_gcrf = jnp.asarray(x_gcrf, dtype=dtype)

    bpn = bias_precession_nutation(eop, epc)
    er = earth_rotation_angle(eop, epc)
    pm = polar_motion(eop, epc)

    omega = jnp.array([0.0, 0.0, OMEGA_EARTH], dtype=dtype)

    r_gcrf = x_gcrf[:3]
    v_gcrf = x_gcrf[3:6]

    # Position: PM @ ER @ BPN @ r
    r_cirs = bpn @ r_gcrf
    r_tirs = er @ r_cirs
    r_itrf = pm @ r_tirs

    # Velocity: PM @ (ER @ BPN @ v - omega x (ER @ BPN @ r))
    v_cirs = bpn @ v_gcrf
    v_tirs = er @ v_cirs - jnp.cross(omega, r_tirs)
    v_itrf = pm @ v_tirs

    return jnp.concatenate([r_itrf, v_itrf])


def state_itrf_to_gcrf(eop: EOPData, epc: Epoch, x_itrf: ArrayLike) -> Array:
    """Transform a 6-element state vector from ITRF to GCRF.

    Applies the inverse of :func:`state_gcrf_to_itrf`.

    Args:
        eop: EOP data.
        epc: Epoch (UTC).
        x_itrf: 6-element ITRF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        6-element GCRF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.
    """
    dtype = get_dtype()
    x_itrf = jnp.asarray(x_itrf, dtype=dtype)

    bpn = bias_precession_nutation(eop, epc)
    er = earth_rotation_angle(eop, epc)
    pm = polar_motion(eop, epc)

    omega = jnp.array([0.0, 0.0, OMEGA_EARTH], dtype=dtype)

    r_itrf = x_itrf[:3]
    v_itrf = x_itrf[3:6]

    # Inverse: r_tirs = PM^T @ r_itrf, etc.
    r_tirs = pm.T @ r_itrf
    v_tirs = pm.T @ v_itrf

    r_cirs = er.T @ r_tirs
    v_cirs = er.T @ (v_tirs + jnp.cross(omega, r_tirs))

    r_gcrf = bpn.T @ r_cirs
    v_gcrf = bpn.T @ v_cirs

    return jnp.concatenate([r_gcrf, v_gcrf])


# ---------------------------------------------------------------------------
# ECI/ECEF aliases (backward compatibility)
# ---------------------------------------------------------------------------


def earth_rotation(eop: EOPData, epc: Epoch) -> Array:
    """Compute the Earth rotation matrix at the given epoch.

    This is the full IAU 2006/2000A GCRF -> ITRF rotation combining
    bias-precession-nutation, Earth rotation angle, and polar motion.
    Equivalent to :func:`rotation_gcrf_to_itrf`.

    Args:
        eop: EOP data.
        epc: Epoch (UTC).

    Returns:
        3x3 Earth rotation matrix (GCRF -> ITRF).
    """
    return rotation_gcrf_to_itrf(eop, epc)


def rotation_eci_to_ecef(eop: EOPData, epc: Epoch) -> Array:
    """Compute the 3x3 rotation matrix from ECI (GCRF) to ECEF (ITRF).

    Alias for :func:`rotation_gcrf_to_itrf`.

    Args:
        eop: EOP data.
        epc: Epoch (UTC).

    Returns:
        3x3 rotation matrix (ECI -> ECEF).
    """
    return rotation_gcrf_to_itrf(eop, epc)


def rotation_ecef_to_eci(eop: EOPData, epc: Epoch) -> Array:
    """Compute the 3x3 rotation matrix from ECEF (ITRF) to ECI (GCRF).

    Alias for :func:`rotation_itrf_to_gcrf`.

    Args:
        eop: EOP data.
        epc: Epoch (UTC).

    Returns:
        3x3 rotation matrix (ECEF -> ECI).
    """
    return rotation_itrf_to_gcrf(eop, epc)


def state_eci_to_ecef(eop: EOPData, epc: Epoch, x_eci: ArrayLike) -> Array:
    """Transform a 6-element state vector from ECI (GCRF) to ECEF (ITRF).

    Alias for :func:`state_gcrf_to_itrf`.

    Args:
        eop: EOP data.
        epc: Epoch (UTC).
        x_eci: 6-element ECI state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        6-element ECEF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.
    """
    return state_gcrf_to_itrf(eop, epc, x_eci)


def state_ecef_to_eci(eop: EOPData, epc: Epoch, x_ecef: ArrayLike) -> Array:
    """Transform a 6-element state vector from ECEF (ITRF) to ECI (GCRF).

    Alias for :func:`state_itrf_to_gcrf`.

    Args:
        eop: EOP data.
        epc: Epoch (UTC).
        x_ecef: 6-element ECEF state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.

    Returns:
        6-element ECI state ``[x, y, z, vx, vy, vz]``.
            Units: m, m/s.
    """
    return state_itrf_to_gcrf(eop, epc, x_ecef)
