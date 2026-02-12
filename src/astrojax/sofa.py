"""JAX translations of IAU SOFA routines for Earth orientation modeling.

Implements the IAU 2006/2000A CIO-based precession-nutation model.
Uses routines and computations derived from software provided by SOFA
under license to the user. Does not itself constitute software provided
by and/or endorsed by SOFA.

All functions respect :func:`~astrojax.config.get_dtype` for float precision,
following the same pattern as all other astrojax modules.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from astrojax._sofa_nutation_data import LUNI_SOLAR_COEFFS, PLANETARY_COEFFS
from astrojax.attitude_representations import Rx, Ry, Rz
from astrojax.config import get_dtype

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DJ00: float = 2451545.0
"""Julian Date of J2000.0."""

DJC: float = 36525.0
"""Days per Julian century."""

DAS2R: float = 4.848136811095359935899141e-6
"""Arcseconds to radians."""

D2PI: float = 6.283185307179586476925287
"""2*pi."""

TURNAS: float = 1296000.0
"""Arcseconds in a full circle."""

MJD_ZERO: float = 2400000.5
"""Julian Date of MJD zero-point."""

# Units of 0.1 microarcsecond to radians
_U2R: float = DAS2R / 1e7


# ---------------------------------------------------------------------------
# Fundamental arguments (IERS Conventions 2003)
# ---------------------------------------------------------------------------


def fal03(t: Array) -> Array:
    """Mean anomaly of the Moon (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        l in radians.
    """
    return (
        jnp.fmod(
            485868.249036
            + t * (1717915923.2178 + t * (31.8792 + t * (0.051635 + t * (-0.00024470)))),
            TURNAS,
        )
        * DAS2R
    )


def falp03(t: Array) -> Array:
    """Mean anomaly of the Sun (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        l' in radians.
    """
    return (
        jnp.fmod(
            1287104.793048
            + t * (129596581.0481 + t * (-0.5532 + t * (0.000136 + t * (-0.00001149)))),
            TURNAS,
        )
        * DAS2R
    )


def faf03(t: Array) -> Array:
    """Mean argument of the latitude of the Moon (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        F in radians.
    """
    return (
        jnp.fmod(
            335779.526232
            + t * (1739527262.8478 + t * (-12.7512 + t * (-0.001037 + t * (0.00000417)))),
            TURNAS,
        )
        * DAS2R
    )


def fad03(t: Array) -> Array:
    """Mean elongation of the Moon from the Sun (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        D in radians.
    """
    return (
        jnp.fmod(
            1072260.703692
            + t * (1602961601.2090 + t * (-6.3706 + t * (0.006593 + t * (-0.00003169)))),
            TURNAS,
        )
        * DAS2R
    )


def faom03(t: Array) -> Array:
    """Mean longitude of the Moon's ascending node (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        Omega in radians.
    """
    return (
        jnp.fmod(
            450160.398036 + t * (-6962890.5431 + t * (7.4722 + t * (0.007702 + t * (-0.00005939)))),
            TURNAS,
        )
        * DAS2R
    )


def fame03(t: Array) -> Array:
    """Mean longitude of Mercury (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        Mean longitude in radians.
    """
    return jnp.fmod(4.402608842 + 2608.7903141574 * t, D2PI)


def fave03(t: Array) -> Array:
    """Mean longitude of Venus (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        Mean longitude in radians.
    """
    return jnp.fmod(3.176146697 + 1021.3285546211 * t, D2PI)


def fae03(t: Array) -> Array:
    """Mean longitude of Earth (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        Mean longitude in radians.
    """
    return jnp.fmod(1.753470314 + 628.3075849991 * t, D2PI)


def fama03(t: Array) -> Array:
    """Mean longitude of Mars (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        Mean longitude in radians.
    """
    return jnp.fmod(6.203480913 + 334.0612426700 * t, D2PI)


def faju03(t: Array) -> Array:
    """Mean longitude of Jupiter (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        Mean longitude in radians.
    """
    return jnp.fmod(0.599546497 + 52.9690962641 * t, D2PI)


def fasa03(t: Array) -> Array:
    """Mean longitude of Saturn (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        Mean longitude in radians.
    """
    return jnp.fmod(0.874016757 + 21.3299104960 * t, D2PI)


def faur03(t: Array) -> Array:
    """Mean longitude of Uranus (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        Mean longitude in radians.
    """
    return jnp.fmod(5.481293872 + 7.4781598567 * t, D2PI)


def fane03(t: Array) -> Array:
    """Mean longitude of Neptune (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        Mean longitude in radians.
    """
    return jnp.fmod(5.311886287 + 3.8133035638 * t, D2PI)


def fapa03(t: Array) -> Array:
    """General accumulated precession in longitude (IERS 2003).

    Args:
        t: TDB Julian centuries since J2000.0.

    Returns:
        General precession in radians.
    """
    return (0.024381750 + 0.00000538691 * t) * t


# ---------------------------------------------------------------------------
# Mean obliquity
# ---------------------------------------------------------------------------


def obl06(date1: Array, date2: Array) -> Array:
    """Mean obliquity of the ecliptic, IAU 2006 precession.

    Args:
        date1: TT as 2-part Julian Date (part 1).
        date2: TT as 2-part Julian Date (part 2).

    Returns:
        Obliquity of the ecliptic in radians.
    """
    t = ((date1 - DJ00) + date2) / DJC
    eps0 = 84381.406 + t * (
        -46.836769 + t * (-0.0001831 + t * (0.00200340 + t * (-0.000000576 + t * (-0.0000000434))))
    )
    return eps0 * DAS2R


# ---------------------------------------------------------------------------
# Fukushima-Williams angles
# ---------------------------------------------------------------------------


def pfw06(date1: Array, date2: Array) -> tuple[Array, Array, Array, Array]:
    """Precession angles, IAU 2006, Fukushima-Williams 4-angle formulation.

    Args:
        date1: TT as 2-part Julian Date (part 1).
        date2: TT as 2-part Julian Date (part 2).

    Returns:
        Tuple of (gamb, phib, psib, epsa) in radians.
    """
    t = ((date1 - DJ00) + date2) / DJC

    gamb = (
        -0.052928
        + t
        * (
            10.556378
            + t * (0.4932044 + t * (-0.00031238 + t * (-0.000002788 + t * (0.0000000260))))
        )
    ) * DAS2R

    phib = (
        84381.412819
        + t
        * (
            -46.811016
            + t * (0.0511268 + t * (0.00053289 + t * (-0.000000440 + t * (-0.0000000176))))
        )
    ) * DAS2R

    psib = (
        -0.041775
        + t
        * (
            5038.481484
            + t * (1.5584175 + t * (-0.00018522 + t * (-0.000026452 + t * (-0.0000000148))))
        )
    ) * DAS2R

    epsa = obl06(date1, date2)

    return gamb, phib, psib, epsa


# ---------------------------------------------------------------------------
# Nutation IAU 2000A
# ---------------------------------------------------------------------------


def nut00a(date1: Array, date2: Array) -> tuple[Array, Array]:
    """Nutation, IAU 2000A model (MHB2000 luni-solar and planetary).

    Vectorized implementation using matmul over all 678 luni-solar and
    687 planetary nutation terms.

    Args:
        date1: TT as 2-part Julian Date (part 1).
        date2: TT as 2-part Julian Date (part 2).

    Returns:
        Tuple of (dpsi, deps) nutation in longitude and obliquity [radians].
    """
    dtype = get_dtype()
    t = dtype(((date1 - DJ00) + date2) / DJC)

    # Fundamental (Delaunay) arguments
    el = fal03(t)
    elp = falp03(t)
    f = faf03(t)
    d = fad03(t)
    om = faom03(t)
    delaunay = jnp.array([el, elp, f, d, om])

    # Planetary longitudes
    al = fal03(t)
    af = faf03(t)
    ad = fad03(t)
    aom = faom03(t)
    ame = fame03(t)
    ave = fave03(t)
    ae = fae03(t)
    ama = fama03(t)
    aju = faju03(t)
    asa = fasa03(t)
    aur = faur03(t)
    ane = fane03(t)
    apa = fapa03(t)
    planetary = jnp.array([al, af, ad, aom, ame, ave, ae, ama, aju, asa, aur, ane, apa])

    # ---- Luni-solar nutation (678 terms) ----
    ls = jnp.array(LUNI_SOLAR_COEFFS, dtype=dtype)
    ls_nfa = ls[:, :5]  # (678, 5) integer multipliers
    ls_sp = ls[:, 5]  # sine coefficient for longitude
    ls_spt = ls[:, 6]  # time-dependent sine coefficient
    ls_cp = ls[:, 7]  # cosine coefficient for longitude
    ls_ce = ls[:, 8]  # cosine coefficient for obliquity
    ls_cet = ls[:, 9]  # time-dependent cosine coefficient
    ls_se = ls[:, 10]  # sine coefficient for obliquity

    # All arguments at once: (678, 5) @ (5,) -> (678,)
    ls_args = ls_nfa @ delaunay
    ls_sin = jnp.sin(ls_args)
    ls_cos = jnp.cos(ls_args)

    dpsi_ls = jnp.sum((ls_sp + ls_spt * t) * ls_sin + ls_cp * ls_cos)
    deps_ls = jnp.sum((ls_ce + ls_cet * t) * ls_cos + ls_se * ls_sin)

    # ---- Planetary nutation (687 terms) ----
    pl = jnp.array(PLANETARY_COEFFS, dtype=dtype)
    pl_nfa = pl[:, :13]  # (687, 13) integer multipliers
    pl_sp = pl[:, 13]  # sine coefficient for longitude
    pl_cp = pl[:, 14]  # cosine coefficient for longitude
    pl_se = pl[:, 15]  # sine coefficient for obliquity
    pl_ce = pl[:, 16]  # cosine coefficient for obliquity

    # All arguments at once: (687, 13) @ (13,) -> (687,)
    pl_args = pl_nfa @ planetary
    pl_sin = jnp.sin(pl_args)
    pl_cos = jnp.cos(pl_args)

    dpsi_pl = jnp.sum(pl_sp * pl_sin + pl_cp * pl_cos)
    deps_pl = jnp.sum(pl_se * pl_sin + pl_ce * pl_cos)

    # Total nutation (convert from 0.1 uas to radians)
    dpsi = (dpsi_ls + dpsi_pl) * _U2R
    deps = (deps_ls + deps_pl) * _U2R

    return dpsi, deps


# ---------------------------------------------------------------------------
# Nutation IAU 2006/2000A (P03 corrected)
# ---------------------------------------------------------------------------


def nut06a(date1: Array, date2: Array) -> tuple[Array, Array]:
    """Nutation, IAU 2006/2000A (with P03 precession adjustment).

    Applies the P03-compatible J2 correction factor to the IAU 2000A nutation.

    Args:
        date1: TT as 2-part Julian Date (part 1).
        date2: TT as 2-part Julian Date (part 2).

    Returns:
        Tuple of (dpsi, deps) nutation in longitude and obliquity [radians].
    """
    t = ((date1 - DJ00) + date2) / DJC

    # J2 correction factor for P03 precession
    fj2 = -2.7774e-6 * t

    # IAU 2000A nutation
    dp, de = nut00a(date1, date2)

    # Apply P03 adjustments
    dpsi = dp * (1.0 + 0.4697e-6 * fj2)
    deps = de * (1.0 + fj2)

    return dpsi, deps


# ---------------------------------------------------------------------------
# Fukushima-Williams angles to rotation matrix
# ---------------------------------------------------------------------------


def fw2m(gamb: Array, phib: Array, psi: Array, eps: Array) -> Array:
    """Fukushima-Williams angles to rotation matrix.

    Form the rotation matrix given the Fukushima-Williams angles:
    ``NxPxB = R_1(-eps) . R_3(-psi) . R_1(phib) . R_3(gamb)``

    Where R_1 and R_3 use the same convention as Rx/Rz
    (SOFA/ERFA convention with +sin in off-diagonal).

    Args:
        gamb: F-W angle gamma_bar (radians).
        phib: F-W angle phi_bar (radians).
        psi: F-W angle psi (radians).
        eps: F-W angle epsilon (radians).

    Returns:
        3x3 rotation matrix (NPB matrix).
    """
    return Rx(-eps) @ Rz(-psi) @ Rx(phib) @ Rz(gamb)


# ---------------------------------------------------------------------------
# Precession-nutation matrix
# ---------------------------------------------------------------------------


def pnm06a(date1: Array, date2: Array) -> Array:
    """Form the bias-precession-nutation matrix, IAU 2006/2000A.

    Combines Fukushima-Williams precession angles with IAU 2006/2000A
    nutation to form the complete GCRS-to-true (BPN) rotation matrix.

    Args:
        date1: TT as 2-part Julian Date (part 1).
        date2: TT as 2-part Julian Date (part 2).

    Returns:
        3x3 bias-precession-nutation matrix.
    """
    # Fukushima-Williams angles for frame bias and precession
    gamb, phib, psib, epsa = pfw06(date1, date2)

    # Nutation components
    dpsi, deps = nut06a(date1, date2)

    # Form the matrix: FW angles with nutation added
    return fw2m(gamb, phib, psib + dpsi, epsa + deps)


# ---------------------------------------------------------------------------
# Extract CIP coordinates from BPN matrix
# ---------------------------------------------------------------------------


def bpn2xy(rbpn: Array) -> tuple[Array, Array]:
    """Extract CIP X, Y coordinates from the BPN matrix.

    Args:
        rbpn: 3x3 bias-precession-nutation matrix.

    Returns:
        Tuple of (x, y) CIP coordinates.
    """
    return rbpn[2, 0], rbpn[2, 1]


# ---------------------------------------------------------------------------
# CIO locator s
# ---------------------------------------------------------------------------

# Polynomial coefficients for s + XY/2 (arcseconds -> radians at evaluation)
_S06_SP = (94.00e-6, 3808.65e-6, -122.68e-6, -72574.11e-6, 27.98e-6, 15.62e-6)

# S06 CIO locator series coefficient data.
# Stored as Python tuples to avoid module-level jnp.array calls
# (which would require x64 to be enabled at import time).
# Each group: (nfa_tuple, sc_tuple) where nfa has 8-element integer rows
# and sc has 2-element (sin, cos) coefficient rows.

# fmt: off
# Terms of order t^0 (33 terms)
_S06_S0_NFA_DATA = (
    (0,0,0,0,1,0,0,0), (0,0,0,0,2,0,0,0), (0,0,2,-2,3,0,0,0),
    (0,0,2,-2,1,0,0,0), (0,0,2,-2,2,0,0,0), (0,0,2,0,3,0,0,0),
    (0,0,2,0,1,0,0,0), (0,0,0,0,3,0,0,0), (0,1,0,0,1,0,0,0),
    (0,1,0,0,-1,0,0,0), (1,0,0,0,-1,0,0,0), (1,0,0,0,1,0,0,0),
    (0,1,2,-2,3,0,0,0), (0,1,2,-2,1,0,0,0), (0,0,4,-4,4,0,0,0),
    (0,0,1,-1,1,-8,12,0), (0,0,2,0,0,0,0,0), (0,0,2,0,2,0,0,0),
    (1,0,2,0,3,0,0,0), (1,0,2,0,1,0,0,0), (0,0,2,-2,0,0,0,0),
    (0,1,-2,2,-3,0,0,0), (0,1,-2,2,-1,0,0,0), (0,0,0,0,0,8,-13,-1),
    (0,0,0,2,0,0,0,0), (2,0,-2,0,-1,0,0,0), (0,1,2,-2,2,0,0,0),
    (1,0,0,-2,1,0,0,0), (1,0,0,-2,-1,0,0,0), (0,0,4,-2,4,0,0,0),
    (0,0,2,-2,4,0,0,0), (1,0,-2,0,-3,0,0,0), (1,0,-2,0,-1,0,0,0),
)
_S06_S0_SC_DATA = (
    (-2640.73e-6, 0.39e-6), (-63.53e-6, 0.02e-6), (-11.75e-6, -0.01e-6),
    (-11.21e-6, -0.01e-6), (4.57e-6, 0.00e-6), (-2.02e-6, 0.00e-6),
    (-1.98e-6, 0.00e-6), (1.72e-6, 0.00e-6), (1.41e-6, 0.01e-6),
    (1.26e-6, 0.01e-6), (0.63e-6, 0.00e-6), (0.63e-6, 0.00e-6),
    (-0.46e-6, 0.00e-6), (-0.45e-6, 0.00e-6), (-0.36e-6, 0.00e-6),
    (0.24e-6, 0.12e-6), (-0.32e-6, 0.00e-6), (-0.28e-6, 0.00e-6),
    (-0.27e-6, 0.00e-6), (-0.26e-6, 0.00e-6), (0.21e-6, 0.00e-6),
    (-0.19e-6, 0.00e-6), (-0.18e-6, 0.00e-6), (0.10e-6, -0.05e-6),
    (-0.15e-6, 0.00e-6), (0.14e-6, 0.00e-6), (0.14e-6, 0.00e-6),
    (-0.14e-6, 0.00e-6), (-0.14e-6, 0.00e-6), (-0.13e-6, 0.00e-6),
    (0.11e-6, 0.00e-6), (-0.11e-6, 0.00e-6), (-0.11e-6, 0.00e-6),
)

# Terms of order t^1 (3 terms)
_S06_S1_NFA_DATA = (
    (0,0,0,0,2,0,0,0), (0,0,0,0,1,0,0,0), (0,0,2,-2,3,0,0,0),
)
_S06_S1_SC_DATA = (
    (-0.07e-6, 3.57e-6), (1.73e-6, -0.03e-6), (0.00e-6, 0.48e-6),
)

# Terms of order t^2 (25 terms)
_S06_S2_NFA_DATA = (
    (0,0,0,0,1,0,0,0), (0,0,2,-2,2,0,0,0), (0,0,2,0,2,0,0,0),
    (0,0,0,0,2,0,0,0), (0,1,0,0,0,0,0,0), (1,0,0,0,0,0,0,0),
    (0,1,2,-2,2,0,0,0), (0,0,2,0,1,0,0,0), (1,0,2,0,2,0,0,0),
    (0,1,-2,2,-2,0,0,0), (1,0,0,-2,0,0,0,0), (0,0,2,-2,1,0,0,0),
    (1,0,-2,0,-2,0,0,0), (0,0,0,2,0,0,0,0), (1,0,0,0,1,0,0,0),
    (1,0,-2,-2,-2,0,0,0), (1,0,0,0,-1,0,0,0), (1,0,2,0,1,0,0,0),
    (2,0,0,-2,0,0,0,0), (2,0,-2,0,-1,0,0,0), (0,0,2,2,2,0,0,0),
    (2,0,2,0,2,0,0,0), (2,0,0,0,0,0,0,0), (1,0,2,-2,2,0,0,0),
    (0,0,2,0,0,0,0,0),
)
_S06_S2_SC_DATA = (
    (743.52e-6, -0.17e-6), (56.91e-6, 0.06e-6), (9.84e-6, -0.01e-6),
    (-8.85e-6, 0.01e-6), (-6.38e-6, -0.05e-6), (-3.07e-6, 0.00e-6),
    (2.23e-6, 0.00e-6), (1.67e-6, 0.00e-6), (1.30e-6, 0.00e-6),
    (0.93e-6, 0.00e-6), (0.68e-6, 0.00e-6), (-0.55e-6, 0.00e-6),
    (0.53e-6, 0.00e-6), (-0.27e-6, 0.00e-6), (-0.27e-6, 0.00e-6),
    (-0.26e-6, 0.00e-6), (-0.25e-6, 0.00e-6), (0.22e-6, 0.00e-6),
    (-0.21e-6, 0.00e-6), (0.20e-6, 0.00e-6), (0.17e-6, 0.00e-6),
    (0.13e-6, 0.00e-6), (-0.13e-6, 0.00e-6), (-0.12e-6, 0.00e-6),
    (-0.11e-6, 0.00e-6),
)

# Terms of order t^3 (4 terms)
_S06_S3_NFA_DATA = (
    (0,0,0,0,1,0,0,0), (0,0,2,-2,2,0,0,0),
    (0,0,2,0,2,0,0,0), (0,0,0,0,2,0,0,0),
)
_S06_S3_SC_DATA = (
    (0.30e-6, -23.42e-6), (-0.03e-6, -1.46e-6),
    (-0.01e-6, -0.25e-6), (0.00e-6, 0.23e-6),
)

# Terms of order t^4 (1 term)
_S06_S4_NFA_DATA = ((0,0,0,0,1,0,0,0),)
_S06_S4_SC_DATA = ((-0.26e-6, -0.01e-6),)
# fmt: on


def _s06_series(nfa: Array, sc: Array, fa: Array) -> Array:
    """Evaluate one order of the CIO locator series.

    Args:
        nfa: Integer multiplier array, shape (N, 8).
        sc: Sine/cosine coefficient array, shape (N, 2).
        fa: Fundamental arguments, shape (8,).

    Returns:
        Sum of s*sin(a) + c*cos(a) over all terms.
    """
    args = nfa @ fa  # (N,)
    return jnp.sum(sc[:, 0] * jnp.sin(args) + sc[:, 1] * jnp.cos(args))


def s06(date1: Array, date2: Array, x: Array, y: Array) -> Array:
    """CIO locator s, positioning the Celestial Intermediate Origin on the
    equator of the CIP. Compatible with IAU 2006/2000A precession-nutation.

    The series is actually for s + XY/2. The function subtracts XY/2 to
    return s itself.

    Args:
        date1: TT as 2-part Julian Date (part 1).
        date2: TT as 2-part Julian Date (part 2).
        x: CIP x coordinate.
        y: CIP y coordinate.

    Returns:
        CIO locator s in radians.
    """
    dtype = get_dtype()
    t = dtype(((date1 - DJ00) + date2) / DJC)

    # Fundamental arguments
    fa = jnp.array(
        [
            fal03(t),  # l
            falp03(t),  # l'
            faf03(t),  # F
            fad03(t),  # D
            faom03(t),  # Om
            fave03(t),  # LVe
            fae03(t),  # LE
            fapa03(t),  # pA
        ]
    )

    # Create coefficient arrays from tuple data at runtime
    s0_nfa = jnp.array(_S06_S0_NFA_DATA, dtype=dtype)
    s0_sc = jnp.array(_S06_S0_SC_DATA, dtype=dtype)
    s1_nfa = jnp.array(_S06_S1_NFA_DATA, dtype=dtype)
    s1_sc = jnp.array(_S06_S1_SC_DATA, dtype=dtype)
    s2_nfa = jnp.array(_S06_S2_NFA_DATA, dtype=dtype)
    s2_sc = jnp.array(_S06_S2_SC_DATA, dtype=dtype)
    s3_nfa = jnp.array(_S06_S3_NFA_DATA, dtype=dtype)
    s3_sc = jnp.array(_S06_S3_SC_DATA, dtype=dtype)
    s4_nfa = jnp.array(_S06_S4_NFA_DATA, dtype=dtype)
    s4_sc = jnp.array(_S06_S4_SC_DATA, dtype=dtype)

    # Evaluate series at each power of t
    w0 = dtype(_S06_SP[0]) + _s06_series(s0_nfa, s0_sc, fa)
    w1 = dtype(_S06_SP[1]) + _s06_series(s1_nfa, s1_sc, fa)
    w2 = dtype(_S06_SP[2]) + _s06_series(s2_nfa, s2_sc, fa)
    w3 = dtype(_S06_SP[3]) + _s06_series(s3_nfa, s3_sc, fa)
    w4 = dtype(_S06_SP[4]) + _s06_series(s4_nfa, s4_sc, fa)
    w5 = dtype(_S06_SP[5])

    # Horner form + convert arcseconds to radians, subtract XY/2
    s = (w0 + (w1 + (w2 + (w3 + (w4 + w5 * t) * t) * t) * t) * t) * DAS2R - x * y / 2.0

    return s


# ---------------------------------------------------------------------------
# Composite: X, Y, s
# ---------------------------------------------------------------------------


def xys06a(date1: Array, date2: Array) -> tuple[Array, Array, Array]:
    """CIP X, Y coordinates and CIO locator s, IAU 2006/2000A.

    Combines pnm06a, bpn2xy, and s06 to return the full set of
    CIO-based parameters.

    Args:
        date1: TT as 2-part Julian Date (part 1).
        date2: TT as 2-part Julian Date (part 2).

    Returns:
        Tuple of (x, y, s) where x,y are CIP coordinates and s is the
        CIO locator, all in radians.
    """
    # Form the bias-precession-nutation matrix
    rbpn = pnm06a(date1, date2)

    # Extract CIP X, Y
    x, y = bpn2xy(rbpn)

    # CIO locator s
    s_val = s06(date1, date2, x, y)

    return x, y, s_val


# ---------------------------------------------------------------------------
# CIP to celestial-to-intermediate matrix
# ---------------------------------------------------------------------------


def c2ixys(x: Array, y: Array, s: Array) -> Array:
    """Form the celestial-to-intermediate matrix given CIP X, Y and CIO locator s.

    Uses the method: ``Rz(-(e+s)) @ Ry(d) @ Rz(e)`` where
    ``d = arctan(sqrt((x^2 + y^2) / (1 - x^2 - y^2)))``
    and ``e = atan2(y, x)``.

    This matches the SOFA iauC2ixys implementation exactly (astrojax and SOFA
    use identical Rx/Ry/Rz conventions).

    Args:
        x: CIP x coordinate.
        y: CIP y coordinate.
        s: CIO locator.

    Returns:
        3x3 celestial-to-intermediate matrix.
    """
    # Derived quantities
    r2 = x * x + y * y
    e = jnp.where(r2 > 0.0, jnp.arctan2(y, x), 0.0)
    d = jnp.arctan(jnp.sqrt(r2 / (1.0 - r2)))

    return Rz(-(e + s)) @ Ry(d) @ Rz(e)


# ---------------------------------------------------------------------------
# Earth Rotation Angle
# ---------------------------------------------------------------------------


def era00(dj1: Array, dj2: Array) -> Array:
    """Earth Rotation Angle (IAU 2000 model).

    Args:
        dj1: UT1 as 2-part Julian Date (part 1).
        dj2: UT1 as 2-part Julian Date (part 2).

    Returns:
        Earth Rotation Angle in radians (0 to 2*pi).
    """
    # Days since J2000.0
    t = dj1 + dj2 - DJ00

    # Fractional part of dj1 + dj2
    f = jnp.fmod(dj1, 1.0) + jnp.fmod(dj2, 1.0)

    # Earth Rotation Angle
    theta = (
        jnp.fmod(
            f + 0.7790572732640 + 0.00273781191135448 * t,
            1.0,
        )
        * D2PI
    )

    return theta


# ---------------------------------------------------------------------------
# TIO locator s'
# ---------------------------------------------------------------------------


def sp00(date1: Array, date2: Array) -> Array:
    """TIO locator s', positioning the Terrestrial Intermediate Origin.

    The dominant terms of s' can be approximated as ``-47e-6 * t * DAS2R``
    where t is Julian centuries of TT since J2000.0.

    Args:
        date1: TT as 2-part Julian Date (part 1).
        date2: TT as 2-part Julian Date (part 2).

    Returns:
        TIO locator s' in radians.
    """
    t = ((date1 - DJ00) + date2) / DJC
    return -47e-6 * t * DAS2R


# ---------------------------------------------------------------------------
# Polar motion matrix
# ---------------------------------------------------------------------------


def pom00(xp: Array, yp: Array, sp: Array) -> Array:
    """Form the polar motion matrix (TIRS -> ITRS).

    The matrix is ``Rx(-yp) @ Ry(-xp) @ Rz(sp)``.

    This matches the SOFA iauPom00 implementation exactly (astrojax and SOFA
    use identical Rx/Ry/Rz conventions).

    Args:
        xp: Polar motion x-component (radians, positive towards Greenwich).
        yp: Polar motion y-component (radians, positive towards 270E).
        sp: TIO locator s' (radians).

    Returns:
        3x3 polar motion matrix.
    """
    return Rx(-yp) @ Ry(-xp) @ Rz(sp)
