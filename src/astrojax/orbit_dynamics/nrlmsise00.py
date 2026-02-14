"""NRLMSISE-00 atmospheric density model in pure JAX.

Provides temperature and density profiles of the Earth's atmosphere from
ground to thermospheric heights using the NRLMSISE-00 empirical model.
All functions are JIT-compatible and work with ``jax.jit``, ``jax.vmap``,
and ``jax.grad``.

This implementation is based on Dominik Brodowski's C implementation
(https://www.brodo.de/space/nrlmsise/), translated through
SatelliteDynamics.jl and the brahe Rust crate to pure JAX.

Typical usage::

    from astrojax.space_weather import load_default_sw
    from astrojax.orbit_dynamics.nrlmsise00 import density_nrlmsise00
    from astrojax.epoch import Epoch

    sw = load_default_sw()
    epc = Epoch.from_datetime(2020, 6, 1, 12, 0, 0.0)
    r_ecef = jnp.array([6778137.0, 0.0, 0.0])
    rho = density_nrlmsise00(sw, epc, r_ecef)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.coordinates import position_ecef_to_geodetic
from astrojax.epoch import Epoch
from astrojax.orbit_dynamics.nrlmsise00_data import (
    PAVGM,
    PD,
    PDL,
    PDM,
    PMA,
    PS,
    PT,
    PTL,
    PTM,
)
from astrojax.space_weather._lookup import (
    get_sw_ap_array,
    get_sw_f107_obs,
    get_sw_f107_obs_ctr81,
)
from astrojax.space_weather._types import SpaceWeatherData
from astrojax.time import caldate_to_mjd, mjd_to_caldate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SR: float = 7.2722e-5  # Angular velocity of Earth [rad/s]
_DGTR: float = 1.74533e-2  # Degrees to radians
_DR: float = 1.72142e-2  # Day-angle rate (2*pi/365.25 in rad/day)
_HR: float = 0.2618  # Hour angle rate (2*pi/24 in rad/hour)
_RGAS: float = 831.4  # Gas constant [J/(kmol·K)] (actually in cgs-compatible units)


# ---------------------------------------------------------------------------
# Utility helpers (all JIT-compatible)
# ---------------------------------------------------------------------------


def _glatf(lat: ArrayLike) -> tuple[Array, Array]:
    """Latitude-dependent gravity and effective Earth radius.

    Args:
        lat: Geodetic latitude [degrees].

    Returns:
        Tuple of (surface gravity [cm/s^2], effective Earth radius [km]).
    """
    c2 = jnp.cos(2.0 * _DGTR * lat)
    gv = 980.616 * (1.0 - 0.0026373 * c2)
    reff = 2.0 * gv / (3.085462e-6 + 2.27e-9 * c2) * 1.0e-5
    return gv, reff


def _ccor(alt: ArrayLike, r: ArrayLike, h1: ArrayLike, zh: ArrayLike) -> Array:
    """Chemistry/dissociation correction factor.

    Args:
        alt: Altitude [km].
        r: Target ratio.
        h1: Transition scale length.
        zh: Altitude of 1/2 R.

    Returns:
        Correction coefficient.
    """
    e = (alt - zh) / h1
    e_clipped = jnp.clip(e, -70.0, 70.0)
    ex = jnp.exp(e_clipped)
    return jnp.where(
        e > 70.0,
        1.0,
        jnp.where(e < -70.0, jnp.exp(r), jnp.exp(r / (1.0 + ex))),
    )


def _ccor2(alt: ArrayLike, r: ArrayLike, h1: ArrayLike, zh: ArrayLike, h2: ArrayLike) -> Array:
    """Chemistry/dissociation correction with two scale lengths.

    Args:
        alt: Altitude [km].
        r: Target ratio.
        h1: Transition scale length.
        zh: Altitude of 1/2 R.
        h2: Second transition scale length.

    Returns:
        Correction coefficient.
    """
    e1 = (alt - zh) / h1
    e2 = (alt - zh) / h2
    e1_clipped = jnp.clip(e1, -70.0, 70.0)
    e2_clipped = jnp.clip(e2, -70.0, 70.0)
    ex1 = jnp.exp(e1_clipped)
    ex2 = jnp.exp(e2_clipped)
    return jnp.where(
        (e1 > 70.0) | (e2 > 70.0),
        1.0,
        jnp.where(
            (e1 < -70.0) & (e2 < -70.0),
            jnp.exp(r),
            jnp.exp(r / (1.0 + 0.5 * (ex1 + ex2))),
        ),
    )


def _scaleh(
    alt: ArrayLike, xm: ArrayLike, temp: ArrayLike, gsurf: ArrayLike, re: ArrayLike
) -> Array:
    """Scale height.

    Args:
        alt: Altitude [km].
        xm: Molecular weight.
        temp: Temperature [K].
        gsurf: Surface gravity [cm/s^2].
        re: Effective Earth radius [km].

    Returns:
        Scale height [km].
    """
    g = gsurf / (1.0 + alt / re) ** 2
    return _RGAS * temp / (g * xm)


def _dnet(dd: ArrayLike, dm: ArrayLike, zhm: ArrayLike, xmm: ArrayLike, xm: ArrayLike) -> Array:
    """Turbopause correction combining diffusive and mixed densities.

    Args:
        dd: Diffusive density.
        dm: Full mixed density.
        zhm: Transition scale length.
        xmm: Full mixed molecular weight.
        xm: Species molecular weight.

    Returns:
        Combined density.
    """
    a = zhm / (xmm - xm)

    # Handle edge cases where dm or dd are zero/negative
    both_zero = (dd == 0.0) & (dm == 0.0)
    dm_zero = dm == 0.0
    dd_zero = dd == 0.0
    neither_positive = ~((dm > 0.0) & (dd > 0.0))

    # Safe log: avoid log(0) by adding a tiny value
    safe_dm = jnp.where(dm > 0.0, dm, 1.0)
    safe_dd = jnp.where(dd > 0.0, dd, 1.0)
    ylog = a * jnp.log(safe_dm / safe_dd)

    normal_result = jnp.where(
        ylog < -10.0,
        dd,
        jnp.where(ylog > 10.0, dm, dd * (1.0 + jnp.exp(ylog)) ** (1.0 / a)),
    )

    # Handle edge cases
    result = jnp.where(
        neither_positive,
        jnp.where(both_zero, 1.0, jnp.where(dm_zero, dd, jnp.where(dd_zero, dm, normal_result))),
        normal_result,
    )
    return result


def _zeta(zz: ArrayLike, zl: ArrayLike, re: ArrayLike) -> Array:
    """Geopotential altitude.

    Args:
        zz: Geometric altitude [km].
        zl: Reference altitude [km].
        re: Effective Earth radius [km].

    Returns:
        Geopotential altitude.
    """
    return (zz - zl) * (re + zl) / (re + zz)


# ---------------------------------------------------------------------------
# Spline routines (small n, fully unrolled at trace time)
# ---------------------------------------------------------------------------


def _spline(x: Array, y: Array, n: int, yp1: ArrayLike, ypn: ArrayLike) -> Array:
    """Compute cubic spline second derivatives.

    Args:
        x: Knot x-coordinates, shape ``(n,)``.
        y: Knot y-values, shape ``(n,)``.
        n: Number of knots (compile-time constant, <= 10).
        yp1: First derivative at left boundary.
        ypn: First derivative at right boundary.

    Returns:
        Second derivatives at knots, shape ``(n,)``.
    """
    u = jnp.zeros(n)
    y2 = jnp.zeros(n)

    # Left boundary condition
    natural_left = yp1 > 0.99e30
    y2_0 = jnp.where(natural_left, 0.0, -0.5)
    u_0 = jnp.where(
        natural_left,
        0.0,
        (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1),
    )
    y2 = y2.at[0].set(y2_0)
    u = u.at[0].set(u_0)

    # Forward sweep (n is small, compile-time unroll)
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2 = y2.at[i].set((sig - 1.0) / p)
        u_val = (
            6.0
            * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
            / (x[i + 1] - x[i - 1])
            - sig * u[i - 1]
        ) / p
        u = u.at[i].set(u_val)

    # Right boundary condition
    natural_right = ypn > 0.99e30
    qn = jnp.where(natural_right, 0.0, 0.5)
    un = jnp.where(
        natural_right,
        0.0,
        (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])),
    )
    y2 = y2.at[n - 1].set((un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0))

    # Back substitution
    for k in range(n - 2, -1, -1):
        y2 = y2.at[k].set(y2[k] * y2[k + 1] + u[k])

    return y2


def _splint(xa: Array, ya: Array, y2a: Array, n: int, x: ArrayLike) -> Array:
    """Evaluate cubic spline interpolation at ``x``.

    Args:
        xa: Knot x-coordinates, shape ``(n,)``.
        ya: Knot y-values, shape ``(n,)``.
        y2a: Second derivatives from :func:`_spline`, shape ``(n,)``.
        n: Number of knots.
        x: Evaluation point.

    Returns:
        Interpolated value.
    """
    # Binary search for interval (n is small, unrolled)
    klo = jnp.int32(0)
    khi = jnp.int32(n - 1)
    for _ in range(20):  # enough iterations for n<=10
        k = (khi + klo) // 2
        khi = jnp.where(xa[k] > x, k, khi)
        klo = jnp.where(xa[k] > x, klo, k)
        # Stop when adjacent
        done = (khi - klo) <= 1
        klo = jnp.where(done, klo, klo)
        khi = jnp.where(done, khi, khi)

    h = xa[khi] - xa[klo]
    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h

    return a * ya[klo] + b * ya[khi] + ((a**3 - a) * y2a[klo] + (b**3 - b) * y2a[khi]) * h * h / 6.0


def _splini(xa: Array, ya: Array, y2a: Array, n: int, x: ArrayLike) -> Array:
    """Integrate cubic spline from ``xa[0]`` to ``x``.

    Args:
        xa: Knot x-coordinates, shape ``(n,)``.
        ya: Knot y-values, shape ``(n,)``.
        y2a: Second derivatives from :func:`_spline`, shape ``(n,)``.
        n: Number of knots.
        x: Upper integration limit.

    Returns:
        Integral value.
    """
    yi = jnp.float64(0.0) if get_dtype() == jnp.float64 else jnp.float32(0.0)

    for i in range(n - 1):
        klo = i
        khi = i + 1

        # Determine upper limit for this segment
        xx = jnp.where((khi < n - 1) & (x >= xa[khi]), xa[khi], x)

        # Only accumulate if x > xa[klo]
        h = xa[khi] - xa[klo]
        a = (xa[khi] - xx) / h
        b = (xx - xa[klo]) / h
        a2 = a * a
        b2 = b * b

        segment = (
            (1.0 - a2) * ya[klo] / 2.0
            + b2 * ya[khi] / 2.0
            + (
                (-(1.0 + a2 * a2) / 4.0 + a2 / 2.0) * y2a[klo]
                + (b2 * b2 / 4.0 - b2 / 2.0) * y2a[khi]
            )
            * h
            * h
            / 6.0
        ) * h

        yi = jnp.where(x > xa[klo], yi + segment, yi)

    return yi


# ---------------------------------------------------------------------------
# Lower atmosphere temperature/density (densm)
# ---------------------------------------------------------------------------


def _densm(
    alt: ArrayLike,
    d0: ArrayLike,
    xm: ArrayLike,
    tz_in: ArrayLike,
    mn3: int,
    zn3: Array,
    tn3: Array,
    tgn3: Array,
    mn2: int,
    zn2: Array,
    tn2: Array,
    tgn2: Array,
    gsurf: ArrayLike,
    re: ArrayLike,
) -> tuple[Array, Array]:
    """Temperature and density for lower atmosphere.

    Computes profiles through the stratosphere/mesosphere (zn2) and
    optionally troposphere/stratosphere (zn3) layers using cubic spline
    interpolation.

    Args:
        alt: Altitude [km].
        d0: Input density or temperature.
        xm: Molecular weight (0 for temperature-only).
        tz_in: Input temperature [K].
        mn3: Number of zn3 nodes.
        zn3: Lower-layer altitude nodes [km].
        tn3: Lower-layer temperature nodes [K].
        tgn3: Lower-layer temperature gradients.
        mn2: Number of zn2 nodes.
        zn2: Upper-layer altitude nodes [km].
        tn2: Upper-layer temperature nodes [K].
        tgn2: Upper-layer temperature gradients.
        gsurf: Surface gravity [cm/s^2].
        re: Effective Earth radius [km].

    Returns:
        Tuple of (temperature [K], density or temperature).
    """
    densm_tmp = d0

    # Above zn2[0]: return immediately
    above_zn2 = alt > zn2[0]

    # --- Stratosphere/Mesosphere (zn2 layer) ---
    z = jnp.where(alt > zn2[mn2 - 1], alt, zn2[mn2 - 1])
    z1 = zn2[0]
    z2 = zn2[mn2 - 1]
    t1 = tn2[0]
    t2 = tn2[mn2 - 1]
    zg = _zeta(z, z1, re)
    zgdif = _zeta(z2, z1, re)

    # Spline node setup
    xs2 = jnp.zeros(10)
    ys2 = jnp.zeros(10)
    for k in range(mn2):
        xs2 = xs2.at[k].set(_zeta(zn2[k], z1, re) / zgdif)
        ys2 = ys2.at[k].set(1.0 / tn2[k])

    yd1 = -tgn2[0] / (t1 * t1) * zgdif
    yd2 = -tgn2[1] / (t2 * t2) * zgdif * ((re + z2) / (re + z1)) ** 2

    y2out = _spline(xs2[:mn2], ys2[:mn2], mn2, yd1, yd2)
    x_val = zg / zgdif
    y_val = _splint(xs2[:mn2], ys2[:mn2], y2out, mn2, x_val)

    tz = 1.0 / y_val

    # Density calculation for zn2 layer (only if xm != 0)
    glb = gsurf / (1.0 + z1 / re) ** 2
    gamm = xm * glb * zgdif / _RGAS
    yi = _splini(xs2[:mn2], ys2[:mn2], y2out, mn2, x_val)
    expl = jnp.clip(gamm * yi, -50.0, 50.0)
    densm_tmp_zn2 = densm_tmp * (t1 / tz) * jnp.exp(-expl)

    # Update densm_tmp based on xm
    densm_tmp = jnp.where(xm != 0.0, densm_tmp_zn2, densm_tmp)

    # Above zn3[0]: stop here
    above_zn3 = alt > zn3[0]

    # --- Troposphere/Stratosphere (zn3 layer) ---
    z = alt
    z1_3 = zn3[0]
    z2_3 = zn3[mn3 - 1]
    t1_3 = tn3[0]
    t2_3 = tn3[mn3 - 1]
    zg_3 = _zeta(z, z1_3, re)
    zgdif_3 = _zeta(z2_3, z1_3, re)

    xs3 = jnp.zeros(10)
    ys3 = jnp.zeros(10)
    for k in range(mn3):
        xs3 = xs3.at[k].set(_zeta(zn3[k], z1_3, re) / zgdif_3)
        ys3 = ys3.at[k].set(1.0 / tn3[k])

    yd1_3 = -tgn3[0] / (t1_3 * t1_3) * zgdif_3
    yd2_3 = -tgn3[1] / (t2_3 * t2_3) * zgdif_3 * ((re + z2_3) / (re + z1_3)) ** 2

    y2out_3 = _spline(xs3[:mn3], ys3[:mn3], mn3, yd1_3, yd2_3)
    x_val_3 = zg_3 / zgdif_3
    y_val_3 = _splint(xs3[:mn3], ys3[:mn3], y2out_3, mn3, x_val_3)

    tz_3 = 1.0 / y_val_3

    # Density for zn3 layer
    glb_3 = gsurf / (1.0 + z1_3 / re) ** 2
    gamm_3 = xm * glb_3 * zgdif_3 / _RGAS
    yi_3 = _splini(xs3[:mn3], ys3[:mn3], y2out_3, mn3, x_val_3)
    expl_3 = jnp.clip(gamm_3 * yi_3, -50.0, 50.0)
    densm_tmp_zn3 = densm_tmp * (t1_3 / tz_3) * jnp.exp(-expl_3)

    # Select final values
    tz_final = jnp.where(above_zn2, tz_in, jnp.where(above_zn3, tz, tz_3))
    densm_final = jnp.where(
        above_zn2,
        d0,
        jnp.where(
            above_zn3,
            jnp.where(xm != 0.0, densm_tmp, tz),
            jnp.where(xm != 0.0, densm_tmp_zn3, tz_3),
        ),
    )

    return tz_final, densm_final


# ---------------------------------------------------------------------------
# Upper atmosphere temperature/density (densu)
# ---------------------------------------------------------------------------


def _densu(
    alt: ArrayLike,
    dlb: ArrayLike,
    tinf: ArrayLike,
    tlb: ArrayLike,
    xm: ArrayLike,
    alpha: ArrayLike,
    tz_in: ArrayLike,
    zlb: ArrayLike,
    s2: ArrayLike,
    mn1: int,
    zn1: Array,
    tn1: Array,
    tgn1: Array,
    gsurf: ArrayLike,
    re: ArrayLike,
) -> tuple[Array, Array]:
    """Temperature and density for upper atmosphere (Bates profile).

    Computes profiles using a Bates temperature model above the lower
    thermosphere boundary, with cubic spline interpolation below.

    Args:
        alt: Altitude [km].
        dlb: Density at lower boundary.
        tinf: Exospheric temperature [K].
        tlb: Temperature at lower boundary [K].
        xm: Molecular weight (0 for temperature-only).
        alpha: Thermal diffusion coefficient.
        tz_in: Input temperature [K].
        zlb: Lower boundary altitude [km].
        s2: Shape parameter.
        mn1: Number of temperature nodes.
        zn1: Temperature node altitudes [km].
        tn1: Temperature node values [K].
        tgn1: Temperature gradients at boundaries.
        gsurf: Surface gravity [cm/s^2].
        re: Effective Earth radius [km].

    Returns:
        Tuple of (temperature [K], density).
    """
    za = zn1[0]
    z = jnp.where(alt > za, alt, za)

    # Geopotential altitude difference from ZLB
    zg2 = _zeta(z, zlb, re)

    # Bates temperature
    tt = tinf - (tinf - tlb) * jnp.exp(-s2 * zg2)
    ta = tt
    tz = tt

    # Temperature gradient at za from Bates profile
    dta = (tinf - ta) * s2 * ((re + zlb) / (re + za)) ** 2

    # Build spline temperature profile below za
    tn1_mod = tn1.at[0].set(ta)
    tgn1_mod = tgn1.at[0].set(dta)

    z_below = jnp.where(alt > zn1[mn1 - 1], alt, zn1[mn1 - 1])
    z1 = zn1[0]
    z2 = zn1[mn1 - 1]
    t1 = tn1_mod[0]
    t2 = tn1_mod[mn1 - 1]

    zg = _zeta(z_below, z1, re)
    zgdif = _zeta(z2, z1, re)

    xs = jnp.zeros(5)
    ys = jnp.zeros(5)
    for k in range(mn1):
        xs = xs.at[k].set(_zeta(zn1[k], z1, re) / zgdif)
        ys = ys.at[k].set(1.0 / tn1_mod[k])

    yd1 = -tgn1_mod[0] / (t1 * t1) * zgdif
    yd2 = -tgn1_mod[1] / (t2 * t2) * zgdif * ((re + z2) / (re + z1)) ** 2

    y2out = _spline(xs[:mn1], ys[:mn1], mn1, yd1, yd2)
    x_val = zg / zgdif
    y_val = _splint(xs[:mn1], ys[:mn1], y2out, mn1, x_val)
    tz_below = 1.0 / y_val

    # Choose temperature based on altitude
    tz = jnp.where(alt < za, tz_below, tz)

    # If xm == 0, return temperature only
    # Calculate density above za
    glb = gsurf / (1.0 + zlb / re) ** 2
    gamma = xm * glb / (s2 * _RGAS * tinf)
    expl_above = jnp.clip(-s2 * gamma * zg2, -50.0, 50.0)
    expl_above = jnp.where(tt <= 0.0, -50.0, expl_above)

    densa = (
        dlb * (tlb / jnp.where(tt > 0.0, tt, 1.0)) ** (1.0 + alpha + gamma) * jnp.exp(expl_above)
    )

    # Calculate density below za using spline integration
    glb_below = gsurf / (1.0 + z1 / re) ** 2
    gamm_below = xm * glb_below * zgdif / _RGAS

    # Recompute spline for integration (same knots)
    yi = _splini(xs[:mn1], ys[:mn1], y2out, mn1, x_val)
    expl_below = jnp.clip(gamm_below * yi, -50.0, 50.0)
    expl_below = jnp.where(tz_below <= 0.0, 50.0, expl_below)

    densa_below = (
        densa
        * (t1 / jnp.where(tz_below > 0.0, tz_below, 1.0)) ** (1.0 + alpha)
        * jnp.exp(-expl_below)
    )

    densu_temp = jnp.where(alt >= za, densa, densa_below)
    densu_temp = jnp.where(xm == 0.0, tz, densu_temp)

    return tz, densu_temp


# ---------------------------------------------------------------------------
# G(L) parametric functions (globe7, glob7s)
# ---------------------------------------------------------------------------


def _g0(a: ArrayLike, p: Array) -> Array:
    """Equation A24a helper."""
    sqrt_p24sq = jnp.sqrt(p[24] * p[24])
    safe_sqrt = jnp.where(sqrt_p24sq > 0.0, sqrt_p24sq, 1e-30)
    return a - 4.0 + (p[25] - 1.0) * (a - 4.0 + (jnp.exp(-safe_sqrt * (a - 4.0)) - 1.0) / safe_sqrt)


def _sumex(ex: ArrayLike) -> Array:
    """Equation A24c helper."""
    return 1.0 + (1.0 - ex**19) / (1.0 - ex) * ex**0.5


def _sg0(ex: ArrayLike, p: Array, ap: Array) -> Array:
    """Equation A24a: Ap-dependent magnetic activity function."""
    return (
        _g0(ap[1], p)
        + (
            _g0(ap[2], p) * ex
            + _g0(ap[3], p) * ex**2
            + _g0(ap[4], p) * ex**3
            + (_g0(ap[5], p) * ex**4 + _g0(ap[6], p) * ex**12) * (1.0 - ex**8) / (1.0 - ex)
        )
    ) / _sumex(ex)


def _globe7(
    p: Array,
    doy: ArrayLike,
    sec: ArrayLike,
    g_lat: ArrayLike,
    g_lon: ArrayLike,
    lst: ArrayLike,
    f107a: ArrayLike,
    f107: ArrayLike,
    ap: ArrayLike,
    ap_array: Array,
    use_ap_array: bool = True,
) -> tuple[
    Array,  # tinf
    Array,  # dfa
    Array,  # plg (4, 9)
    Array,  # ctloc
    Array,  # stloc
    Array,  # c2tloc
    Array,  # s2tloc
    Array,  # c3tloc
    Array,  # s3tloc
    Array,  # apdf
    Array,  # apt (4,)
]:
    """Main parametric G(L) function for thermospheric profiles.

    Computes the combined effect of solar activity, geomagnetic activity,
    latitude, longitude, local solar time, and season on atmospheric
    parameters using Legendre polynomials and harmonic terms.

    Args:
        p: Coefficient array, shape ``(150,)``.
        doy: Day of year.
        sec: Seconds in day (UT).
        g_lat: Geodetic latitude [degrees].
        g_lon: Geodetic longitude [degrees].
        lst: Local apparent solar time [hours].
        f107a: 81-day average F10.7 flux.
        f107: Daily F10.7 flux.
        ap: Daily Ap magnetic index.
        ap_array: 7-element Ap array for NRLMSISE-00.

    Returns:
        Tuple of (tinf contribution, dfa, plg, ctloc, stloc,
        c2tloc, s2tloc, c3tloc, s3tloc, apdf, apt).
    """
    t = jnp.zeros(15)

    # Legendre polynomials
    c = jnp.sin(g_lat * _DGTR)
    s = jnp.cos(g_lat * _DGTR)
    c2 = c * c
    c4 = c2 * c2
    s2 = s * s

    plg = jnp.zeros((4, 9))
    plg = plg.at[0, 1].set(c)
    plg = plg.at[0, 2].set(0.5 * (3.0 * c2 - 1.0))
    plg = plg.at[0, 3].set(0.5 * (5.0 * c * c2 - 3.0 * c))
    plg = plg.at[0, 4].set((35.0 * c4 - 30.0 * c2 + 3.0) / 8.0)
    plg = plg.at[0, 5].set((63.0 * c2 * c2 * c - 70.0 * c2 * c + 15.0 * c) / 8.0)
    plg = plg.at[0, 6].set((11.0 * c * plg[0, 5] - 5.0 * plg[0, 4]) / 6.0)

    plg = plg.at[1, 1].set(s)
    plg = plg.at[1, 2].set(3.0 * c * s)
    plg = plg.at[1, 3].set(1.5 * (5.0 * c2 - 1.0) * s)
    plg = plg.at[1, 4].set(2.5 * (7.0 * c2 * c - 3.0 * c) * s)
    plg = plg.at[1, 5].set(1.875 * (21.0 * c4 - 14.0 * c2 + 1.0) * s)
    plg = plg.at[1, 6].set((11.0 * c * plg[1, 5] - 6.0 * plg[1, 4]) / 5.0)

    plg = plg.at[2, 2].set(3.0 * s2)
    plg = plg.at[2, 3].set(15.0 * s2 * c)
    plg = plg.at[2, 4].set(7.5 * (7.0 * c2 - 1.0) * s2)
    plg = plg.at[2, 5].set(3.0 * c * plg[2, 4] - 2.0 * plg[2, 3])
    plg = plg.at[2, 6].set((11.0 * c * plg[2, 5] - 7.0 * plg[2, 4]) / 4.0)
    plg = plg.at[2, 7].set((13.0 * c * plg[2, 6] - 8.0 * plg[2, 5]) / 5.0)

    plg = plg.at[3, 3].set(15.0 * s2 * s)
    plg = plg.at[3, 4].set(105.0 * s2 * s * c)
    plg = plg.at[3, 5].set((9.0 * c * plg[3, 4] - 7.0 * plg[3, 3]) / 2.0)
    plg = plg.at[3, 6].set((11.0 * c * plg[3, 5] - 8.0 * plg[3, 4]) / 3.0)

    # Local solar time harmonics
    tloc = lst
    stloc = jnp.sin(_HR * tloc)
    ctloc = jnp.cos(_HR * tloc)
    s2tloc = jnp.sin(2.0 * _HR * tloc)
    c2tloc = jnp.cos(2.0 * _HR * tloc)
    s3tloc = jnp.sin(3.0 * _HR * tloc)
    c3tloc = jnp.cos(3.0 * _HR * tloc)

    # Seasonal variations
    cd32 = jnp.cos(_DR * (doy - p[31]))
    cd18 = jnp.cos(2.0 * _DR * (doy - p[17]))
    cd14 = jnp.cos(_DR * (doy - p[13]))
    cd39 = jnp.cos(2.0 * _DR * (doy - p[38]))

    # F10.7 effect
    df = f107 - f107a
    dfa = f107a - 150.0
    t = t.at[0].set(
        p[19] * df * (1.0 + p[59] * dfa) + p[20] * df * df + p[21] * dfa + p[29] * dfa**2
    )
    f1 = 1.0 + (p[47] * dfa + p[19] * df + p[20] * df * df) * 1.0  # swc[1]=1
    f2 = 1.0 + (p[49] * dfa + p[19] * df + p[20] * df * df) * 1.0  # swc[1]=1

    # Time independent
    t = t.at[1].set(
        (p[1] * plg[0, 2] + p[2] * plg[0, 4] + p[22] * plg[0, 6])
        + (p[14] * plg[0, 2]) * dfa * 1.0  # swc[1]=1
        + p[26] * plg[0, 1]
    )

    # Symmetrical annual
    t = t.at[2].set(p[18] * cd32)

    # Symmetrical semiannual
    t = t.at[3].set((p[15] + p[16] * plg[0, 2]) * cd18)

    # Asymmetrical annual
    t = t.at[4].set(f1 * (p[9] * plg[0, 1] + p[10] * plg[0, 3]) * cd14)

    # Asymmetrical semiannual
    t = t.at[5].set(p[37] * plg[0, 1] * cd39)

    # Diurnal (sw[7]=1)
    t71 = (p[11] * plg[1, 2]) * cd14 * 1.0  # swc[5]=1
    t72 = (p[12] * plg[1, 2]) * cd14 * 1.0  # swc[5]=1
    t = t.at[6].set(
        f2
        * (
            (p[3] * plg[1, 1] + p[4] * plg[1, 3] + p[27] * plg[1, 5] + t71) * ctloc
            + (p[6] * plg[1, 1] + p[7] * plg[1, 3] + p[28] * plg[1, 5] + t72) * stloc
        )
    )

    # Semidiurnal (sw[8]=1)
    t81 = (p[23] * plg[2, 3] + p[35] * plg[2, 5]) * cd14 * 1.0  # swc[5]=1
    t82 = (p[33] * plg[2, 3] + p[36] * plg[2, 5]) * cd14 * 1.0  # swc[5]=1
    t = t.at[7].set(
        f2
        * (
            (p[5] * plg[2, 2] + p[41] * plg[2, 4] + t81) * c2tloc
            + (p[8] * plg[2, 2] + p[42] * plg[2, 4] + t82) * s2tloc
        )
    )

    # Terdiurnal (sw[14]=1)
    t = t.at[13].set(
        f2
        * (
            (p[39] * plg[3, 3] + (p[93] * plg[3, 4] + p[46] * plg[3, 6]) * cd14 * 1.0) * s3tloc
            + (p[40] * plg[3, 3] + (p[94] * plg[3, 4] + p[48] * plg[3, 6]) * cd14 * 1.0) * c3tloc
        )
    )

    # Magnetic activity
    apt = jnp.zeros(4)
    apdf = jnp.float64(0.0) if get_dtype() == jnp.float64 else jnp.float32(0.0)

    if use_ap_array:
        # AP array mode (sw[9]=-1): use structured magnetic activity
        exp1_raw = jnp.where(
            p[51] != 0.0,
            jnp.exp(
                -10800.0
                * jnp.sqrt(p[51] * p[51])
                / (1.0 + p[138] * (45.0 - jnp.sqrt(g_lat * g_lat)))
            ),
            0.0,
        )
        exp1 = jnp.clip(exp1_raw, 0.0, 0.99999)
        apt = apt.at[0].set(_sg0(exp1, p, ap_array))

        t = t.at[8].set(
            apt[0]
            * (
                p[50]
                + p[96] * plg[0, 2]
                + p[54] * plg[0, 4]
                + (p[125] * plg[0, 1] + p[126] * plg[0, 3] + p[127] * plg[0, 5])
                * cd14
                * 1.0  # swc[5]=1
                + (p[128] * plg[1, 1] + p[129] * plg[1, 3] + p[130] * plg[1, 5])
                * 1.0  # swc[7]=1
                * jnp.cos(_HR * (tloc - p[131]))
            )
        )
    else:
        # Daily AP mode (sw[9]=1): simple daily Ap index
        apd = ap - 4.0
        p44 = jnp.where(p[43] < 0.0, 1.0e-5, p[43])
        p45 = p[44]
        apdf = apd + (p45 - 1.0) * (apd + (jnp.exp(-p44 * apd) - 1.0) / p44)

        t = t.at[8].set(
            apdf
            * (
                p[32]
                + p[45] * plg[0, 2]
                + p[34] * plg[0, 4]
                + (p[100] * plg[0, 1] + p[101] * plg[0, 3] + p[102] * plg[0, 5])
                * cd14
                * 1.0  # swc[5]=1
                + (p[121] * plg[1, 1] + p[122] * plg[1, 3] + p[123] * plg[1, 5])
                * 1.0  # swc[7]=1
                * jnp.cos(_HR * (tloc - p[124]))
            )
        )

    # Longitudinal (sw[10]=1, sw[11]=1)
    t = t.at[10].set(
        (1.0 + p[80] * dfa * 1.0)  # swc[1]=1
        * (
            (
                p[64] * plg[1, 2]
                + p[65] * plg[1, 4]
                + p[66] * plg[1, 6]
                + p[103] * plg[1, 1]
                + p[104] * plg[1, 3]
                + p[105] * plg[1, 5]
                + 1.0  # swc[5]=1
                * (p[109] * plg[1, 1] + p[110] * plg[1, 3] + p[111] * plg[1, 5])
                * cd14
            )
            * jnp.cos(_DGTR * g_lon)
            + (
                p[90] * plg[1, 2]
                + p[91] * plg[1, 4]
                + p[92] * plg[1, 6]
                + p[106] * plg[1, 1]
                + p[107] * plg[1, 3]
                + p[108] * plg[1, 5]
                + 1.0  # swc[5]=1
                * (p[112] * plg[1, 1] + p[113] * plg[1, 3] + p[114] * plg[1, 5])
                * cd14
            )
            * jnp.sin(_DGTR * g_lon)
        )
    )

    # UT and mixed UT/longitude (sw[12]=1)
    t11_base = (
        (1.0 + p[95] * plg[0, 1])
        * (1.0 + p[81] * dfa * 1.0)  # swc[1]=1
        * (1.0 + p[119] * plg[0, 1] * 1.0 * cd14)  # swc[5]=1
        * (
            (p[68] * plg[0, 1] + p[69] * plg[0, 3] + p[70] * plg[0, 5])
            * jnp.cos(_SR * (sec - p[71]))
        )
    )
    t11_mixed = (
        1.0  # swc[11]=1
        * (p[76] * plg[2, 3] + p[77] * plg[2, 5] + p[78] * plg[2, 7])
        * jnp.cos(_SR * (sec - p[79]) + 2.0 * _DGTR * g_lon)
        * (1.0 + p[137] * dfa * 1.0)  # swc[1]=1
    )
    t = t.at[11].set(t11_base + t11_mixed)

    # UT longitude magnetic activity (sw[13]=1)
    if use_ap_array:
        t = t.at[12].set(
            apt[0]
            * 1.0  # swc[11]=1
            * (1.0 + p[132] * plg[0, 1])
            * (
                (p[52] * plg[1, 2] + p[98] * plg[1, 4] + p[67] * plg[1, 6])
                * jnp.cos(_DGTR * (g_lon - p[97]))
            )
            + apt[0]
            * 1.0  # swc[11]=1
            * 1.0  # swc[5]=1
            * (p[133] * plg[1, 1] + p[134] * plg[1, 3] + p[135] * plg[1, 5])
            * cd14
            * jnp.cos(_DGTR * (g_lon - p[136]))
            + apt[0]
            * 1.0  # swc[12]=1
            * (p[55] * plg[0, 1] + p[56] * plg[0, 3] + p[57] * plg[0, 5])
            * jnp.cos(_SR * (sec - p[58]))
        )
    else:
        t = t.at[12].set(
            apdf
            * 1.0  # swc[11]=1
            * (1.0 + p[120] * plg[0, 1])
            * (
                (p[60] * plg[1, 2] + p[61] * plg[1, 4] + p[62] * plg[1, 6])
                * jnp.cos(_DGTR * (g_lon - p[63]))
            )
            + apdf
            * 1.0  # swc[11]=1
            * 1.0  # swc[5]=1
            * (p[115] * plg[1, 1] + p[116] * plg[1, 3] + p[117] * plg[1, 5])
            * cd14
            * jnp.cos(_DGTR * (g_lon - p[118]))
            + apdf
            * 1.0  # swc[12]=1
            * (p[83] * plg[0, 1] + p[84] * plg[0, 3] + p[85] * plg[0, 5])
            * jnp.cos(_SR * (sec - p[75]))
        )

    # Sum contributions: tinf = p[30] + sum(|sw[i+1]| * t[i]) for i in 0..14
    # With all switches = 1: tinf = p[30] + sum(t[0:14])
    tinf = p[30]
    for i in range(14):
        tinf = tinf + jnp.abs(1.0) * t[i]  # sw[i+1] = 1.0

    return tinf, dfa, plg, ctloc, stloc, c2tloc, s2tloc, c3tloc, s3tloc, apdf, apt


def _glob7s(
    p: Array,
    doy: ArrayLike,
    g_lat: ArrayLike,
    g_lon: ArrayLike,
    lst: ArrayLike,
    sec: ArrayLike,
    dfa: ArrayLike,
    plg: Array,
    ctloc: ArrayLike,
    stloc: ArrayLike,
    c2tloc: ArrayLike,
    s2tloc: ArrayLike,
    s3tloc: ArrayLike,
    c3tloc: ArrayLike,
    ap_array: Array,
) -> Array:
    """Simplified parametric G(L) function for lower atmosphere.

    Args:
        p: Coefficient array, shape ``(100,)``.
        doy: Day of year.
        g_lat: Geodetic latitude [degrees].
        g_lon: Geodetic longitude [degrees].
        lst: Local solar time [hours].
        sec: Seconds in day (UT).
        dfa: F10.7a - 150.
        plg: Legendre polynomials, shape ``(4, 9)``.
        ctloc: cos(HR*lst).
        stloc: sin(HR*lst).
        c2tloc: cos(2*HR*lst).
        s2tloc: sin(2*HR*lst).
        s3tloc: sin(3*HR*lst).
        c3tloc: cos(3*HR*lst).
        ap_array: 7-element Ap array.

    Returns:
        Scalar G(L) contribution.
    """
    t = jnp.zeros(14)

    cd32 = jnp.cos(_DR * (doy - p[31]))
    cd18 = jnp.cos(2.0 * _DR * (doy - p[17]))
    cd14 = jnp.cos(_DR * (doy - p[13]))
    cd39 = jnp.cos(2.0 * _DR * (doy - p[38]))

    # F10.7
    t = t.at[0].set(p[21] * dfa)

    # Time independent
    t = t.at[1].set(
        p[1] * plg[0, 2]
        + p[2] * plg[0, 4]
        + p[22] * plg[0, 6]
        + p[26] * plg[0, 1]
        + p[14] * plg[0, 3]
        + p[59] * plg[0, 5]
    )

    # Symmetrical annual
    t = t.at[2].set((p[18] + p[47] * plg[0, 2] + p[29] * plg[0, 4]) * cd32)

    # Symmetrical semiannual
    t = t.at[3].set((p[15] + p[16] * plg[0, 2] + p[30] * plg[0, 4]) * cd18)

    # Asymmetrical annual
    t = t.at[4].set((p[9] * plg[0, 1] + p[10] * plg[0, 3] + p[20] * plg[0, 5]) * cd14)

    # Asymmetrical semiannual
    t = t.at[5].set((p[37] * plg[0, 1]) * cd39)

    # Diurnal (sw[7]=1)
    t71 = p[11] * plg[1, 2] * cd14 * 1.0  # swc[5]=1
    t72 = p[12] * plg[1, 2] * cd14 * 1.0  # swc[5]=1
    t = t.at[6].set(
        (p[3] * plg[1, 1] + p[4] * plg[1, 3] + t71) * ctloc
        + (p[6] * plg[1, 1] + p[7] * plg[1, 3] + t72) * stloc
    )

    # Semidiurnal (sw[8]=1)
    t81 = (p[23] * plg[2, 3] + p[35] * plg[2, 5]) * cd14 * 1.0  # swc[5]=1
    t82 = (p[33] * plg[2, 3] + p[36] * plg[2, 5]) * cd14 * 1.0  # swc[5]=1
    t = t.at[7].set(
        (p[5] * plg[2, 2] + p[41] * plg[2, 4] + t81) * c2tloc
        + (p[8] * plg[2, 2] + p[42] * plg[2, 4] + t82) * s2tloc
    )

    # Terdiurnal (sw[14]=1)
    t = t.at[13].set(p[39] * plg[3, 3] * s3tloc + p[40] * plg[3, 3] * c3tloc)

    # Magnetic activity (sw[10]=1, sw[11]=1)
    t = t.at[10].set(
        (
            1.0
            + plg[0, 1]
            * (
                p[80] * 1.0 * jnp.cos(_DR * (doy - p[81]))  # swc[5]=1
                + p[85] * 1.0 * jnp.cos(2.0 * _DR * (doy - p[86]))  # swc[6]=1
            )
            + p[83] * 1.0 * jnp.cos(_DR * (doy - p[84]))  # swc[3]=1
            + p[87] * 1.0 * jnp.cos(2.0 * _DR * (doy - p[88]))  # swc[4]=1
        )
        * (
            (
                p[64] * plg[1, 2]
                + p[65] * plg[1, 4]
                + p[66] * plg[1, 6]
                + p[74] * plg[1, 1]
                + p[75] * plg[1, 3]
                + p[76] * plg[1, 5]
            )
            * jnp.cos(_DGTR * g_lon)
            + (
                p[90] * plg[1, 2]
                + p[91] * plg[1, 4]
                + p[92] * plg[1, 6]
                + p[77] * plg[1, 1]
                + p[78] * plg[1, 3]
                + p[79] * plg[1, 5]
            )
            * jnp.sin(_DGTR * g_lon)
        )
    )

    # Sum: tt = sum(|sw[i+1]| * t[i]) for i in 0..14 (all sw=1)
    tt = jnp.float64(0.0) if get_dtype() == jnp.float64 else jnp.float32(0.0)
    for i in range(14):
        tt = tt + jnp.abs(1.0) * t[i]

    return tt


# ---------------------------------------------------------------------------
# Thermospheric driver (gts7)
# ---------------------------------------------------------------------------


def _gts7(
    doy: ArrayLike,
    sec: ArrayLike,
    alt: ArrayLike,
    g_lat: ArrayLike,
    g_lon: ArrayLike,
    lst: ArrayLike,
    f107a: ArrayLike,
    f107: ArrayLike,
    ap: ArrayLike,
    ap_array: Array,
    gsurf: ArrayLike,
    re: ArrayLike,
    use_ap_array: bool = True,
) -> tuple[
    Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
]:
    """Thermospheric species densities and temperatures (alt > 72.5 km).

    This is the core model function for computing number densities of
    individual atmospheric species and temperature in the thermosphere.

    Args:
        doy: Day of year.
        sec: Seconds in day (UT).
        alt: Altitude [km].
        g_lat: Geodetic latitude [degrees].
        g_lon: Geodetic longitude [degrees].
        lst: Local solar time [hours].
        f107a: 81-day average F10.7 flux.
        f107: Daily F10.7 flux.
        ap: Daily Ap index.
        ap_array: 7-element Ap array.
        gsurf: Surface gravity [cm/s^2].
        re: Effective Earth radius [km].

    Returns:
        Tuple of (d[9], t[2], dm28, tn1[5], tgn1[2], dfa, plg, ctloc,
        stloc, c2tloc, s2tloc, s3tloc, c3tloc).
    """
    mn1 = 5
    alpha = jnp.array([-0.38, 0.0, 0.0, 0.0, 0.17, 0.0, -0.38, 0.0, 0.0])
    altl = jnp.array([200.0, 300.0, 160.0, 250.0, 240.0, 450.0, 320.0, 450.0])

    za = PDL[1, 15]
    zn1 = jnp.array([za, 110.0, 100.0, 90.0, 72.5])

    d = jnp.zeros(9)
    t_out = jnp.zeros(2)

    # Globe7 for temperature
    tinf_contrib, dfa, plg, ctloc, stloc, c2tloc, s2tloc, c3tloc, s3tloc, _apdf, _apt = _globe7(
        PT, doy, sec, g_lat, g_lon, lst, f107a, f107, ap, ap_array, use_ap_array
    )

    tinf = jnp.where(
        alt > zn1[0],
        PTM[0] * PT[0] * (1.0 + 1.0 * tinf_contrib),  # sw[16]=1
        PTM[0] * PT[0],
    )
    t_out = t_out.at[0].set(tinf)

    # Globe7 for gradient
    g0_contrib, _, _, _, _, _, _, _, _, _, _ = _globe7(
        PS, doy, sec, g_lat, g_lon, lst, f107a, f107, ap, ap_array, use_ap_array
    )
    g0_val = jnp.where(
        alt > zn1[4],
        PTM[3] * PS[0] * (1.0 + 1.0 * g0_contrib),  # sw[19]=1
        PTM[3] * PS[0],
    )

    # TLB (temperature at lower boundary)
    tlb_contrib, _, _, _, _, _, _, _, _, _, _ = _globe7(
        PD[3], doy, sec, g_lat, g_lon, lst, f107a, f107, ap, ap_array, use_ap_array
    )
    tlb = PTM[1] * (1.0 + 1.0 * tlb_contrib) * PD[3, 0]  # sw[17]=1
    s = g0_val / (tinf - tlb)

    # Lower thermosphere temperature variations
    glob7s_args = (
        doy,
        g_lat,
        g_lon,
        lst,
        sec,
        dfa,
        plg,
        ctloc,
        stloc,
        c2tloc,
        s2tloc,
        s3tloc,
        c3tloc,
        ap_array,
    )

    # Meso_tn1 and meso_tgn1 setup
    # Below 300 km: compute lower thermosphere temps; above 300 km: use base values
    meso_tn1 = jnp.zeros(5)
    meso_tgn1 = jnp.zeros(2)

    ptl0_val = _glob7s(PTL[0], *glob7s_args)
    tn1_1_below = PTM[6] * PTL[0, 0] / (1.0 - 1.0 * ptl0_val)  # sw[18]=1
    tn1_1_above = PTM[6] * PTL[0, 0]
    meso_tn1 = meso_tn1.at[1].set(jnp.where(alt < 300.0, tn1_1_below, tn1_1_above))

    ptl1_val = _glob7s(PTL[1], *glob7s_args)
    tn1_2_below = PTM[2] * PTL[1, 0] / (1.0 - 1.0 * ptl1_val)  # sw[18]=1
    tn1_2_above = PTM[2] * PTL[1, 0]
    meso_tn1 = meso_tn1.at[2].set(jnp.where(alt < 300.0, tn1_2_below, tn1_2_above))

    ptl2_val = _glob7s(PTL[2], *glob7s_args)
    tn1_3_below = PTM[7] * PTL[2, 0] / (1.0 - 1.0 * ptl2_val)  # sw[18]=1
    tn1_3_above = PTM[7] * PTL[2, 0]
    meso_tn1 = meso_tn1.at[3].set(jnp.where(alt < 300.0, tn1_3_below, tn1_3_above))

    ptl3_val = _glob7s(PTL[3], *glob7s_args)
    tn1_4_below = PTM[4] * PTL[3, 0] / (1.0 - 1.0 * 1.0 * ptl3_val)  # sw[18]*sw[20]=1
    tn1_4_above = PTM[4] * PTL[3, 0]
    meso_tn1 = meso_tn1.at[4].set(jnp.where(alt < 300.0, tn1_4_below, tn1_4_above))

    pma8_val = _glob7s(PMA[8], *glob7s_args)
    tgn1_1_below = (
        PTM[8]
        * PMA[8, 0]
        * (1.0 + 1.0 * 1.0 * pma8_val)  # sw[18]*sw[20]=1
        * meso_tn1[4]
        * meso_tn1[4]
        / (PTM[4] * PTL[3, 0]) ** 2
    )
    tgn1_1_above = PTM[8] * PMA[8, 0] * meso_tn1[4] * meso_tn1[4] / (PTM[4] * PTL[3, 0]) ** 2
    meso_tgn1 = meso_tgn1.at[1].set(jnp.where(alt < 300.0, tgn1_1_below, tgn1_1_above))

    # N2 variations factor at Zlb
    g28_contrib, _, _, _, _, _, _, _, _, _, _ = _globe7(
        PD[2], doy, sec, g_lat, g_lon, lst, f107a, f107, ap, ap_array, use_ap_array
    )
    g28 = 1.0 * g28_contrib  # sw[21]=1

    # Variation of turbopause height
    zhf = PDL[1, 24] * (
        1.0 + 1.0 * PDL[0, 24] * jnp.sin(_DGTR * g_lat) * jnp.cos(_DR * (doy - PT[13]))
    )  # sw[5]=1

    t_out = t_out.at[0].set(tinf)
    xmm = PDM[2, 4]
    z = alt

    # ===== N2 density =====
    db28 = PDM[2, 0] * jnp.exp(g28) * PD[2, 0]
    tz, d3 = _densu(
        z, db28, tinf, tlb, 28.0, alpha[2], 0.0, PTM[5], s, mn1, zn1, meso_tn1, meso_tgn1, gsurf, re
    )
    d = d.at[2].set(d3)
    t_out = t_out.at[1].set(tz)

    # Turbopause
    zh28 = PDM[2, 2] * zhf
    zhm28 = PDM[2, 3] * PDL[1, 5]
    xmd = 28.0 - xmm

    # Mixed density at Zlb
    tz_mix, b28 = _densu(
        zh28,
        db28,
        tinf,
        tlb,
        xmd,
        alpha[2] - 1.0,
        tz,
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )

    # Mixed density at alt (sw[15]=1)
    _tz_new, dm28_val = _densu(
        z, b28, tinf, tlb, xmm, alpha[2], tz, PTM[5], s, mn1, zn1, meso_tn1, meso_tgn1, gsurf, re
    )
    d = d.at[2].set(jnp.where(z <= altl[2], _dnet(d[2], dm28_val, zhm28, xmm, 28.0), d[2]))
    dm28 = jnp.where(z <= altl[2], dm28_val, 0.0)

    # ===== He density =====
    g4_contrib, _, _, _, _, _, _, _, _, _, _ = _globe7(
        PD[0], doy, sec, g_lat, g_lon, lst, f107a, f107, ap, ap_array, use_ap_array
    )
    g4 = 1.0 * g4_contrib  # sw[21]=1
    db04 = PDM[0, 0] * jnp.exp(g4) * PD[0, 0]

    _tz, d1 = _densu(
        z,
        db04,
        tinf,
        tlb,
        4.0,
        alpha[0],
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    d = d.at[0].set(d1)

    # Turbopause (sw[15]=1)
    zh04 = PDM[0, 2]
    _tz_new, b04 = _densu(
        zh04,
        db04,
        tinf,
        tlb,
        4.0 - xmm,
        alpha[0] - 1.0,
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    _tz_new, dm04 = _densu(
        z, b04, tinf, tlb, xmm, 0.0, t_out[1], PTM[5], s, mn1, zn1, meso_tn1, meso_tgn1, gsurf, re
    )
    zhm04 = zhm28

    d_he_net = _dnet(d[0], dm04, zhm04, xmm, 4.0)
    rl = jnp.log(b28 * PDM[0, 1] / b04)
    zc04 = PDM[0, 4] * PDL[1, 0]
    hc04 = PDM[0, 5] * PDL[1, 1]
    d_he_corrected = d_he_net * _ccor(z, rl, hc04, zc04)

    d = d.at[0].set(jnp.where(z < altl[0], d_he_corrected, d[0]))

    # ===== O density =====
    g16_contrib, _, _, _, _, _, _, _, _, _, _ = _globe7(
        PD[1], doy, sec, g_lat, g_lon, lst, f107a, f107, ap, ap_array, use_ap_array
    )
    g16 = 1.0 * g16_contrib  # sw[21]=1
    db16 = PDM[1, 0] * jnp.exp(g16) * PD[1, 0]

    _tz, d2 = _densu(
        z,
        db16,
        tinf,
        tlb,
        16.0,
        alpha[1],
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    d = d.at[1].set(d2)

    # Turbopause (sw[15]=1)
    zh16 = PDM[1, 2]
    _tz_new, b16 = _densu(
        zh16,
        db16,
        tinf,
        tlb,
        16.0 - xmm,
        alpha[1] - 1.0,
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    _tz_new, dm16 = _densu(
        z, b16, tinf, tlb, xmm, 0.0, t_out[1], PTM[5], s, mn1, zn1, meso_tn1, meso_tgn1, gsurf, re
    )
    zhm16 = zhm28

    d_o_net = _dnet(d[1], dm16, zhm16, xmm, 16.0)
    rl_o = PDM[1, 1] * PDL[1, 16] * (1.0 + 1.0 * PDL[0, 23] * (f107a - 150.0))  # sw[1]=1
    hc16 = PDM[1, 5] * PDL[1, 3]
    zc16 = PDM[1, 4] * PDL[1, 2]
    hc216 = PDM[1, 5] * PDL[1, 4]
    d_o_corrected = d_o_net * _ccor2(z, rl_o, hc16, zc16, hc216)

    hcc16 = PDM[1, 7] * PDL[1, 13]
    zcc16 = PDM[1, 6] * PDL[1, 12]
    rc16 = PDM[1, 3] * PDL[1, 14]
    d_o_corrected = d_o_corrected * _ccor(z, rc16, hcc16, zcc16)

    d = d.at[1].set(jnp.where(z <= altl[1], d_o_corrected, d[1]))

    # ===== O2 density =====
    g32_contrib, _, _, _, _, _, _, _, _, _, _ = _globe7(
        PD[4], doy, sec, g_lat, g_lon, lst, f107a, f107, ap, ap_array, use_ap_array
    )
    g32 = 1.0 * g32_contrib  # sw[21]=1
    db32 = PDM[3, 0] * jnp.exp(g32) * PD[4, 0]

    _tz, d4 = _densu(
        z,
        db32,
        tinf,
        tlb,
        32.0,
        alpha[3],
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    d = d.at[3].set(d4)

    # Turbopause (sw[15]=1)
    zh32 = PDM[3, 2]
    _tz_new, b32 = _densu(
        zh32,
        db32,
        tinf,
        tlb,
        32.0 - xmm,
        alpha[3] - 1.0,
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    _tz_new, dm32 = _densu(
        z, b32, tinf, tlb, xmm, 0.0, t_out[1], PTM[5], s, mn1, zn1, meso_tn1, meso_tgn1, gsurf, re
    )
    zhm32 = zhm28

    d_o2_net = _dnet(d[3], dm32, zhm32, xmm, 32.0)
    rl_o2 = jnp.log(b28 * PDM[3, 1] / b32)
    hc32 = PDM[3, 5] * PDL[1, 7]
    zc32 = PDM[3, 4] * PDL[1, 6]
    d_o2_corrected = d_o2_net * _ccor(z, rl_o2, hc32, zc32)
    d = d.at[3].set(jnp.where(z <= altl[3], d_o2_corrected, d[3]))

    # O2 correction for departure from diffusive equilibrium
    hcc32 = PDM[3, 7] * PDL[1, 22]
    hcc232 = PDM[3, 7] * PDL[0, 22]
    zcc32 = PDM[3, 6] * PDL[1, 21]
    rc32 = PDM[3, 3] * PDL[1, 23] * (1.0 + 1.0 * PDL[0, 23] * (f107a - 150.0))  # sw[1]=1
    d = d.at[3].set(d[3] * _ccor2(z, rc32, hcc32, zcc32, hcc232))

    # ===== Ar density =====
    g40_contrib, _, _, _, _, _, _, _, _, _, _ = _globe7(
        PD[5], doy, sec, g_lat, g_lon, lst, f107a, f107, ap, ap_array, use_ap_array
    )
    g40 = 1.0 * g40_contrib  # sw[21]=1
    db40 = PDM[4, 0] * jnp.exp(g40) * PD[5, 0]

    _tz, d5 = _densu(
        z,
        db40,
        tinf,
        tlb,
        40.0,
        alpha[4],
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    d = d.at[4].set(d5)

    # Turbopause (sw[15]=1)
    zh40 = PDM[4, 2]
    _tz_new, b40 = _densu(
        zh40,
        db40,
        tinf,
        tlb,
        40.0 - xmm,
        alpha[4] - 1.0,
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    _tz_new, dm40 = _densu(
        z, b40, tinf, tlb, xmm, 0.0, t_out[1], PTM[5], s, mn1, zn1, meso_tn1, meso_tgn1, gsurf, re
    )
    zhm40 = zhm28

    d_ar_net = _dnet(d[4], dm40, zhm40, xmm, 40.0)
    rl_ar = jnp.log(b28 * PDM[4, 1] / b40)
    hc40 = PDM[4, 5] * PDL[1, 9]
    zc40 = PDM[4, 4] * PDL[1, 8]
    d_ar_corrected = d_ar_net * _ccor(z, rl_ar, hc40, zc40)

    d = d.at[4].set(jnp.where(z <= altl[4], d_ar_corrected, d[4]))

    # ===== H density =====
    g1_contrib, _, _, _, _, _, _, _, _, _, _ = _globe7(
        PD[6], doy, sec, g_lat, g_lon, lst, f107a, f107, ap, ap_array, use_ap_array
    )
    g1 = 1.0 * g1_contrib  # sw[21]=1
    db01 = PDM[5, 0] * jnp.exp(g1) * PD[6, 0]

    _tz, d7 = _densu(
        z,
        db01,
        tinf,
        tlb,
        1.0,
        alpha[6],
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    d = d.at[6].set(d7)

    # Turbopause (sw[15]=1)
    zh01 = PDM[5, 2]
    _tz_new, b01 = _densu(
        zh01,
        db01,
        tinf,
        tlb,
        1.0 - xmm,
        alpha[6] - 1.0,
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    _tz_new, dm01 = _densu(
        z, b01, tinf, tlb, xmm, 0.0, t_out[1], PTM[5], s, mn1, zn1, meso_tn1, meso_tgn1, gsurf, re
    )
    zhm01 = zhm28

    d_h_net = _dnet(d[6], dm01, zhm01, xmm, 1.0)
    rl_h = jnp.log(b28 * PDM[5, 1] * jnp.sqrt(PDL[1, 17] * PDL[1, 17]) / b01)
    hc01 = PDM[5, 5] * PDL[1, 11]
    zc01 = PDM[5, 4] * PDL[1, 10]
    d_h_corrected = d_h_net * _ccor(z, rl_h, hc01, zc01)

    hcc01 = PDM[5, 7] * PDL[1, 19]
    zcc01 = PDM[5, 6] * PDL[1, 18]
    rc01 = PDM[5, 3] * PDL[1, 20]
    d_h_corrected = d_h_corrected * _ccor(z, rc01, hcc01, zcc01)

    d = d.at[6].set(jnp.where(z <= altl[6], d_h_corrected, d[6]))

    # ===== Atomic N density =====
    g14_contrib, _, _, _, _, _, _, _, _, _, _ = _globe7(
        PD[7], doy, sec, g_lat, g_lon, lst, f107a, f107, ap, ap_array, use_ap_array
    )
    g14 = 1.0 * g14_contrib  # sw[21]=1
    db14 = PDM[6, 0] * jnp.exp(g14) * PD[7, 0]

    _tz, d8 = _densu(
        z,
        db14,
        tinf,
        tlb,
        14.0,
        alpha[7],
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    d = d.at[7].set(d8)

    # Turbopause (sw[15]=1)
    zh14 = PDM[6, 2]
    _tz_new, b14 = _densu(
        zh14,
        db14,
        tinf,
        tlb,
        14.0 - xmm,
        alpha[7] - 1.0,
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    _tz_new, dm14 = _densu(
        z, b14, tinf, tlb, xmm, 0.0, t_out[1], PTM[5], s, mn1, zn1, meso_tn1, meso_tgn1, gsurf, re
    )
    zhm14 = zhm28

    d_n_net = _dnet(d[7], dm14, zhm14, xmm, 14.0)
    rl_n = jnp.log(b28 * PDM[6, 1] * jnp.sqrt(PDL[0, 2] * PDL[0, 2]) / b14)
    hc14 = PDM[6, 5] * PDL[0, 1]
    zc14 = PDM[6, 4] * PDL[0, 0]
    d_n_corrected = d_n_net * _ccor(z, rl_n, hc14, zc14)

    hcc14 = PDM[6, 7] * PDL[0, 4]
    zcc14 = PDM[6, 6] * PDL[0, 3]
    rc14 = PDM[6, 3] * PDL[0, 5]
    d_n_corrected = d_n_corrected * _ccor(z, rc14, hcc14, zcc14)

    d = d.at[7].set(jnp.where(z <= altl[7], d_n_corrected, d[7]))

    # ===== Anomalous O density =====
    g16h_contrib, _, _, _, _, _, _, _, _, _, _ = _globe7(
        PD[8], doy, sec, g_lat, g_lon, lst, f107a, f107, ap, ap_array, use_ap_array
    )
    g16h = 1.0 * g16h_contrib  # sw[21]=1
    db16h = PDM[7, 0] * jnp.exp(g16h) * PD[8, 0]
    tho = PDM[7, 9] * PDL[0, 6]

    tz_anom, dd_anom = _densu(
        z,
        db16h,
        tho,
        tho,
        16.0,
        alpha[8],
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    t_out = t_out.at[1].set(tz_anom)

    zsht = PDM[7, 5]
    zmho = PDM[7, 4]
    zsho = _scaleh(zmho, 16.0, tho, gsurf, re)
    d = d.at[8].set(dd_anom * jnp.exp(-zsht / zsho * (jnp.exp(-(z - zmho) / zsht) - 1.0)))

    # ===== Total mass density =====
    d = d.at[5].set(
        1.66e-24
        * (
            4.0 * d[0]
            + 16.0 * d[1]
            + 28.0 * d[2]
            + 32.0 * d[3]
            + 40.0 * d[4]
            + 1.0 * d[6]
            + 14.0 * d[7]
        )
    )

    # Temperature at altitude
    z_abs = jnp.abs(alt)
    tz_final, _ddum = _densu(
        z_abs,
        1.0,
        tinf,
        tlb,
        0.0,
        0.0,
        t_out[1],
        PTM[5],
        s,
        mn1,
        zn1,
        meso_tn1,
        meso_tgn1,
        gsurf,
        re,
    )
    t_out = t_out.at[1].set(tz_final)

    # Convert units: switches[0] = 1 => SI (m^-3 for number densities, kg/m^3 for mass)
    # Number densities: cm^-3 * 1e6 = m^-3
    # Mass density d[5]: in g/cm^3 before conversion; * 1e6 / 1000 = * 1000 = kg/m^3
    #   (since 1 g/cm^3 = 1000 kg/m^3)
    d = d * 1.0e6
    d = d.at[5].set(d[5] / 1000.0)

    return (
        d,
        t_out,
        dm28,
        meso_tn1,
        meso_tgn1,
        dfa,
        plg,
        ctloc,
        stloc,
        c2tloc,
        s2tloc,
        s3tloc,
        c3tloc,
    )


# ---------------------------------------------------------------------------
# Main drivers (gtd7, gtd7d)
# ---------------------------------------------------------------------------


def gtd7(
    doy: ArrayLike,
    sec: ArrayLike,
    alt: ArrayLike,
    g_lat: ArrayLike,
    g_lon: ArrayLike,
    lst: ArrayLike,
    f107a: ArrayLike,
    f107: ArrayLike,
    ap: ArrayLike,
    ap_array: Array,
    use_ap_array: bool = True,
) -> tuple[Array, Array]:
    """NRLMSISE-00 main driver (thermosphere + lower atmosphere).

    Computes atmospheric density and temperature at a given location
    and time. For altitudes above 72.5 km the thermospheric model is
    used directly; below that, mesospheric/stratospheric/tropospheric
    profiles are blended in.

    All model switches are set to 1 (full model), with switch[0]=1
    (SI units) and switch[9]=-1 (use AP array).

    Args:
        doy: Day of year (1-366).
        sec: Seconds in day (UT).
        alt: Altitude [km].
        g_lat: Geodetic latitude [degrees].
        g_lon: Geodetic longitude [degrees].
        lst: Local apparent solar time [hours].
        f107a: 81-day average F10.7 flux.
        f107: Daily F10.7 flux.
        ap: Daily Ap magnetic index.
        ap_array: 7-element Ap array for NRLMSISE-00.

    Returns:
        Tuple of (temperatures [2], densities [9]) where:
            - temperatures[0]: Exospheric temperature [K]
            - temperatures[1]: Temperature at altitude [K]
            - densities[0]: He number density [m^-3]
            - densities[1]: O number density [m^-3]
            - densities[2]: N2 number density [m^-3]
            - densities[3]: O2 number density [m^-3]
            - densities[4]: Ar number density [m^-3]
            - densities[5]: Total mass density [kg/m^3]
            - densities[6]: H number density [m^-3]
            - densities[7]: N number density [m^-3]
            - densities[8]: Anomalous O number density [m^-3]
    """
    # Latitude variation of gravity (sw[2]=1, so use actual latitude)
    gsurf, re = _glatf(g_lat)

    xmm = PDM[2, 4]

    mn3 = 5
    mn2 = 4
    zn3 = jnp.array([32.5, 20.0, 15.0, 10.0, 0.0])
    zn2 = jnp.array([72.5, 55.0, 45.0, 32.5])
    zmix = 62.5

    # Compute at alt or zn2[0], whichever is higher
    altt = jnp.where(alt > zn2[0], alt, zn2[0])

    (
        d_therm,
        t_therm,
        dm28_raw,
        meso_tn1,
        meso_tgn1,
        dfa,
        plg,
        ctloc,
        stloc,
        c2tloc,
        s2tloc,
        s3tloc,
        c3tloc,
    ) = _gts7(doy, sec, altt, g_lat, g_lon, lst, f107a, f107, ap, ap_array, gsurf, re, use_ap_array)

    dm28m = dm28_raw * 1.0e6  # switches[0]=1 => multiply by 1e6

    # For altitudes above zn2[0], return thermospheric values directly
    # For lower altitudes, blend with mesosphere/stratosphere/troposphere

    # --- Lower atmosphere blending ---
    glob7s_args = (
        doy,
        g_lat,
        g_lon,
        lst,
        sec,
        dfa,
        plg,
        ctloc,
        stloc,
        c2tloc,
        s2tloc,
        s3tloc,
        c3tloc,
        ap_array,
    )

    # Mesosphere/stratosphere temperature profile (zn2 layer)
    meso_tgn2 = jnp.zeros(2)
    meso_tn2 = jnp.zeros(4)

    meso_tgn2 = meso_tgn2.at[0].set(meso_tgn1[1])
    meso_tn2 = meso_tn2.at[0].set(meso_tn1[4])

    pma0_val = _glob7s(PMA[0], *glob7s_args)
    meso_tn2 = meso_tn2.at[1].set(PMA[0, 0] * PAVGM[0] / (1.0 - 1.0 * pma0_val))  # sw[20]=1

    pma1_val = _glob7s(PMA[1], *glob7s_args)
    meso_tn2 = meso_tn2.at[2].set(PMA[1, 0] * PAVGM[1] / (1.0 - 1.0 * pma1_val))  # sw[20]=1

    pma2_val = _glob7s(PMA[2], *glob7s_args)
    meso_tn2 = meso_tn2.at[3].set(
        PMA[2, 0] * PAVGM[2] / (1.0 - 1.0 * 1.0 * pma2_val)  # sw[20]*sw[22]=1
    )

    pma9_val = _glob7s(PMA[9], *glob7s_args)
    meso_tgn2 = meso_tgn2.at[1].set(
        PAVGM[8]
        * PMA[9, 0]
        * (1.0 + 1.0 * 1.0 * pma9_val)  # sw[20]*sw[22]=1
        * meso_tn2[3]
        * meso_tn2[3]
        / (PMA[2, 0] * PAVGM[2]) ** 2
    )

    # Troposphere temperature profile (zn3 layer)
    meso_tn3 = jnp.zeros(5)
    meso_tgn3 = jnp.zeros(2)

    meso_tn3 = meso_tn3.at[0].set(meso_tn2[3])

    # Compute zn3 layer temps only when alt < zn3[0] (32.5 km)
    meso_tgn3 = meso_tgn3.at[0].set(meso_tgn2[1])

    pma3_val = _glob7s(PMA[3], *glob7s_args)
    meso_tn3 = meso_tn3.at[1].set(PMA[3, 0] * PAVGM[3] / (1.0 - 1.0 * pma3_val))  # sw[22]=1

    pma4_val = _glob7s(PMA[4], *glob7s_args)
    meso_tn3 = meso_tn3.at[2].set(PMA[4, 0] * PAVGM[4] / (1.0 - 1.0 * pma4_val))  # sw[22]=1

    pma5_val = _glob7s(PMA[5], *glob7s_args)
    meso_tn3 = meso_tn3.at[3].set(PMA[5, 0] * PAVGM[5] / (1.0 - 1.0 * pma5_val))  # sw[22]=1

    pma6_val = _glob7s(PMA[6], *glob7s_args)
    meso_tn3 = meso_tn3.at[4].set(PMA[6, 0] * PAVGM[6] / (1.0 - 1.0 * pma6_val))  # sw[22]=1

    pma7_val = _glob7s(PMA[7], *glob7s_args)
    meso_tgn3 = meso_tgn3.at[1].set(
        PMA[7, 0]
        * PAVGM[7]
        * (1.0 + 1.0 * pma7_val)  # sw[22]=1
        * meso_tn3[4]
        * meso_tn3[4]
        / (PMA[6, 0] * PAVGM[6]) ** 2
    )

    # Linear transition to full mixing below zn2[0]
    dmc = jnp.where(alt > zmix, 1.0 - (zn2[0] - alt) / (zn2[0] - zmix), 0.0)
    dz28 = d_therm[2]

    # N2 density
    dmr_n2 = d_therm[2] / dm28m - 1.0
    tz_low, d3_low = _densm(
        alt,
        dm28m,
        xmm,
        0.0,
        mn3,
        zn3,
        meso_tn3,
        meso_tgn3,
        mn2,
        zn2,
        meso_tn2,
        meso_tgn2,
        gsurf,
        re,
    )
    d_n2 = d3_low * (1.0 + dmr_n2 * dmc)

    # He density
    dmr_he = d_therm[0] / (dz28 * PDM[0, 1]) - 1.0
    d_he = d_n2 * PDM[0, 1] * (1.0 + dmr_he * dmc)

    # O density = 0 below zn2
    # O2 density
    dmr_o2 = d_therm[3] / (dz28 * PDM[3, 1]) - 1.0
    d_o2 = d_n2 * PDM[3, 1] * (1.0 + dmr_o2 * dmc)

    # Ar density
    dmr_ar = d_therm[4] / (dz28 * PDM[4, 1]) - 1.0
    d_ar = d_n2 * PDM[4, 1] * (1.0 + dmr_ar * dmc)

    # Total mass density
    d5_low = 1.66e-24 * (
        4.0 * d_he + 16.0 * 0.0 + 28.0 * d_n2 + 32.0 * d_o2 + 40.0 * d_ar + 1.0 * 0.0 + 14.0 * 0.0
    )
    d5_low = d5_low / 1000.0  # g/cm^3 to kg/m^3 (switches[0]=1)

    # Temperature at altitude (below zn2)
    tz_alt, _ = _densm(
        alt,
        1.0,
        0.0,
        tz_low,
        mn3,
        zn3,
        meso_tn3,
        meso_tgn3,
        mn2,
        zn2,
        meso_tn2,
        meso_tgn2,
        gsurf,
        re,
    )

    # Choose between thermospheric and lower atmosphere values
    above_zn2 = alt >= zn2[0]

    d_out = jnp.zeros(9)
    d_out = d_out.at[0].set(jnp.where(above_zn2, d_therm[0], d_he))
    d_out = d_out.at[1].set(jnp.where(above_zn2, d_therm[1], 0.0))
    d_out = d_out.at[2].set(jnp.where(above_zn2, d_therm[2], d_n2))
    d_out = d_out.at[3].set(jnp.where(above_zn2, d_therm[3], d_o2))
    d_out = d_out.at[4].set(jnp.where(above_zn2, d_therm[4], d_ar))
    d_out = d_out.at[5].set(jnp.where(above_zn2, d_therm[5], d5_low))
    d_out = d_out.at[6].set(jnp.where(above_zn2, d_therm[6], 0.0))
    d_out = d_out.at[7].set(jnp.where(above_zn2, d_therm[7], 0.0))
    d_out = d_out.at[8].set(jnp.where(above_zn2, d_therm[8], 0.0))

    t_out = jnp.zeros(2)
    t_out = t_out.at[0].set(t_therm[0])
    t_out = t_out.at[1].set(jnp.where(above_zn2, t_therm[1], tz_alt))

    return t_out, d_out


def gtd7d(
    doy: ArrayLike,
    sec: ArrayLike,
    alt: ArrayLike,
    g_lat: ArrayLike,
    g_lon: ArrayLike,
    lst: ArrayLike,
    f107a: ArrayLike,
    f107: ArrayLike,
    ap: ArrayLike,
    ap_array: Array,
    use_ap_array: bool = True,
) -> tuple[Array, Array]:
    """NRLMSISE-00 driver with anomalous oxygen in total density.

    Same as :func:`gtd7` but recalculates total mass density ``d[5]``
    to include the contribution from anomalous oxygen ``d[8]``, which
    is important above ~500 km.

    Args:
        doy: Day of year (1-366).
        sec: Seconds in day (UT).
        alt: Altitude [km].
        g_lat: Geodetic latitude [degrees].
        g_lon: Geodetic longitude [degrees].
        lst: Local apparent solar time [hours].
        f107a: 81-day average F10.7 flux.
        f107: Daily F10.7 flux.
        ap: Daily Ap magnetic index.
        ap_array: 7-element Ap array for NRLMSISE-00.

    Returns:
        Tuple of (temperatures [2], densities [9]).
        See :func:`gtd7` for field descriptions.
    """
    t_out, d_out = gtd7(doy, sec, alt, g_lat, g_lon, lst, f107a, f107, ap, ap_array, use_ap_array)

    # Recalculate total mass density including anomalous oxygen
    d5 = 1.66e-24 * (
        4.0 * d_out[0]
        + 16.0 * d_out[1]
        + 28.0 * d_out[2]
        + 32.0 * d_out[3]
        + 40.0 * d_out[4]
        + 1.0 * d_out[6]
        + 14.0 * d_out[7]
        + 16.0 * d_out[8]
    )
    d5 = d5 / 1000.0  # kg/m^3 (switches[0]=1)
    d_out = d_out.at[5].set(d5)

    return t_out, d_out


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def density_nrlmsise00(
    sw: SpaceWeatherData,
    epc: Epoch,
    r_ecef: ArrayLike,
) -> Array:
    """Atmospheric density [kg/m^3] from ECEF position using NRLMSISE-00.

    Functional interface: space weather data is passed as an argument
    (not global state), enabling use in Monte Carlo simulations where
    each sample may use different space weather conditions.

    Uses ``gtd7d`` which includes the contribution of anomalous oxygen
    in total density (important above ~500 km).

    Args:
        sw: Space weather data (loaded via
            :func:`~astrojax.space_weather.load_default_sw` or similar).
        epc: Epoch of computation.
        r_ecef: Satellite position in ECEF frame [m], shape ``(3,)``.

    Returns:
        Atmospheric density [kg/m^3] (scalar).

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.space_weather import load_default_sw
        from astrojax.orbit_dynamics.nrlmsise00 import density_nrlmsise00
        from astrojax.epoch import Epoch

        sw = load_default_sw()
        epc = Epoch.from_datetime(2020, 6, 1, 12, 0, 0.0)
        r_ecef = jnp.array([6778137.0, 0.0, 0.0])
        rho = density_nrlmsise00(sw, epc, r_ecef)
        ```
    """
    r_ecef = jnp.asarray(r_ecef, dtype=get_dtype())
    geod = position_ecef_to_geodetic(r_ecef, use_degrees=True)
    return density_nrlmsise00_geod(sw, epc, geod)


def density_nrlmsise00_geod(
    sw: SpaceWeatherData,
    epc: Epoch,
    geod: ArrayLike,
) -> Array:
    """Atmospheric density [kg/m^3] from geodetic coordinates using NRLMSISE-00.

    Functional interface: space weather data is passed as an argument.

    Args:
        sw: Space weather data.
        epc: Epoch of computation.
        geod: Geodetic position as ``[lon_deg, lat_deg, alt_m]`` where
            longitude and latitude are in degrees and altitude is in metres.

    Returns:
        Atmospheric density [kg/m^3] (scalar).

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.space_weather import load_default_sw
        from astrojax.orbit_dynamics.nrlmsise00 import density_nrlmsise00_geod
        from astrojax.epoch import Epoch

        sw = load_default_sw()
        epc = Epoch.from_datetime(2020, 6, 1, 12, 0, 0.0)
        geod = jnp.array([-74.0, 40.7, 400e3])
        rho = density_nrlmsise00_geod(sw, epc, geod)
        ```
    """
    geod = jnp.asarray(geod, dtype=get_dtype())

    lon_deg = geod[0]
    lat_deg = geod[1]
    alt_km = geod[2] / 1000.0  # m to km

    # Extract time components from epoch
    mjd = epc.mjd()
    year, month, day, hour, minute, second = mjd_to_caldate(mjd)

    # Day of year
    jan1_mjd = caldate_to_mjd(year, 1, 1)
    doy = jnp.floor(mjd - jan1_mjd).astype(jnp.int32) + 1

    # Seconds of day
    sec_of_day = hour * 3600.0 + minute * 60.0 + second

    # Local solar time
    lst = sec_of_day / 3600.0 + lon_deg / 15.0

    # Space weather lookups
    f107_val = get_sw_f107_obs(sw, mjd)
    f107a_val = get_sw_f107_obs_ctr81(sw, mjd)
    ap_arr = get_sw_ap_array(sw, mjd)
    ap_daily = ap_arr[0]

    # Run NRLMSISE-00
    t_out, d_out = gtd7d(
        doy, sec_of_day, alt_km, lat_deg, lon_deg, lst, f107a_val, f107_val, ap_daily, ap_arr
    )

    # Return total mass density (d[5] in kg/m^3)
    return d_out[5]
