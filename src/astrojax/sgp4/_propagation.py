"""
SGP4/SDP4 propagation core translated to JAX.

This module provides the initialization and propagation routines for the
SGP4 (near-Earth) and SDP4 (deep-space) analytical orbit propagators.
The implementation is a direct translation of the reference ``sgp4`` Python
library into JAX-compatible code, supporting JIT compilation and ``vmap``.

The satellite parameters are stored in a flat ``jnp.array`` with named
indices defined in ``_IDX``. The init function (``sgp4_init``) runs at
Python time and returns this array plus a method flag. The propagation
function (``sgp4_propagate``) is a pure JAX function suitable for JIT.
"""

from __future__ import annotations

from collections.abc import Callable
from math import cos as _py_cos
from math import fabs as _py_fabs
from math import pi as _py_pi
from math import sin as _py_sin
from math import sqrt as _py_sqrt
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.sgp4._constants import GRAVITY_MODELS, WGS72, EarthGravity
from astrojax.sgp4._tle import parse_omm, parse_tle
from astrojax.sgp4._types import SGP4Elements, elements_to_array

if TYPE_CHECKING:
    from astrojax._gp_record import GPRecord

# ---------------------------------------------------------------------------
# Parameter index layout for the flat params array
# ---------------------------------------------------------------------------
_PARAM_NAMES = [
    # Gravity constants (0-7)
    "radiusearthkm",
    "xke",
    "j2",
    "j3oj2",
    "j4",
    "tumin",
    "mu",
    "j3",
    # TLE-derived orbital elements (8-16)
    "bstar",
    "ecco",
    "argpo",
    "inclo",
    "mo",
    "no_kozai",
    "nodeo",
    "ndot",
    "nddot",
    # Computed by _initl (17-20)
    "no_unkozai",
    "con41",
    "gsto",
    "a",
    # Computed by sgp4init — near-earth (21-45)
    "cc1",
    "cc4",
    "cc5",
    "d2",
    "d3",
    "d4",
    "delmo",
    "eta",
    "argpdot",
    "omgcof",
    "sinmao",
    "t2cof",
    "t3cof",
    "t4cof",
    "t5cof",
    "x1mth2",
    "x7thm1",
    "mdot",
    "nodedot",
    "xlcof",
    "xmcof",
    "nodecf",
    "aycof",
    "isimp",  # 0.0 or 1.0
    # Deep-space parameters (45+)
    "irez",
    "d2201",
    "d2211",
    "d3210",
    "d3222",
    "d4410",
    "d4422",
    "d5220",
    "d5232",
    "d5421",
    "d5433",
    "dedt",
    "del1",
    "del2",
    "del3",
    "didt",
    "dmdt",
    "dnodt",
    "domdt",
    "e3",
    "ee2",
    "peo",
    "pgho",
    "pho",
    "pinco",
    "plo",
    "se2",
    "se3",
    "sgh2",
    "sgh3",
    "sgh4",
    "sh2",
    "sh3",
    "si2",
    "si3",
    "sl2",
    "sl3",
    "sl4",
    "xfact",
    "xgh2",
    "xgh3",
    "xgh4",
    "xh2",
    "xh3",
    "xi2",
    "xi3",
    "xl2",
    "xl3",
    "xl4",
    "xlamo",
    "xli",
    "xni",
    "zmol",
    "zmos",
    "atime",
    # Method flag (96): 0.0 = near-earth, 1.0 = deep-space
    "method",
]

_IDX = {name: i for i, name in enumerate(_PARAM_NAMES)}
_NUM_PARAMS = len(_PARAM_NAMES)

# Shorthand index constants for frequently accessed parameters
_I = _IDX  # alias for brevity in propagation code

# ---------------------------------------------------------------------------
# Python-time helpers (used during init, not under JIT)
# ---------------------------------------------------------------------------
_twopi = 2.0 * _py_pi


def _gstime(jdut1: float) -> float:
    """Compute Greenwich Sidereal Time from Julian date (Python floats)."""
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = (
        -6.2e-6 * tut1 * tut1 * tut1
        + 0.093104 * tut1 * tut1
        + (876600.0 * 3600 + 8640184.812866) * tut1
        + 67310.54841
    )
    temp = (temp * (_py_pi / 180.0) / 240.0) % _twopi
    if temp < 0.0:
        temp += _twopi
    return temp


def _initl(
    xke: float,
    j2: float,
    ecco: float,
    epoch: float,
    inclo: float,
    no: float,
    opsmode: str,
) -> tuple:
    """Initialize SGP4 auxiliary quantities (Python floats).

    Args:
        xke: Gravity constant xke.
        j2: J2 zonal harmonic.
        ecco: Eccentricity.
        epoch: Days since 1950 Jan 0.
        inclo: Inclination [rad].
        no: Mean motion (Kozai) [rad/min].
        opsmode: Operation mode ('a' or 'i').

    Returns:
        Tuple of computed quantities.
    """
    x2o3 = 2.0 / 3.0

    eccsq = ecco * ecco
    omeosq = 1.0 - eccsq
    rteosq = _py_sqrt(omeosq)
    cosio = _py_cos(inclo)
    cosio2 = cosio * cosio

    # Un-Kozai the mean motion
    ak = (xke / no) ** x2o3
    d1 = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)
    del_ = d1 / (ak * ak)
    adel = ak * (1.0 - del_ * del_ - del_ * (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0))
    del_ = d1 / (adel * adel)
    no = no / (1.0 + del_)

    ao = (xke / no) ** x2o3
    sinio = _py_sin(inclo)
    po = ao * omeosq
    con42 = 1.0 - 5.0 * cosio2
    con41 = -con42 - cosio2 - cosio2
    posq = po * po
    rp = ao * (1.0 - ecco)

    # Sidereal time
    if opsmode == "a":
        ts70 = epoch - 7305.0
        ds70 = (ts70 + 1.0e-8) // 1.0
        tfrac = ts70 - ds70
        c1 = 1.72027916940703639e-2
        thgr70 = 1.7321343856509374
        fk5r = 5.07551419432269442e-15
        c1p2p = c1 + _twopi
        gsto = (thgr70 + c1 * ds70 + c1p2p * tfrac + ts70 * ts70 * fk5r) % _twopi
        if gsto < 0.0:
            gsto = gsto + _twopi
    else:
        gsto = _gstime(epoch + 2433281.5)

    return (
        no,
        1.0 / ao,  # ainv
        ao,
        con41,
        con42,
        cosio,
        cosio2,
        eccsq,
        omeosq,
        posq,
        rp,
        rteosq,
        sinio,
        gsto,
    )


# ---------------------------------------------------------------------------
# JAX-compatible init helpers (JIT-safe)
# ---------------------------------------------------------------------------

_twopi_jax = 2.0 * jnp.pi


def _gstime_jax(jdut1: ArrayLike) -> Array:
    """Compute Greenwich Sidereal Time from Julian date (JAX).

    Args:
        jdut1: Julian date (UT1).

    Returns:
        Greenwich sidereal time [rad].
    """
    tut1 = (jdut1 - 2451545.0) / 36525.0
    temp = (
        -6.2e-6 * tut1 * tut1 * tut1
        + 0.093104 * tut1 * tut1
        + (876600.0 * 3600 + 8640184.812866) * tut1
        + 67310.54841
    )
    temp = (temp * (jnp.pi / 180.0) / 240.0) % _twopi_jax
    temp = jnp.where(temp < 0.0, temp + _twopi_jax, temp)
    return temp


def _initl_jax(
    xke: ArrayLike,
    j2: ArrayLike,
    ecco: ArrayLike,
    epoch: ArrayLike,
    inclo: ArrayLike,
    no: ArrayLike,
    opsmode: str,
) -> tuple:
    """Initialize SGP4 auxiliary quantities (JAX, JIT-compatible).

    Args:
        xke: Gravity constant xke.
        j2: J2 zonal harmonic.
        ecco: Eccentricity.
        epoch: Days since 1950 Jan 0.
        inclo: Inclination [rad].
        no: Mean motion (Kozai) [rad/min].
        opsmode: Operation mode ('a' or 'i'). Resolved at trace time.

    Returns:
        Tuple of computed quantities.
    """
    x2o3 = 2.0 / 3.0

    eccsq = ecco * ecco
    omeosq = 1.0 - eccsq
    rteosq = jnp.sqrt(omeosq)
    cosio = jnp.cos(inclo)
    cosio2 = cosio * cosio

    # Un-Kozai the mean motion
    ak = (xke / no) ** x2o3
    d1 = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)
    del_ = d1 / (ak * ak)
    adel = ak * (1.0 - del_ * del_ - del_ * (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0))
    del_ = d1 / (adel * adel)
    no = no / (1.0 + del_)

    ao = (xke / no) ** x2o3
    sinio = jnp.sin(inclo)
    po = ao * omeosq
    con42 = 1.0 - 5.0 * cosio2
    con41 = -con42 - cosio2 - cosio2
    posq = po * po
    rp = ao * (1.0 - ecco)

    # Sidereal time
    if opsmode == "a":
        ts70 = epoch - 7305.0
        ds70 = jnp.floor(ts70 + 1.0e-8)
        tfrac = ts70 - ds70
        c1 = 1.72027916940703639e-2
        thgr70 = 1.7321343856509374
        fk5r = 5.07551419432269442e-15
        c1p2p = c1 + _twopi_jax
        gsto = (thgr70 + c1 * ds70 + c1p2p * tfrac + ts70 * ts70 * fk5r) % _twopi_jax
        gsto = jnp.where(gsto < 0.0, gsto + _twopi_jax, gsto)
    else:
        gsto = _gstime_jax(epoch + 2433281.5)

    return (
        no,
        1.0 / ao,  # ainv
        ao,
        con41,
        con42,
        cosio,
        cosio2,
        eccsq,
        omeosq,
        posq,
        rp,
        rteosq,
        sinio,
        gsto,
    )


# ---------------------------------------------------------------------------
# SGP4 Initialization (Python-time)
# ---------------------------------------------------------------------------


def sgp4_init(
    elements: SGP4Elements,
    gravity: EarthGravity = WGS72,
    opsmode: str = "i",
) -> tuple[Array, str]:
    """Initialize SGP4 satellite parameters from parsed TLE elements.

    This function runs at Python time (not under JIT). It computes all the
    intermediate parameters needed by the SGP4 propagator and packs them
    into a flat JAX array.

    Args:
        elements: Parsed TLE elements from ``parse_tle`` or ``parse_omm``.
        gravity: Earth gravity model constants.
        opsmode: Operation mode ('i' for improved, 'a' for AFSPC).

    Returns:
        Tuple of ``(params, method)`` where ``params`` is a flat
        ``jnp.array`` of shape ``(_NUM_PARAMS,)`` and ``method`` is
        ``'n'`` (near-Earth) or ``'d'`` (deep-space).
    """
    # Initialize parameter dict with zeros
    d: dict[str, float] = {name: 0.0 for name in _PARAM_NAMES}

    # Store gravity constants
    d["tumin"] = gravity.tumin
    d["mu"] = gravity.mu
    d["radiusearthkm"] = gravity.radiusearthkm
    d["xke"] = gravity.xke
    d["j2"] = gravity.j2
    d["j3"] = gravity.j3
    d["j4"] = gravity.j4
    d["j3oj2"] = gravity.j3oj2

    # Store TLE elements
    d["bstar"] = elements.bstar
    d["ecco"] = elements.ecco
    d["argpo"] = elements.argpo
    d["inclo"] = elements.inclo
    d["mo"] = elements.mo
    d["no_kozai"] = elements.no_kozai
    d["nodeo"] = elements.nodeo
    d["ndot"] = elements.ndot
    d["nddot"] = elements.nddot

    # Compute epoch in days since 1950 Jan 0
    epoch = elements.jdsatepoch + elements.jdsatepochF - 2433281.5

    temp4 = 1.5e-12
    x2o3 = 2.0 / 3.0
    method = "n"

    # Earth constants
    ss = 78.0 / gravity.radiusearthkm + 1.0
    qzms2ttemp = (120.0 - 78.0) / gravity.radiusearthkm
    qzms2t = qzms2ttemp**4
    sfour = ss

    # Initialize auxiliary quantities
    (
        no_unkozai,
        ainv,
        ao,
        con41,
        con42,
        cosio,
        cosio2,
        eccsq,
        omeosq,
        posq,
        rp,
        rteosq,
        sinio,
        gsto,
    ) = _initl(
        gravity.xke,
        gravity.j2,
        elements.ecco,
        epoch,
        elements.inclo,
        elements.no_kozai,
        opsmode,
    )

    d["no_unkozai"] = no_unkozai
    d["con41"] = con41
    d["gsto"] = gsto
    d["a"] = (no_unkozai * gravity.tumin) ** (-2.0 / 3.0)

    if omeosq >= 0.0 or no_unkozai >= 0.0:
        isimp = 0
        if rp < 220.0 / gravity.radiusearthkm + 1.0:
            isimp = 1

        qzms24 = qzms2t
        perige = (rp - 1.0) * gravity.radiusearthkm

        # For perigees below 156 km, s and qoms2t are altered
        if perige < 156.0:
            sfour = perige - 78.0
            if perige < 98.0:
                sfour = 20.0
            qzms24temp = (120.0 - sfour) / gravity.radiusearthkm
            qzms24 = qzms24temp**4
            sfour = sfour / gravity.radiusearthkm + 1.0

        pinvsq = 1.0 / posq
        tsi = 1.0 / (ao - sfour)
        eta = ao * elements.ecco * tsi
        etasq = eta * eta
        eeta = elements.ecco * eta
        psisq = _py_fabs(1.0 - etasq)
        coef = qzms24 * tsi**4
        coef1 = coef / psisq**3.5
        cc2 = (
            coef1
            * no_unkozai
            * (
                ao * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq))
                + 0.375 * gravity.j2 * tsi / psisq * con41 * (8.0 + 3.0 * etasq * (8.0 + etasq))
            )
        )
        cc1 = elements.bstar * cc2
        cc3 = 0.0
        if elements.ecco > 1.0e-4:
            cc3 = -2.0 * coef * tsi * gravity.j3oj2 * no_unkozai * sinio / elements.ecco
        x1mth2 = 1.0 - cosio2
        cc4 = (
            2.0
            * no_unkozai
            * coef1
            * ao
            * omeosq
            * (
                eta * (2.0 + 0.5 * etasq)
                + elements.ecco * (0.5 + 2.0 * etasq)
                - gravity.j2
                * tsi
                / (ao * psisq)
                * (
                    -3.0 * con41 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta))
                    + 0.75
                    * x1mth2
                    * (2.0 * etasq - eeta * (1.0 + etasq))
                    * _py_cos(2.0 * elements.argpo)
                )
            )
        )
        cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq)
        cosio4 = cosio2 * cosio2
        temp1 = 1.5 * gravity.j2 * pinvsq * no_unkozai
        temp2 = 0.5 * temp1 * gravity.j2 * pinvsq
        temp3 = -0.46875 * gravity.j4 * pinvsq * pinvsq * no_unkozai
        mdot = (
            no_unkozai
            + 0.5 * temp1 * rteosq * con41
            + 0.0625 * temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4)
        )
        argpdot = (
            -0.5 * temp1 * con42
            + 0.0625 * temp2 * (7.0 - 114.0 * cosio2 + 395.0 * cosio4)
            + temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4)
        )
        xhdot1 = -temp1 * cosio
        nodedot = (
            xhdot1
            + (0.5 * temp2 * (4.0 - 19.0 * cosio2) + 2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio
        )
        xpidot = argpdot + nodedot
        omgcof = elements.bstar * cc3 * _py_cos(elements.argpo)
        xmcof = 0.0
        if elements.ecco > 1.0e-4:
            xmcof = -x2o3 * coef * elements.bstar / eeta
        nodecf = 3.5 * omeosq * xhdot1 * cc1
        t2cof = 1.5 * cc1

        # sgp4fix for divide by zero with xinco = 180 deg
        if _py_fabs(cosio + 1.0) > 1.5e-12:
            xlcof = -0.25 * gravity.j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio)
        else:
            xlcof = -0.25 * gravity.j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4

        aycof = -0.5 * gravity.j3oj2 * sinio
        delmotemp = 1.0 + eta * _py_cos(elements.mo)
        delmo = delmotemp * delmotemp * delmotemp
        sinmao = _py_sin(elements.mo)
        x7thm1 = 7.0 * cosio2 - 1.0

        # Check for deep space (period >= 225 min)
        if _twopi / no_unkozai >= 225.0:
            method = "d"
            isimp = 1
            d["method"] = 1.0

        # Store near-earth parameters
        d["cc1"] = cc1
        d["cc4"] = cc4
        d["cc5"] = cc5
        d["d2"] = 0.0
        d["d3"] = 0.0
        d["d4"] = 0.0
        d["delmo"] = delmo
        d["eta"] = eta
        d["argpdot"] = argpdot
        d["omgcof"] = omgcof
        d["sinmao"] = sinmao
        d["t2cof"] = t2cof
        d["t3cof"] = 0.0
        d["t4cof"] = 0.0
        d["t5cof"] = 0.0
        d["x1mth2"] = x1mth2
        d["x7thm1"] = x7thm1
        d["mdot"] = mdot
        d["nodedot"] = nodedot
        d["xlcof"] = xlcof
        d["xmcof"] = xmcof
        d["nodecf"] = nodecf
        d["aycof"] = aycof
        d["isimp"] = float(isimp)

        # Deep-space initialization
        if method == "d":
            from astrojax.sgp4._deep_space import deep_space_init

            deep_space_init(
                d,
                elements,
                gravity,
                epoch,
                opsmode,
                no_unkozai,
                ao,
                cosio,
                cosio2,
                eccsq,
                omeosq,
                sinio,
                rteosq,
                con41,
                con42,
                cosio4,
                posq,
                xpidot,
                xhdot1,
                sfour,
                tsi,
                eta,
                etasq,
                eeta,
                psisq,
                cc1,
                cc3,
                coef,
                coef1,
                temp1,
                temp2,
                temp3,
                pinvsq,
            )

        # Higher-order secular terms for non-simplified orbits
        if isimp != 1:
            cc1sq = cc1 * cc1
            d["d2"] = 4.0 * ao * tsi * cc1sq
            temp = d["d2"] * tsi * cc1 / 3.0
            d["d3"] = (17.0 * ao + sfour) * temp
            d["d4"] = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * cc1
            d["t3cof"] = d["d2"] + 2.0 * cc1sq
            d["t4cof"] = 0.25 * (3.0 * d["d3"] + cc1 * (12.0 * d["d2"] + 10.0 * cc1sq))
            d["t5cof"] = 0.2 * (
                3.0 * d["d4"]
                + 12.0 * cc1 * d["d3"]
                + 6.0 * d["d2"] * d["d2"]
                + 15.0 * cc1sq * (2.0 * d["d2"] + cc1sq)
            )

    # Pack into flat JAX array
    params = jnp.array([d[name] for name in _PARAM_NAMES])

    # Run propagation at t=0 to finalize (matches reference behavior)
    # The reference calls sgp4(satrec, 0.0) here; we do the same
    # but discard the result — it's just to validate init is consistent
    if method == "n":
        _sgp4_propagate_near_earth(params, jnp.float64(0.0))
    else:
        _sgp4_propagate_deep_space(params, jnp.float64(0.0))

    return params, method


# ---------------------------------------------------------------------------
# JIT-compilable SGP4 Initialization
# ---------------------------------------------------------------------------


def sgp4_init_jax(
    elements: Array,
    gravity: EarthGravity = WGS72,
    opsmode: str = "i",
) -> Array:
    """Initialize SGP4 satellite parameters (JAX, JIT-compatible).

    Unlike ``sgp4_init``, this function is compatible with ``jax.jit``,
    ``jax.vmap``, and ``jax.grad``. It accepts a flat array of numeric
    orbital elements and returns a flat parameter array with an embedded
    method flag (0.0 = near-earth, 1.0 = deep-space).

    Use with ``jax.jit``::

        init_jit = jax.jit(sgp4_init_jax, static_argnames=('gravity', 'opsmode'))
        params = init_jit(elements_arr)

    Args:
        elements: Array of shape ``(11,)`` containing the numeric orbital
            elements. Use ``elements_to_array`` to convert from
            ``SGP4Elements``. Order: jdsatepoch, jdsatepochF, no_kozai,
            ecco, inclo, nodeo, argpo, mo, bstar, ndot, nddot.
        gravity: Earth gravity model constants. Treated as a static
            argument under JIT (use ``static_argnames``).
        opsmode: Operation mode ('i' for improved, 'a' for AFSPC).
            Treated as a static argument under JIT.

    Returns:
        Flat ``jnp.array`` of shape ``(_NUM_PARAMS,)`` with all SGP4
        parameters, including a method flag at index ``_IDX["method"]``.
    """
    from astrojax.sgp4._deep_space import _deep_space_init_jax

    # Unpack elements
    jdsatepoch = elements[0]
    jdsatepochF = elements[1]
    no_kozai = elements[2]
    ecco = elements[3]
    inclo = elements[4]
    nodeo = elements[5]
    argpo = elements[6]
    mo = elements[7]
    bstar = elements[8]
    ndot = elements[9]
    nddot = elements[10]

    # Initialize params array
    params = jnp.zeros(_NUM_PARAMS)

    # Store gravity constants (static — known at trace time)
    params = params.at[_IDX["tumin"]].set(gravity.tumin)
    params = params.at[_IDX["mu"]].set(gravity.mu)
    params = params.at[_IDX["radiusearthkm"]].set(gravity.radiusearthkm)
    params = params.at[_IDX["xke"]].set(gravity.xke)
    params = params.at[_IDX["j2"]].set(gravity.j2)
    params = params.at[_IDX["j3"]].set(gravity.j3)
    params = params.at[_IDX["j4"]].set(gravity.j4)
    params = params.at[_IDX["j3oj2"]].set(gravity.j3oj2)

    # Store TLE elements
    params = params.at[_IDX["bstar"]].set(bstar)
    params = params.at[_IDX["ecco"]].set(ecco)
    params = params.at[_IDX["argpo"]].set(argpo)
    params = params.at[_IDX["inclo"]].set(inclo)
    params = params.at[_IDX["mo"]].set(mo)
    params = params.at[_IDX["no_kozai"]].set(no_kozai)
    params = params.at[_IDX["nodeo"]].set(nodeo)
    params = params.at[_IDX["ndot"]].set(ndot)
    params = params.at[_IDX["nddot"]].set(nddot)

    # Compute epoch in days since 1950 Jan 0
    epoch = jdsatepoch + jdsatepochF - 2433281.5

    temp4 = 1.5e-12
    x2o3 = 2.0 / 3.0
    twopi = 2.0 * jnp.pi

    # Earth constants
    ss = 78.0 / gravity.radiusearthkm + 1.0
    qzms2ttemp = (120.0 - 78.0) / gravity.radiusearthkm
    qzms2t = qzms2ttemp**4
    sfour_base = ss

    # Initialize auxiliary quantities
    (
        no_unkozai, ainv, ao, con41, con42, cosio, cosio2,
        eccsq, omeosq, posq, rp, rteosq, sinio, gsto,
    ) = _initl_jax(gravity.xke, gravity.j2, ecco, epoch, inclo, no_kozai, opsmode)

    params = params.at[_IDX["no_unkozai"]].set(no_unkozai)
    params = params.at[_IDX["con41"]].set(con41)
    params = params.at[_IDX["gsto"]].set(gsto)
    params = params.at[_IDX["a"]].set((no_unkozai * gravity.tumin) ** (-2.0 / 3.0))

    # Perigee-based s/qoms2t adjustments
    isimp = jnp.where(rp < 220.0 / gravity.radiusearthkm + 1.0, 1.0, 0.0)

    qzms24 = qzms2t
    perige = (rp - 1.0) * gravity.radiusearthkm

    # For perigees below 156 km
    sfour_low = jnp.where(perige < 98.0, 20.0, perige - 78.0)
    qzms24temp_low = (120.0 - sfour_low) / gravity.radiusearthkm
    qzms24_low = qzms24temp_low**4
    sfour_low_adj = sfour_low / gravity.radiusearthkm + 1.0

    is_low_perige = perige < 156.0
    sfour = jnp.where(is_low_perige, sfour_low_adj, sfour_base)
    qzms24 = jnp.where(is_low_perige, qzms24_low, qzms24)

    pinvsq = 1.0 / posq
    tsi = 1.0 / (ao - sfour)
    eta = ao * ecco * tsi
    etasq = eta * eta
    eeta = ecco * eta
    psisq = jnp.abs(1.0 - etasq)
    coef = qzms24 * tsi**4
    coef1 = coef / psisq**3.5
    cc2 = (
        coef1
        * no_unkozai
        * (
            ao * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq))
            + 0.375 * gravity.j2 * tsi / psisq * con41 * (8.0 + 3.0 * etasq * (8.0 + etasq))
        )
    )
    cc1 = bstar * cc2
    cc3 = jnp.where(
        ecco > 1.0e-4,
        -2.0 * coef * tsi * gravity.j3oj2 * no_unkozai * sinio / ecco,
        0.0,
    )
    x1mth2 = 1.0 - cosio2
    cc4 = (
        2.0
        * no_unkozai
        * coef1
        * ao
        * omeosq
        * (
            eta * (2.0 + 0.5 * etasq)
            + ecco * (0.5 + 2.0 * etasq)
            - gravity.j2
            * tsi
            / (ao * psisq)
            * (
                -3.0 * con41 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta))
                + 0.75
                * x1mth2
                * (2.0 * etasq - eeta * (1.0 + etasq))
                * jnp.cos(2.0 * argpo)
            )
        )
    )
    cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq)
    cosio4 = cosio2 * cosio2
    temp1 = 1.5 * gravity.j2 * pinvsq * no_unkozai
    temp2 = 0.5 * temp1 * gravity.j2 * pinvsq
    temp3 = -0.46875 * gravity.j4 * pinvsq * pinvsq * no_unkozai
    mdot = (
        no_unkozai
        + 0.5 * temp1 * rteosq * con41
        + 0.0625 * temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4)
    )
    argpdot = (
        -0.5 * temp1 * con42
        + 0.0625 * temp2 * (7.0 - 114.0 * cosio2 + 395.0 * cosio4)
        + temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4)
    )
    xhdot1 = -temp1 * cosio
    nodedot = (
        xhdot1
        + (0.5 * temp2 * (4.0 - 19.0 * cosio2) + 2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio
    )
    xpidot = argpdot + nodedot
    omgcof = bstar * cc3 * jnp.cos(argpo)
    xmcof = jnp.where(ecco > 1.0e-4, -x2o3 * coef * bstar / eeta, 0.0)
    nodecf = 3.5 * omeosq * xhdot1 * cc1
    t2cof = 1.5 * cc1

    # Cosio singularity fix
    xlcof = jnp.where(
        jnp.abs(cosio + 1.0) > 1.5e-12,
        -0.25 * gravity.j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio),
        -0.25 * gravity.j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4,
    )

    aycof = -0.5 * gravity.j3oj2 * sinio
    delmotemp = 1.0 + eta * jnp.cos(mo)
    delmo = delmotemp**3
    sinmao = jnp.sin(mo)
    x7thm1 = 7.0 * cosio2 - 1.0

    # Deep-space detection
    is_deep_space = twopi / no_unkozai >= 225.0
    isimp = jnp.where(is_deep_space, 1.0, isimp)

    # Store near-earth parameters
    params = params.at[_IDX["cc1"]].set(cc1)
    params = params.at[_IDX["cc4"]].set(cc4)
    params = params.at[_IDX["cc5"]].set(cc5)
    params = params.at[_IDX["delmo"]].set(delmo)
    params = params.at[_IDX["eta"]].set(eta)
    params = params.at[_IDX["argpdot"]].set(argpdot)
    params = params.at[_IDX["omgcof"]].set(omgcof)
    params = params.at[_IDX["sinmao"]].set(sinmao)
    params = params.at[_IDX["t2cof"]].set(t2cof)
    params = params.at[_IDX["x1mth2"]].set(x1mth2)
    params = params.at[_IDX["x7thm1"]].set(x7thm1)
    params = params.at[_IDX["mdot"]].set(mdot)
    params = params.at[_IDX["nodedot"]].set(nodedot)
    params = params.at[_IDX["xlcof"]].set(xlcof)
    params = params.at[_IDX["xmcof"]].set(xmcof)
    params = params.at[_IDX["nodecf"]].set(nodecf)
    params = params.at[_IDX["aycof"]].set(aycof)
    params = params.at[_IDX["isimp"]].set(isimp)

    # Deep-space initialization (always computed, conditionally applied)
    ds_values = _deep_space_init_jax(
        ecco=ecco, argpo=argpo, inclo=inclo, mo=mo, nodeo=nodeo,
        no_unkozai=no_unkozai, epoch=epoch, gsto=gsto, mdot=mdot,
        nodedot=nodedot, xpidot=xpidot, eccsq=eccsq, xke=gravity.xke,
    )
    for name, val in ds_values.items():
        params = params.at[_IDX[name]].set(
            jnp.where(is_deep_space, val, params[_IDX[name]])
        )

    # Higher-order secular terms for non-simplified orbits
    is_not_simple = isimp < 0.5
    cc1sq = cc1 * cc1
    d2_val = 4.0 * ao * tsi * cc1sq
    temp_d = d2_val * tsi * cc1 / 3.0
    d3_val = (17.0 * ao + sfour) * temp_d
    d4_val = 0.5 * temp_d * ao * tsi * (221.0 * ao + 31.0 * sfour) * cc1
    t3cof_val = d2_val + 2.0 * cc1sq
    t4cof_val = 0.25 * (3.0 * d3_val + cc1 * (12.0 * d2_val + 10.0 * cc1sq))
    t5cof_val = 0.2 * (
        3.0 * d4_val
        + 12.0 * cc1 * d3_val
        + 6.0 * d2_val * d2_val
        + 15.0 * cc1sq * (2.0 * d2_val + cc1sq)
    )

    params = params.at[_IDX["d2"]].set(jnp.where(is_not_simple, d2_val, 0.0))
    params = params.at[_IDX["d3"]].set(jnp.where(is_not_simple, d3_val, 0.0))
    params = params.at[_IDX["d4"]].set(jnp.where(is_not_simple, d4_val, 0.0))
    params = params.at[_IDX["t3cof"]].set(jnp.where(is_not_simple, t3cof_val, 0.0))
    params = params.at[_IDX["t4cof"]].set(jnp.where(is_not_simple, t4cof_val, 0.0))
    params = params.at[_IDX["t5cof"]].set(jnp.where(is_not_simple, t5cof_val, 0.0))

    # Method flag
    params = params.at[_IDX["method"]].set(jnp.where(is_deep_space, 1.0, 0.0))

    return params


# ---------------------------------------------------------------------------
# SGP4 Propagation (JAX — JIT-compatible)
# ---------------------------------------------------------------------------


def _sgp4_propagate_near_earth(params: Array, tsince: ArrayLike) -> tuple[Array, Array]:
    """Propagate near-Earth satellite using SGP4 (JAX, JIT-compatible).

    Args:
        params: Flat parameter array from ``sgp4_init``.
        tsince: Time since epoch in minutes.

    Returns:
        Tuple of ``(r, v)`` where ``r`` is position [km] and ``v`` is
        velocity [km/s], both as 3-element arrays in the TEME frame.
        Returns NaN arrays if the satellite has decayed or an error occurs.
    """
    twopi = 2.0 * jnp.pi
    x2o3 = 2.0 / 3.0

    # Unpack parameters
    p = params
    radiusearthkm = p[_I["radiusearthkm"]]
    xke = p[_I["xke"]]
    j2 = p[_I["j2"]]
    bstar = p[_I["bstar"]]
    ecco = p[_I["ecco"]]
    argpo = p[_I["argpo"]]
    inclo = p[_I["inclo"]]
    mo = p[_I["mo"]]
    nodeo = p[_I["nodeo"]]
    no_unkozai = p[_I["no_unkozai"]]
    con41 = p[_I["con41"]]
    cc1 = p[_I["cc1"]]
    cc4 = p[_I["cc4"]]
    cc5 = p[_I["cc5"]]
    d2 = p[_I["d2"]]
    d3 = p[_I["d3"]]
    d4 = p[_I["d4"]]
    delmo = p[_I["delmo"]]
    eta = p[_I["eta"]]
    argpdot = p[_I["argpdot"]]
    omgcof = p[_I["omgcof"]]
    sinmao = p[_I["sinmao"]]
    t2cof = p[_I["t2cof"]]
    t3cof = p[_I["t3cof"]]
    t4cof = p[_I["t4cof"]]
    t5cof = p[_I["t5cof"]]
    x1mth2 = p[_I["x1mth2"]]
    x7thm1 = p[_I["x7thm1"]]
    mdot = p[_I["mdot"]]
    nodedot = p[_I["nodedot"]]
    xlcof = p[_I["xlcof"]]
    xmcof = p[_I["xmcof"]]
    nodecf = p[_I["nodecf"]]
    aycof = p[_I["aycof"]]
    isimp = p[_I["isimp"]]

    vkmpersec = radiusearthkm * xke / 60.0

    # --- Update for secular gravity and atmospheric drag ---
    t = tsince
    xmdf = mo + mdot * t
    argpdf = argpo + argpdot * t
    nodedf = nodeo + nodedot * t
    argpm = argpdf
    mm = xmdf
    t2 = t * t
    nodem = nodedf + nodecf * t2
    tempa = 1.0 - cc1 * t
    tempe = bstar * cc4 * t
    templ = t2cof * t2

    # Non-simplified orbit corrections
    is_not_simple = isimp < 0.5  # isimp == 0
    delomg = omgcof * t
    delmtemp = 1.0 + eta * jnp.cos(xmdf)
    delm = xmcof * (delmtemp * delmtemp * delmtemp - delmo)
    temp_corr = delomg + delm
    mm_ns = xmdf + temp_corr
    argpm_ns = argpdf - temp_corr
    t3 = t2 * t
    t4 = t3 * t
    tempa_ns = tempa - d2 * t2 - d3 * t3 - d4 * t4
    tempe_ns = tempe + bstar * cc5 * (jnp.sin(mm_ns) - sinmao)
    templ_ns = templ + t3cof * t3 + t4 * (t4cof + t * t5cof)

    mm = jnp.where(is_not_simple, mm_ns, mm)
    argpm = jnp.where(is_not_simple, argpm_ns, argpm)
    tempa = jnp.where(is_not_simple, tempa_ns, tempa)
    tempe = jnp.where(is_not_simple, tempe_ns, tempe)
    templ = jnp.where(is_not_simple, templ_ns, templ)

    nm = no_unkozai
    em = ecco
    inclm = inclo

    # Error check: nm <= 0
    nm_ok = nm > 0.0

    am = (xke / nm) ** x2o3 * tempa * tempa
    nm = xke / am**1.5
    em = em - tempe

    # Eccentricity bounds check
    em_ok = (em < 1.0) & (em >= -0.001)
    em = jnp.clip(em, 1.0e-6, 0.999999)

    mm = mm + no_unkozai * templ
    xlm = mm + argpm + nodem

    nodem = nodem % twopi
    argpm = argpm % twopi
    xlm = xlm % twopi
    mm = (xlm - argpm - nodem) % twopi

    # --- Compute extra mean quantities ---
    sinim = jnp.sin(inclm)
    cosim = jnp.cos(inclm)

    # Near-Earth: no lunar-solar periodics
    ep = em
    xincp = inclm
    argpp = argpm
    nodep = nodem
    mp = mm
    sinip = sinim
    cosip = cosim

    # --- Long period periodics ---
    axnl = ep * jnp.cos(argpp)
    temp_lp = 1.0 / (am * (1.0 - ep * ep))
    aynl = ep * jnp.sin(argpp) + temp_lp * aycof
    xl = mp + argpp + nodep + temp_lp * xlcof * axnl

    # --- Solve Kepler's equation ---
    u = (xl - nodep) % twopi

    def kepler_step(i, eo1):
        sineo1 = jnp.sin(eo1)
        coseo1 = jnp.cos(eo1)
        denom = 1.0 - coseo1 * axnl - sineo1 * aynl
        tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / denom
        tem5 = jnp.clip(tem5, -0.95, 0.95)
        return eo1 + tem5

    eo1 = jax.lax.fori_loop(0, 10, kepler_step, u)

    sineo1 = jnp.sin(eo1)
    coseo1 = jnp.cos(eo1)

    # --- Short period preliminary quantities ---
    ecose = axnl * coseo1 + aynl * sineo1
    esine = axnl * sineo1 - aynl * coseo1
    el2 = axnl * axnl + aynl * aynl
    pl = am * (1.0 - el2)

    # Error check: pl < 0
    pl_ok = pl >= 0.0

    rl = am * (1.0 - ecose)
    rdotl = jnp.sqrt(am) * esine / rl
    rvdotl = jnp.sqrt(pl) / rl
    betal = jnp.sqrt(1.0 - el2)
    temp_sp = esine / (1.0 + betal)
    sinu = am / rl * (sineo1 - aynl - axnl * temp_sp)
    cosu = am / rl * (coseo1 - axnl + aynl * temp_sp)
    su = jnp.arctan2(sinu, cosu)
    sin2u = (cosu + cosu) * sinu
    cos2u = 1.0 - 2.0 * sinu * sinu
    temp_sp2 = 1.0 / pl
    temp1 = 0.5 * j2 * temp_sp2
    temp2 = temp1 * temp_sp2

    # --- Update for short period periodics ---
    mrt = rl * (1.0 - 1.5 * temp2 * betal * con41) + 0.5 * temp1 * x1mth2 * cos2u
    su = su - 0.25 * temp2 * x7thm1 * sin2u
    xnode = nodep + 1.5 * temp2 * cosip * sin2u
    xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u
    mvt = rdotl - nm * temp1 * x1mth2 * sin2u / xke
    rvdot = rvdotl + nm * temp1 * (x1mth2 * cos2u + 1.5 * con41) / xke

    # --- Orientation vectors ---
    sinsu = jnp.sin(su)
    cossu = jnp.cos(su)
    snod = jnp.sin(xnode)
    cnod = jnp.cos(xnode)
    sini = jnp.sin(xinc)
    cosi = jnp.cos(xinc)
    xmx = -snod * cosi
    xmy = cnod * cosi
    ux = xmx * sinsu + cnod * cossu
    uy = xmy * sinsu + snod * cossu
    uz = sini * sinsu
    vx = xmx * cossu - cnod * sinsu
    vy = xmy * cossu - snod * sinsu
    vz = sini * cossu

    # --- Position and velocity (km and km/s) ---
    _mr = mrt * radiusearthkm
    r = jnp.array([_mr * ux, _mr * uy, _mr * uz])
    v = jnp.array(
        [
            (mvt * ux + rvdot * vx) * vkmpersec,
            (mvt * uy + rvdot * vy) * vkmpersec,
            (mvt * uz + rvdot * vz) * vkmpersec,
        ]
    )

    # Mask errors with NaN
    valid = nm_ok & em_ok & pl_ok & (mrt >= 1.0)
    nan3 = jnp.full(3, jnp.nan)
    r = jnp.where(valid, r, nan3)
    v = jnp.where(valid, v, nan3)

    return r, v


def _sgp4_propagate_deep_space(params: Array, tsince: ArrayLike) -> tuple[Array, Array]:
    """Propagate deep-space satellite using SDP4 (JAX, JIT-compatible).

    Args:
        params: Flat parameter array from ``sgp4_init``.
        tsince: Time since epoch in minutes.

    Returns:
        Tuple of ``(r, v)`` where ``r`` is position [km] and ``v`` is
        velocity [km/s], both as 3-element arrays in the TEME frame.
    """
    from astrojax.sgp4._deep_space import sgp4_propagate_deep_space_impl

    return sgp4_propagate_deep_space_impl(params, tsince, _I)


def sgp4_propagate(params: Array, tsince: ArrayLike, method: str) -> tuple[Array, Array]:
    """Propagate a satellite using SGP4/SDP4.

    This is the main propagation entry point. The ``method`` flag selects
    near-Earth ('n') or deep-space ('d') code path at Python trace time,
    making it compatible with ``jax.jit``.

    Args:
        params: Flat parameter array from ``sgp4_init``.
        tsince: Time since epoch in minutes.
        method: ``'n'`` for near-Earth SGP4, ``'d'`` for deep-space SDP4.

    Returns:
        Tuple of ``(r, v)`` where ``r`` is position [km] and ``v`` is
        velocity [km/s], both as 3-element arrays in the TEME frame.
    """
    if method == "n":
        return _sgp4_propagate_near_earth(params, tsince)
    else:
        return _sgp4_propagate_deep_space(params, tsince)


def sgp4_propagate_unified(params: Array, tsince: ArrayLike) -> tuple[Array, Array]:
    """Propagate a satellite using SGP4/SDP4 with auto-detection (JAX, JIT-compatible).

    Unlike ``sgp4_propagate``, this function does not require a separate
    ``method`` string. It reads the method flag from ``params`` (set by
    ``sgp4_init_jax`` or ``sgp4_init``) and dispatches to the correct
    propagation branch using ``jax.lax.cond``.

    This enables ``vmap`` over satellites with mixed near-earth and
    deep-space orbits in a single batch.

    Args:
        params: Flat parameter array from ``sgp4_init`` or ``sgp4_init_jax``.
        tsince: Time since epoch in minutes.

    Returns:
        Tuple of ``(r, v)`` where ``r`` is position [km] and ``v`` is
        velocity [km/s], both as 3-element arrays in the TEME frame.
    """
    is_deep = params[_IDX["method"]] > 0.5
    return jax.lax.cond(
        is_deep,
        _sgp4_propagate_deep_space,
        _sgp4_propagate_near_earth,
        params,
        tsince,
    )


def _create_propagator_from_elements(
    elements: SGP4Elements,
    gravity: EarthGravity,
) -> tuple[Array, Callable[[ArrayLike], tuple[Array, Array]]]:
    """Shared helper: init SGP4 from elements and return (params, propagate_fn).

    Args:
        elements: Parsed orbital elements.
        gravity: Earth gravity model constants.

    Returns:
        Tuple of ``(params, propagate_fn)`` where:
            - ``params`` is a flat ``jnp.array`` of satellite parameters
            - ``propagate_fn(tsince)`` takes time since epoch in minutes
              and returns ``(r_km, v_kms)`` in the TEME frame
    """
    params, method = sgp4_init(elements, gravity)

    if method == "n":
        propagate_fn = lambda tsince: _sgp4_propagate_near_earth(params, tsince)  # noqa: E731
    else:
        propagate_fn = lambda tsince: _sgp4_propagate_deep_space(params, tsince)  # noqa: E731

    return params, propagate_fn


def create_sgp4_propagator(
    line1: str,
    line2: str,
    gravity: str | EarthGravity = "wgs72",
) -> tuple[Array, Callable[[ArrayLike], tuple[Array, Array]]]:
    """Create a JIT-compatible SGP4 propagator from TLE lines.

    This is the functional API that returns a parameter array and a
    propagation closure suitable for ``jax.jit`` and ``jax.vmap``.

    Args:
        line1: First TLE line.
        line2: Second TLE line.
        gravity: Gravity model name (``'wgs72'``, ``'wgs84'``, ``'wgs72old'``)
            or an ``EarthGravity`` instance.

    Returns:
        Tuple of ``(params, propagate_fn)`` where:
            - ``params`` is a flat ``jnp.array`` of satellite parameters
            - ``propagate_fn(tsince)`` takes time since epoch in minutes
              and returns ``(r_km, v_kms)`` in the TEME frame
    """
    elements = parse_tle(line1, line2)

    if isinstance(gravity, str):
        gravity = GRAVITY_MODELS[gravity.lower()]

    return _create_propagator_from_elements(elements, gravity)


def create_sgp4_propagator_from_elements(
    elements: SGP4Elements,
    gravity: str | EarthGravity = "wgs72",
) -> tuple[Array, Callable[[ArrayLike], tuple[Array, Array]]]:
    """Create a JIT-compatible SGP4 propagator from pre-parsed elements.

    Accepts an :class:`SGP4Elements` dataclass (e.g. from ``parse_tle``
    or ``parse_omm``) and returns a parameter array and propagation closure.

    Args:
        elements: Parsed orbital elements from ``parse_tle`` or ``parse_omm``.
        gravity: Gravity model name (``'wgs72'``, ``'wgs84'``, ``'wgs72old'``)
            or an ``EarthGravity`` instance.

    Returns:
        Tuple of ``(params, propagate_fn)`` where:
            - ``params`` is a flat ``jnp.array`` of satellite parameters
            - ``propagate_fn(tsince)`` takes time since epoch in minutes
              and returns ``(r_km, v_kms)`` in the TEME frame
    """
    if isinstance(gravity, str):
        gravity = GRAVITY_MODELS[gravity.lower()]

    return _create_propagator_from_elements(elements, gravity)


def create_sgp4_propagator_from_omm(
    fields: dict[str, str],
    gravity: str | EarthGravity = "wgs72",
) -> tuple[Array, Callable[[ArrayLike], tuple[Array, Array]]]:
    """Create a JIT-compatible SGP4 propagator from OMM fields.

    Accepts a dictionary of OMM (Orbit Mean-Elements Message) field names
    and string values, as would be parsed from a CSV or XML OMM record.

    Args:
        fields: Dictionary mapping OMM field names to string values.
            See :func:`~astrojax.sgp4.parse_omm` for required keys.
        gravity: Gravity model name (``'wgs72'``, ``'wgs84'``, ``'wgs72old'``)
            or an ``EarthGravity`` instance.

    Returns:
        Tuple of ``(params, propagate_fn)`` where:
            - ``params`` is a flat ``jnp.array`` of satellite parameters
            - ``propagate_fn(tsince)`` takes time since epoch in minutes
              and returns ``(r_km, v_kms)`` in the TEME frame

    Raises:
        KeyError: If a required OMM field is missing.
    """
    elements = parse_omm(fields)

    if isinstance(gravity, str):
        gravity = GRAVITY_MODELS[gravity.lower()]

    return _create_propagator_from_elements(elements, gravity)


def create_sgp4_propagator_from_gp_record(
    record: GPRecord,
    gravity: str | EarthGravity = "wgs72",
) -> tuple[Array, Callable[[ArrayLike], tuple[Array, Array]]]:
    """Create a JIT-compatible SGP4 propagator from a GPRecord.

    Accepts a :class:`~astrojax.GPRecord` (e.g. from CelesTrak or
    SpaceTrack) and returns a parameter array and propagation closure.

    Args:
        record: A GP record from CelesTrak or SpaceTrack.
        gravity: Gravity model name (``'wgs72'``, ``'wgs84'``, ``'wgs72old'``)
            or an ``EarthGravity`` instance.

    Returns:
        Tuple of ``(params, propagate_fn)`` where:
            - ``params`` is a flat ``jnp.array`` of satellite parameters
            - ``propagate_fn(tsince)`` takes time since epoch in minutes
              and returns ``(r_km, v_kms)`` in the TEME frame

    Raises:
        KeyError: If the record is missing required fields.
    """
    elements = record.to_sgp4_elements()

    if isinstance(gravity, str):
        gravity = GRAVITY_MODELS[gravity.lower()]

    return _create_propagator_from_elements(elements, gravity)


# ---------------------------------------------------------------------------
# Convenience converters for JIT-compilable init
# ---------------------------------------------------------------------------


def omm_to_array(fields: dict[str, str]) -> Array:
    """Convert OMM fields dict to a flat JAX array for use with ``sgp4_init_jax``.

    This is a convenience wrapper that calls ``parse_omm`` followed by
    ``elements_to_array``.

    Args:
        fields: Dictionary mapping OMM field names to string values.
            See :func:`~astrojax.sgp4.parse_omm` for required keys.

    Returns:
        Array of shape ``(11,)`` containing the numeric orbital elements.

    Raises:
        KeyError: If a required OMM field is missing.
    """
    return elements_to_array(parse_omm(fields))


def gp_record_to_array(record: GPRecord) -> Array:
    """Convert a GPRecord to a flat JAX array for use with ``sgp4_init_jax``.

    This is a convenience wrapper that calls ``record.to_sgp4_elements()``
    followed by ``elements_to_array``.

    Args:
        record: A GP record from CelesTrak or SpaceTrack.

    Returns:
        Array of shape ``(11,)`` containing the numeric orbital elements.

    Raises:
        KeyError: If the record is missing required fields.
    """
    return elements_to_array(record.to_sgp4_elements())
