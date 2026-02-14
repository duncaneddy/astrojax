"""
Deep-space (SDP4) initialization and propagation routines.

This module contains the JAX translations of the deep-space routines:
``_dscom``, ``_dsinit``, ``_dspace``, and ``_dpper``. The init routines
run at Python time; the propagation routines are JAX-compatible.
"""

from math import atan2 as _py_atan2
from math import cos as _py_cos
from math import pi as _py_pi
from math import pow as _py_pow
from math import sin as _py_sin
from math import sqrt as _py_sqrt

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.sgp4._constants import EarthGravity
from astrojax.sgp4._types import SGP4Elements

_twopi = 2.0 * _py_pi


# ---------------------------------------------------------------------------
# Python-time deep-space init helpers
# ---------------------------------------------------------------------------


def _dscom_py(
    epoch: float,
    ep: float,
    argpp: float,
    tc: float,
    inclp: float,
    nodep: float,
    np_: float,
) -> tuple:
    """Compute deep-space common items (solar/lunar coefficients).

    Pure Python. Called during init.

    Args:
        epoch: Days since 1950 Jan 0.
        ep: Eccentricity.
        argpp: Argument of perigee [rad].
        tc: Time constant (0 during init).
        inclp: Inclination [rad].
        nodep: RAAN [rad].
        np_: Mean motion [rad/min].

    Returns:
        Large tuple of computed coefficients (see reference).
    """
    zes = 0.01675
    zel = 0.05490
    c1ss = 2.9864797e-6
    c1l = 4.7968065e-7
    zsinis = 0.39785416
    zcosis = 0.91744867
    zcosgs = 0.1945905
    zsings = -0.98088458

    nm = np_
    em = ep
    snodm = _py_sin(nodep)
    cnodm = _py_cos(nodep)
    sinomm = _py_sin(argpp)
    cosomm = _py_cos(argpp)
    sinim = _py_sin(inclp)
    cosim = _py_cos(inclp)
    emsq = em * em
    betasq = 1.0 - emsq
    rtemsq = _py_sqrt(betasq)

    peo = 0.0
    pinco = 0.0
    plo = 0.0
    pgho = 0.0
    pho = 0.0
    day = epoch + 18261.5 + tc / 1440.0
    xnodce = (4.5236020 - 9.2422029e-4 * day) % _twopi
    stem = _py_sin(xnodce)
    ctem = _py_cos(xnodce)
    zcosil = 0.91375164 - 0.03568096 * ctem
    zsinil = _py_sqrt(1.0 - zcosil * zcosil)
    zsinhl = 0.089683511 * stem / zsinil
    zcoshl = _py_sqrt(1.0 - zsinhl * zsinhl)
    gam = 5.8351514 + 0.0019443680 * day
    zx = 0.39785416 * stem / zsinil
    zy = zcoshl * ctem + 0.91744867 * zsinhl * stem
    zx = _py_atan2(zx, zy)
    zx = gam + zx - xnodce
    zcosgl = _py_cos(zx)
    zsingl = _py_sin(zx)

    zcosg = zcosgs
    zsing = zsings
    zcosi = zcosis
    zsini = zsinis
    zcosh = cnodm
    zsinh = snodm
    cc = c1ss
    xnoi = 1.0 / nm

    for lsflg in (1, 2):
        a1 = zcosg * zcosh + zsing * zcosi * zsinh
        a3 = -zsing * zcosh + zcosg * zcosi * zsinh
        a7 = -zcosg * zsinh + zsing * zcosi * zcosh
        a8 = zsing * zsini
        a9 = zsing * zsinh + zcosg * zcosi * zcosh
        a10 = zcosg * zsini
        a2 = cosim * a7 + sinim * a8
        a4 = cosim * a9 + sinim * a10
        a5 = -sinim * a7 + cosim * a8
        a6 = -sinim * a9 + cosim * a10

        x1 = a1 * cosomm + a2 * sinomm
        x2 = a3 * cosomm + a4 * sinomm
        x3 = -a1 * sinomm + a2 * cosomm
        x4 = -a3 * sinomm + a4 * cosomm
        x5 = a5 * sinomm
        x6 = a6 * sinomm
        x7 = a5 * cosomm
        x8 = a6 * cosomm

        z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3
        z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4
        z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4
        z1 = 3.0 * (a1 * a1 + a2 * a2) + z31 * emsq
        z2 = 6.0 * (a1 * a3 + a2 * a4) + z32 * emsq
        z3 = 3.0 * (a3 * a3 + a4 * a4) + z33 * emsq
        z11 = -6.0 * a1 * a5 + emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5)
        z12 = -6.0 * (a1 * a6 + a3 * a5) + emsq * (
            -24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5)
        )
        z13 = -6.0 * a3 * a6 + emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6)
        z21 = 6.0 * a2 * a5 + emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7)
        z22 = 6.0 * (a4 * a5 + a2 * a6) + emsq * (
            24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8)
        )
        z23 = 6.0 * a4 * a6 + emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8)
        z1 = z1 + z1 + betasq * z31
        z2 = z2 + z2 + betasq * z32
        z3 = z3 + z3 + betasq * z33
        s3 = cc * xnoi
        s2 = -0.5 * s3 / rtemsq
        s4 = s3 * rtemsq
        s1 = -15.0 * em * s4
        s5 = x1 * x3 + x2 * x4
        s6 = x2 * x3 + x1 * x4
        s7 = x2 * x4 - x1 * x3

        if lsflg == 1:
            ss1, ss2, ss3, ss4, ss5, ss6, ss7 = s1, s2, s3, s4, s5, s6, s7
            sz1, sz2, sz3 = z1, z2, z3
            sz11, sz12, sz13 = z11, z12, z13
            sz21, sz22, sz23 = z21, z22, z23
            sz31, sz32, sz33 = z31, z32, z33
            zcosg = zcosgl
            zsing = zsingl
            zcosi = zcosil
            zsini = zsinil
            zcosh = zcoshl * cnodm + zsinhl * snodm
            zsinh = snodm * zcoshl - cnodm * zsinhl
            cc = c1l

    zmol = (4.7199672 + 0.22997150 * day - gam) % _twopi
    zmos = (6.2565837 + 0.017201977 * day) % _twopi

    # Solar terms
    se2 = 2.0 * ss1 * ss6
    se3 = 2.0 * ss1 * ss7
    si2 = 2.0 * ss2 * sz12
    si3 = 2.0 * ss2 * (sz13 - sz11)
    sl2 = -2.0 * ss3 * sz2
    sl3 = -2.0 * ss3 * (sz3 - sz1)
    sl4 = -2.0 * ss3 * (-21.0 - 9.0 * emsq) * zes
    sgh2 = 2.0 * ss4 * sz32
    sgh3 = 2.0 * ss4 * (sz33 - sz31)
    sgh4 = -18.0 * ss4 * zes
    sh2 = -2.0 * ss2 * sz22
    sh3 = -2.0 * ss2 * (sz23 - sz21)

    # Lunar terms
    ee2 = 2.0 * s1 * s6
    e3 = 2.0 * s1 * s7
    xi2 = 2.0 * s2 * z12
    xi3 = 2.0 * s2 * (z13 - z11)
    xl2 = -2.0 * s3 * z2
    xl3 = -2.0 * s3 * (z3 - z1)
    xl4 = -2.0 * s3 * (-21.0 - 9.0 * emsq) * zel
    xgh2 = 2.0 * s4 * z32
    xgh3 = 2.0 * s4 * (z33 - z31)
    xgh4 = -18.0 * s4 * zel
    xh2 = -2.0 * s2 * z22
    xh3 = -2.0 * s2 * (z23 - z21)

    return (
        snodm,
        cnodm,
        sinim,
        cosim,
        sinomm,
        cosomm,
        day,
        e3,
        ee2,
        em,
        emsq,
        gam,
        peo,
        pgho,
        pho,
        pinco,
        plo,
        rtemsq,
        se2,
        se3,
        sgh2,
        sgh3,
        sgh4,
        sh2,
        sh3,
        si2,
        si3,
        sl2,
        sl3,
        sl4,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        ss1,
        ss2,
        ss3,
        ss4,
        ss5,
        ss6,
        ss7,
        sz1,
        sz2,
        sz3,
        sz11,
        sz12,
        sz13,
        sz21,
        sz22,
        sz23,
        sz31,
        sz32,
        sz33,
        xgh2,
        xgh3,
        xgh4,
        xh2,
        xh3,
        xi2,
        xi3,
        xl2,
        xl3,
        xl4,
        nm,
        z1,
        z2,
        z3,
        z11,
        z12,
        z13,
        z21,
        z22,
        z23,
        z31,
        z32,
        z33,
        zmol,
        zmos,
    )


def _dsinit_py(
    xke: float,
    cosim: float,
    emsq: float,
    argpo: float,
    s1: float,
    s2: float,
    s3: float,
    s4: float,
    s5: float,
    sinim: float,
    ss1: float,
    ss2: float,
    ss3: float,
    ss4: float,
    ss5: float,
    sz1: float,
    sz3: float,
    sz11: float,
    sz13: float,
    sz21: float,
    sz23: float,
    sz31: float,
    sz33: float,
    t: float,
    tc: float,
    gsto: float,
    mo: float,
    mdot: float,
    no: float,
    nodeo: float,
    nodedot: float,
    xpidot: float,
    z1: float,
    z3: float,
    z11: float,
    z13: float,
    z21: float,
    z23: float,
    z31: float,
    z33: float,
    ecco: float,
    eccsq: float,
    em: float,
    argpm: float,
    inclm: float,
    mm: float,
    nm: float,
    nodem: float,
) -> tuple:
    """Initialize deep-space resonance parameters.

    Pure Python. Called during init.

    Returns:
        Tuple of computed resonance parameters.
    """
    q22 = 1.7891679e-6
    q31 = 2.1460748e-6
    q33 = 2.2123015e-7
    root22 = 1.7891679e-6
    root44 = 7.3636953e-9
    root54 = 2.1765803e-9
    rptim = 4.37526908801129966e-3
    root32 = 3.7393792e-7
    root52 = 1.1428639e-7
    x2o3 = 2.0 / 3.0
    znl = 1.5835218e-4
    zns = 1.19459e-5

    irez = 0
    if 0.0034906585 < nm < 0.0052359877:
        irez = 1
    if 8.26e-3 <= nm <= 9.24e-3 and em >= 0.5:
        irez = 2

    # Solar terms
    ses = ss1 * zns * ss5
    sis = ss2 * zns * (sz11 + sz13)
    sls = -zns * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq)
    sghs = ss4 * zns * (sz31 + sz33 - 6.0)
    shs = -zns * ss2 * (sz21 + sz23)
    if inclm < 5.2359877e-2 or inclm > _py_pi - 5.2359877e-2:
        shs = 0.0
    if sinim != 0.0:
        shs = shs / sinim
    sgs = sghs - cosim * shs

    # Lunar terms
    dedt = ses + s1 * znl * s5
    didt = sis + s2 * znl * (z11 + z13)
    dmdt = sls - znl * s3 * (z1 + z3 - 14.0 - 6.0 * emsq)
    sghl = s4 * znl * (z31 + z33 - 6.0)
    shll = -znl * s2 * (z21 + z23)
    if inclm < 5.2359877e-2 or inclm > _py_pi - 5.2359877e-2:
        shll = 0.0
    domdt = sgs + sghl
    dnodt = shs
    if sinim != 0.0:
        domdt = domdt - cosim / sinim * shll
        dnodt = dnodt + shll / sinim

    # Deep space resonance effects
    dndt = 0.0
    theta = (gsto + tc * rptim) % _twopi
    em = em + dedt * t
    inclm = inclm + didt * t
    argpm = argpm + domdt * t
    nodem = nodem + dnodt * t
    mm = mm + dmdt * t

    # Initialize resonance terms
    d2201 = d2211 = d3210 = d3222 = 0.0
    d4410 = d4422 = d5220 = d5232 = d5421 = d5433 = 0.0
    del1 = del2 = del3 = 0.0
    xfact = xlamo = xli = xni = atime = 0.0

    if irez != 0:
        aonv = _py_pow(nm / xke, x2o3)

        # 12-hour resonance
        if irez == 2:
            cosisq = cosim * cosim
            emo = em
            em = ecco
            emsqo = emsq
            emsq = eccsq
            eoc = em * emsq
            g201 = -0.306 - (em - 0.64) * 0.440

            if em <= 0.65:
                g211 = 3.616 - 13.2470 * em + 16.2900 * emsq
                g310 = -19.302 + 117.3900 * em - 228.4190 * emsq + 156.5910 * eoc
                g322 = -18.9068 + 109.7927 * em - 214.6334 * emsq + 146.5816 * eoc
                g410 = -41.122 + 242.6940 * em - 471.0940 * emsq + 313.9530 * eoc
                g422 = -146.407 + 841.8800 * em - 1629.014 * emsq + 1083.4350 * eoc
                g520 = -532.114 + 3017.977 * em - 5740.032 * emsq + 3708.2760 * eoc
            else:
                g211 = -72.099 + 331.819 * em - 508.738 * emsq + 266.724 * eoc
                g310 = -346.844 + 1582.851 * em - 2415.925 * emsq + 1246.113 * eoc
                g322 = -342.585 + 1554.908 * em - 2366.899 * emsq + 1215.972 * eoc
                g410 = -1052.797 + 4758.686 * em - 7193.992 * emsq + 3651.957 * eoc
                g422 = -3581.690 + 16178.110 * em - 24462.770 * emsq + 12422.520 * eoc
                if em > 0.715:
                    g520 = -5149.66 + 29936.92 * em - 54087.36 * emsq + 31324.56 * eoc
                else:
                    g520 = 1464.74 - 4664.75 * em + 3763.64 * emsq

            if em < 0.7:
                g533 = -919.22770 + 4988.6100 * em - 9064.7700 * emsq + 5542.21 * eoc
                g521 = -822.71072 + 4568.6173 * em - 8491.4146 * emsq + 5337.524 * eoc
                g532 = -853.66600 + 4690.2500 * em - 8624.7700 * emsq + 5341.4 * eoc
            else:
                g533 = -37995.780 + 161616.52 * em - 229838.20 * emsq + 109377.94 * eoc
                g521 = -51752.104 + 218913.95 * em - 309468.16 * emsq + 146349.42 * eoc
                g532 = -40023.880 + 170470.89 * em - 242699.48 * emsq + 115605.82 * eoc

            sini2 = sinim * sinim
            f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq)
            f221 = 1.5 * sini2
            f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq)
            f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq)
            f441 = 35.0 * sini2 * f220
            f442 = 39.3750 * sini2 * sini2
            f522 = (
                9.84375
                * sinim
                * (
                    sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq)
                    + 0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq)
                )
            )
            f523 = sinim * (
                4.92187512 * sini2 * (-2.0 - 4.0 * cosim + 10.0 * cosisq)
                + 6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq)
            )
            f542 = (
                29.53125
                * sinim
                * (2.0 - 8.0 * cosim + cosisq * (-12.0 + 8.0 * cosim + 10.0 * cosisq))
            )
            f543 = (
                29.53125
                * sinim
                * (-2.0 - 8.0 * cosim + cosisq * (12.0 + 8.0 * cosim - 10.0 * cosisq))
            )
            xno2 = nm * nm
            ainv2 = aonv * aonv
            temp1 = 3.0 * xno2 * ainv2
            temp = temp1 * root22
            d2201 = temp * f220 * g201
            d2211 = temp * f221 * g211
            temp1 = temp1 * aonv
            temp = temp1 * root32
            d3210 = temp * f321 * g310
            d3222 = temp * f322 * g322
            temp1 = temp1 * aonv
            temp = 2.0 * temp1 * root44
            d4410 = temp * f441 * g410
            d4422 = temp * f442 * g422
            temp1 = temp1 * aonv
            temp = temp1 * root52
            d5220 = temp * f522 * g520
            d5232 = temp * f523 * g532
            temp = 2.0 * temp1 * root54
            d5421 = temp * f542 * g521
            d5433 = temp * f543 * g533
            xlamo = (mo + nodeo + nodeo - theta - theta) % _twopi
            xfact = mdot + dmdt + 2.0 * (nodedot + dnodt - rptim) - no
            em = emo
            emsq = emsqo

        # Synchronous resonance
        if irez == 1:
            g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq)
            g310 = 1.0 + 2.0 * emsq
            g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq)
            f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim)
            f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (1.0 + cosim)
            f330 = 1.0 + cosim
            f330 = 1.875 * f330 * f330 * f330
            del1 = 3.0 * nm * nm * aonv * aonv
            del2 = 2.0 * del1 * f220 * g200 * q22
            del3 = 3.0 * del1 * f330 * g300 * q33 * aonv
            del1 = del1 * f311 * g310 * q31 * aonv
            xlamo = (mo + nodeo + argpo - theta) % _twopi
            xfact = mdot + xpidot - rptim + dmdt + domdt + dnodt - no

        # Initialize integrator
        xli = xlamo
        xni = no
        atime = 0.0
        nm = no + dndt

    return (
        em,
        argpm,
        inclm,
        mm,
        nm,
        nodem,
        irez,
        atime,
        d2201,
        d2211,
        d3210,
        d3222,
        d4410,
        d4422,
        d5220,
        d5232,
        d5421,
        d5433,
        dedt,
        didt,
        dmdt,
        dndt,
        dnodt,
        domdt,
        del1,
        del2,
        del3,
        xfact,
        xlamo,
        xli,
        xni,
    )


def _dpper_init_py(
    d: dict[str, float],
    inclo: float,
    ecco: float,
    inclo_in: float,
    nodeo: float,
    argpo: float,
    mo: float,
    opsmode: str,
) -> tuple[float, float, float, float, float]:
    """Apply deep-space periodic perturbations during init (init='y').

    During init, this is essentially a no-op for the orbital elements
    (the corrections are computed but not applied). Returns elements unchanged.
    """
    # When init == 'y', the function computes but doesn't apply corrections
    return ecco, inclo_in, nodeo, argpo, mo


# ---------------------------------------------------------------------------
# Deep-space init orchestrator
# ---------------------------------------------------------------------------


def deep_space_init(
    d: dict[str, float],
    elements: SGP4Elements,
    gravity: EarthGravity,
    epoch: float,
    opsmode: str,
    no_unkozai: float,
    ao: float,
    cosio: float,
    cosio2: float,
    eccsq: float,
    omeosq: float,
    sinio: float,
    rteosq: float,
    con41: float,
    con42: float,
    cosio4: float,
    posq: float,
    xpidot: float,
    xhdot1: float,
    sfour: float,
    tsi: float,
    eta: float,
    etasq: float,
    eeta: float,
    psisq: float,
    cc1: float,
    cc3: float,
    coef: float,
    coef1: float,
    temp1: float,
    temp2: float,
    temp3: float,
    pinvsq: float,
) -> None:
    """Initialize deep-space parameters. Modifies ``d`` in place."""
    tc = 0.0
    inclm = elements.inclo

    # Run dscom
    (
        snodm,
        cnodm,
        sinim,
        cosim,
        sinomm,
        cosomm,
        day,
        e3,
        ee2,
        em,
        emsq,
        gam,
        peo,
        pgho,
        pho,
        pinco,
        plo,
        rtemsq_ds,
        se2,
        se3,
        sgh2,
        sgh3,
        sgh4,
        sh2,
        sh3,
        si2,
        si3,
        sl2,
        sl3,
        sl4,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        ss1,
        ss2,
        ss3,
        ss4,
        ss5,
        ss6,
        ss7,
        sz1,
        sz2,
        sz3,
        sz11,
        sz12,
        sz13,
        sz21,
        sz22,
        sz23,
        sz31,
        sz32,
        sz33,
        xgh2,
        xgh3,
        xgh4,
        xh2,
        xh3,
        xi2,
        xi3,
        xl2,
        xl3,
        xl4,
        nm,
        z1,
        z2,
        z3,
        z11,
        z12,
        z13,
        z21,
        z22,
        z23,
        z31,
        z32,
        z33,
        zmol,
        zmos,
    ) = _dscom_py(
        epoch,
        elements.ecco,
        elements.argpo,
        tc,
        elements.inclo,
        elements.nodeo,
        no_unkozai,
    )

    d["e3"] = e3
    d["ee2"] = ee2
    d["peo"] = peo
    d["pgho"] = pgho
    d["pho"] = pho
    d["pinco"] = pinco
    d["plo"] = plo
    d["se2"] = se2
    d["se3"] = se3
    d["sgh2"] = sgh2
    d["sgh3"] = sgh3
    d["sgh4"] = sgh4
    d["sh2"] = sh2
    d["sh3"] = sh3
    d["si2"] = si2
    d["si3"] = si3
    d["sl2"] = sl2
    d["sl3"] = sl3
    d["sl4"] = sl4
    d["xgh2"] = xgh2
    d["xgh3"] = xgh3
    d["xgh4"] = xgh4
    d["xh2"] = xh2
    d["xh3"] = xh3
    d["xi2"] = xi2
    d["xi3"] = xi3
    d["xl2"] = xl2
    d["xl3"] = xl3
    d["xl4"] = xl4
    d["zmol"] = zmol
    d["zmos"] = zmos

    # dpper during init is a no-op for orbital elements
    # (init='y' means corrections computed but not applied)

    # Run dsinit
    (
        em_ds,
        argpm_ds,
        inclm_ds,
        mm_ds,
        nm_ds,
        nodem_ds,
        irez,
        atime,
        d2201,
        d2211,
        d3210,
        d3222,
        d4410,
        d4422,
        d5220,
        d5232,
        d5421,
        d5433,
        dedt,
        didt,
        dmdt,
        dndt,
        dnodt,
        domdt,
        del1,
        del2,
        del3,
        xfact,
        xlamo,
        xli,
        xni,
    ) = _dsinit_py(
        gravity.xke,
        cosim,
        emsq,
        elements.argpo,
        s1,
        s2,
        s3,
        s4,
        s5,
        sinim,
        ss1,
        ss2,
        ss3,
        ss4,
        ss5,
        sz1,
        sz3,
        sz11,
        sz13,
        sz21,
        sz23,
        sz31,
        sz33,
        0.0,
        tc,
        d["gsto"],
        elements.mo,
        d["mdot"],
        no_unkozai,
        elements.nodeo,
        d["nodedot"],
        xpidot,
        z1,
        z3,
        z11,
        z13,
        z21,
        z23,
        z31,
        z33,
        elements.ecco,
        eccsq,
        em,
        0.0,
        inclm,
        0.0,
        nm,
        0.0,
    )

    d["irez"] = float(irez)
    d["atime"] = atime
    d["d2201"] = d2201
    d["d2211"] = d2211
    d["d3210"] = d3210
    d["d3222"] = d3222
    d["d4410"] = d4410
    d["d4422"] = d4422
    d["d5220"] = d5220
    d["d5232"] = d5232
    d["d5421"] = d5421
    d["d5433"] = d5433
    d["dedt"] = dedt
    d["didt"] = didt
    d["dmdt"] = dmdt
    d["dnodt"] = dnodt
    d["domdt"] = domdt
    d["del1"] = del1
    d["del2"] = del2
    d["del3"] = del3
    d["xfact"] = xfact
    d["xlamo"] = xlamo
    d["xli"] = xli
    d["xni"] = xni


# ---------------------------------------------------------------------------
# JAX-compatible deep-space init helpers (JIT-safe)
# ---------------------------------------------------------------------------

_twopi_jax = 2.0 * jnp.pi


def _dscom_pass(
    zcosg: ArrayLike,
    zsing: ArrayLike,
    zcosi: ArrayLike,
    zsini: ArrayLike,
    zcosh: ArrayLike,
    zsinh: ArrayLike,
    cc: ArrayLike,
    xnoi: ArrayLike,
    em: ArrayLike,
    emsq: ArrayLike,
    betasq: ArrayLike,
    rtemsq: ArrayLike,
    cosomm: ArrayLike,
    sinomm: ArrayLike,
    cosim: ArrayLike,
    sinim: ArrayLike,
) -> tuple:
    """Compute one pass of the dscom solar/lunar coefficient loop (JAX).

    Args:
        zcosg: Cosine of g angle.
        zsing: Sine of g angle.
        zcosi: Cosine of i angle.
        zsini: Sine of i angle.
        zcosh: Cosine of h angle.
        zsinh: Sine of h angle.
        cc: Coupling constant (c1ss or c1l).
        xnoi: 1/nm.
        em: Eccentricity.
        emsq: Eccentricity squared.
        betasq: 1 - emsq.
        rtemsq: sqrt(betasq).
        cosomm: cos(argpp).
        sinomm: sin(argpp).
        cosim: cos(inclp).
        sinim: sin(inclp).

    Returns:
        Tuple of (s1..s7, z1..z3, z11..z13, z21..z23, z31..z33).
    """
    a1 = zcosg * zcosh + zsing * zcosi * zsinh
    a3 = -zsing * zcosh + zcosg * zcosi * zsinh
    a7 = -zcosg * zsinh + zsing * zcosi * zcosh
    a8 = zsing * zsini
    a9 = zsing * zsinh + zcosg * zcosi * zcosh
    a10 = zcosg * zsini
    a2 = cosim * a7 + sinim * a8
    a4 = cosim * a9 + sinim * a10
    a5 = -sinim * a7 + cosim * a8
    a6 = -sinim * a9 + cosim * a10

    x1 = a1 * cosomm + a2 * sinomm
    x2 = a3 * cosomm + a4 * sinomm
    x3 = -a1 * sinomm + a2 * cosomm
    x4 = -a3 * sinomm + a4 * cosomm
    x5 = a5 * sinomm
    x6 = a6 * sinomm
    x7 = a5 * cosomm
    x8 = a6 * cosomm

    z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3
    z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4
    z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4
    z1 = 3.0 * (a1 * a1 + a2 * a2) + z31 * emsq
    z2 = 6.0 * (a1 * a3 + a2 * a4) + z32 * emsq
    z3 = 3.0 * (a3 * a3 + a4 * a4) + z33 * emsq
    z11 = -6.0 * a1 * a5 + emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5)
    z12 = -6.0 * (a1 * a6 + a3 * a5) + emsq * (
        -24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5)
    )
    z13 = -6.0 * a3 * a6 + emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6)
    z21 = 6.0 * a2 * a5 + emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7)
    z22 = 6.0 * (a4 * a5 + a2 * a6) + emsq * (
        24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8)
    )
    z23 = 6.0 * a4 * a6 + emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8)
    z1 = z1 + z1 + betasq * z31
    z2 = z2 + z2 + betasq * z32
    z3 = z3 + z3 + betasq * z33
    s3 = cc * xnoi
    s2 = -0.5 * s3 / rtemsq
    s4 = s3 * rtemsq
    s1 = -15.0 * em * s4
    s5 = x1 * x3 + x2 * x4
    s6 = x2 * x3 + x1 * x4
    s7 = x2 * x4 - x1 * x3

    return (
        s1, s2, s3, s4, s5, s6, s7,
        z1, z2, z3, z11, z12, z13, z21, z22, z23, z31, z32, z33,
    )


def _dscom_jax(
    epoch: ArrayLike,
    ep: ArrayLike,
    argpp: ArrayLike,
    tc: ArrayLike,
    inclp: ArrayLike,
    nodep: ArrayLike,
    np_: ArrayLike,
) -> tuple:
    """Compute deep-space common items (JAX, JIT-compatible).

    Args:
        epoch: Days since 1950 Jan 0.
        ep: Eccentricity.
        argpp: Argument of perigee [rad].
        tc: Time constant (0 during init).
        inclp: Inclination [rad].
        nodep: RAAN [rad].
        np_: Mean motion [rad/min].

    Returns:
        Large tuple of computed coefficients (same as _dscom_py).
    """
    zes = 0.01675
    zel = 0.05490
    c1ss = 2.9864797e-6
    c1l = 4.7968065e-7
    zsinis = 0.39785416
    zcosis = 0.91744867
    zcosgs = 0.1945905
    zsings = -0.98088458

    nm = np_
    em = ep
    snodm = jnp.sin(nodep)
    cnodm = jnp.cos(nodep)
    sinomm = jnp.sin(argpp)
    cosomm = jnp.cos(argpp)
    sinim = jnp.sin(inclp)
    cosim = jnp.cos(inclp)
    emsq = em * em
    betasq = 1.0 - emsq
    rtemsq = jnp.sqrt(betasq)

    peo = 0.0
    pinco = 0.0
    plo = 0.0
    pgho = 0.0
    pho = 0.0
    day = epoch + 18261.5 + tc / 1440.0
    xnodce = (4.5236020 - 9.2422029e-4 * day) % _twopi_jax
    stem = jnp.sin(xnodce)
    ctem = jnp.cos(xnodce)
    zcosil = 0.91375164 - 0.03568096 * ctem
    zsinil = jnp.sqrt(1.0 - zcosil * zcosil)
    zsinhl = 0.089683511 * stem / zsinil
    zcoshl = jnp.sqrt(1.0 - zsinhl * zsinhl)
    gam = 5.8351514 + 0.0019443680 * day
    zx = 0.39785416 * stem / zsinil
    zy = zcoshl * ctem + 0.91744867 * zsinhl * stem
    zx = jnp.arctan2(zx, zy)
    zx = gam + zx - xnodce
    zcosgl = jnp.cos(zx)
    zsingl = jnp.sin(zx)

    xnoi = 1.0 / nm

    # Solar pass (lsflg=1)
    (
        ss1, ss2, ss3, ss4, ss5, ss6, ss7,
        sz1, sz2, sz3, sz11, sz12, sz13, sz21, sz22, sz23, sz31, sz32, sz33,
    ) = _dscom_pass(
        zcosgs, zsings, zcosis, zsinis, cnodm, snodm, c1ss,
        xnoi, em, emsq, betasq, rtemsq, cosomm, sinomm, cosim, sinim,
    )

    # Lunar pass (lsflg=2)
    lunar_zcosh = zcoshl * cnodm + zsinhl * snodm
    lunar_zsinh = snodm * zcoshl - cnodm * zsinhl
    (
        s1, s2, s3, s4, s5, s6, s7,
        z1, z2, z3, z11, z12, z13, z21, z22, z23, z31, z32, z33,
    ) = _dscom_pass(
        zcosgl, zsingl, zcosil, zsinil, lunar_zcosh, lunar_zsinh, c1l,
        xnoi, em, emsq, betasq, rtemsq, cosomm, sinomm, cosim, sinim,
    )

    zmol = (4.7199672 + 0.22997150 * day - gam) % _twopi_jax
    zmos = (6.2565837 + 0.017201977 * day) % _twopi_jax

    # Solar terms
    se2 = 2.0 * ss1 * ss6
    se3 = 2.0 * ss1 * ss7
    si2 = 2.0 * ss2 * sz12
    si3 = 2.0 * ss2 * (sz13 - sz11)
    sl2 = -2.0 * ss3 * sz2
    sl3 = -2.0 * ss3 * (sz3 - sz1)
    sl4 = -2.0 * ss3 * (-21.0 - 9.0 * emsq) * zes
    sgh2 = 2.0 * ss4 * sz32
    sgh3 = 2.0 * ss4 * (sz33 - sz31)
    sgh4 = -18.0 * ss4 * zes
    sh2 = -2.0 * ss2 * sz22
    sh3 = -2.0 * ss2 * (sz23 - sz21)

    # Lunar terms
    ee2 = 2.0 * s1 * s6
    e3 = 2.0 * s1 * s7
    xi2 = 2.0 * s2 * z12
    xi3 = 2.0 * s2 * (z13 - z11)
    xl2 = -2.0 * s3 * z2
    xl3 = -2.0 * s3 * (z3 - z1)
    xl4 = -2.0 * s3 * (-21.0 - 9.0 * emsq) * zel
    xgh2 = 2.0 * s4 * z32
    xgh3 = 2.0 * s4 * (z33 - z31)
    xgh4 = -18.0 * s4 * zel
    xh2 = -2.0 * s2 * z22
    xh3 = -2.0 * s2 * (z23 - z21)

    return (
        snodm, cnodm, sinim, cosim, sinomm, cosomm,
        day, e3, ee2, em, emsq, gam,
        peo, pgho, pho, pinco, plo,
        rtemsq,
        se2, se3, sgh2, sgh3, sgh4, sh2, sh3, si2, si3, sl2, sl3, sl4,
        s1, s2, s3, s4, s5, s6, s7,
        ss1, ss2, ss3, ss4, ss5, ss6, ss7,
        sz1, sz2, sz3, sz11, sz12, sz13, sz21, sz22, sz23, sz31, sz32, sz33,
        xgh2, xgh3, xgh4, xh2, xh3, xi2, xi3, xl2, xl3, xl4,
        nm, z1, z2, z3, z11, z12, z13, z21, z22, z23, z31, z32, z33,
        zmol, zmos,
    )


def _dsinit_jax(
    xke: ArrayLike,
    cosim: ArrayLike,
    emsq: ArrayLike,
    argpo: ArrayLike,
    s1: ArrayLike,
    s2: ArrayLike,
    s3: ArrayLike,
    s4: ArrayLike,
    s5: ArrayLike,
    sinim: ArrayLike,
    ss1: ArrayLike,
    ss2: ArrayLike,
    ss3: ArrayLike,
    ss4: ArrayLike,
    ss5: ArrayLike,
    sz1: ArrayLike,
    sz3: ArrayLike,
    sz11: ArrayLike,
    sz13: ArrayLike,
    sz21: ArrayLike,
    sz23: ArrayLike,
    sz31: ArrayLike,
    sz33: ArrayLike,
    t: ArrayLike,
    tc: ArrayLike,
    gsto: ArrayLike,
    mo: ArrayLike,
    mdot: ArrayLike,
    no: ArrayLike,
    nodeo: ArrayLike,
    nodedot: ArrayLike,
    xpidot: ArrayLike,
    z1: ArrayLike,
    z3: ArrayLike,
    z11: ArrayLike,
    z13: ArrayLike,
    z21: ArrayLike,
    z23: ArrayLike,
    z31: ArrayLike,
    z33: ArrayLike,
    ecco: ArrayLike,
    eccsq: ArrayLike,
    em: ArrayLike,
    argpm: ArrayLike,
    inclm: ArrayLike,
    mm: ArrayLike,
    nm: ArrayLike,
    nodem: ArrayLike,
) -> tuple:
    """Initialize deep-space resonance parameters (JAX, JIT-compatible).

    Args:
        xke: Gravity constant.
        cosim: cos(inclination).
        emsq: Eccentricity squared.
        argpo: Argument of perigee [rad].
        s1..s7: Lunar coupling coefficients from dscom.
        sinim: sin(inclination).
        ss1..ss7: Solar coupling coefficients from dscom.
        sz1..sz33: Solar z-coefficients from dscom.
        t: Time since epoch [min] (0 during init).
        tc: Time constant.
        gsto: Greenwich sidereal time [rad].
        mo: Mean anomaly [rad].
        mdot: Mean motion rate [rad/min].
        no: Un-Kozai mean motion [rad/min].
        nodeo: RAAN [rad].
        nodedot: RAAN rate [rad/min].
        xpidot: argpdot + nodedot.
        z1..z33: Lunar z-coefficients from dscom.
        ecco: Original eccentricity.
        eccsq: Original eccentricity squared.
        em: Eccentricity (may differ from ecco during propagation).
        argpm: Argument of perigee (0 during init).
        inclm: Inclination [rad].
        mm: Mean anomaly (0 during init).
        nm: Mean motion from dscom.
        nodem: RAAN (0 during init).

    Returns:
        Tuple of computed resonance parameters.
    """
    q22 = 1.7891679e-6
    q31 = 2.1460748e-6
    q33 = 2.2123015e-7
    root22 = 1.7891679e-6
    root44 = 7.3636953e-9
    root54 = 2.1765803e-9
    rptim = 4.37526908801129966e-3
    root32 = 3.7393792e-7
    root52 = 1.1428639e-7
    x2o3 = 2.0 / 3.0
    znl = 1.5835218e-4
    zns = 1.19459e-5

    # Resonance type detection
    is_sync = (nm > 0.0034906585) & (nm < 0.0052359877)
    is_12h = (nm >= 8.26e-3) & (nm <= 9.24e-3) & (em >= 0.5)
    irez = jnp.where(is_12h, 2.0, jnp.where(is_sync, 1.0, 0.0))

    # Solar terms
    ses = ss1 * zns * ss5
    sis = ss2 * zns * (sz11 + sz13)
    sls = -zns * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq)
    sghs = ss4 * zns * (sz31 + sz33 - 6.0)
    shs = -zns * ss2 * (sz21 + sz23)

    # Inclination singularity
    is_incl_singular = (inclm < 5.2359877e-2) | (inclm > jnp.pi - 5.2359877e-2)
    shs = jnp.where(is_incl_singular, 0.0, shs)
    safe_sinim = jnp.where(jnp.abs(sinim) > 0.0, sinim, 1.0)
    shs = jnp.where(jnp.abs(sinim) > 0.0, shs / safe_sinim, shs)
    sgs = sghs - cosim * shs

    # Lunar terms
    dedt = ses + s1 * znl * s5
    didt = sis + s2 * znl * (z11 + z13)
    dmdt = sls - znl * s3 * (z1 + z3 - 14.0 - 6.0 * emsq)
    sghl = s4 * znl * (z31 + z33 - 6.0)
    shll = -znl * s2 * (z21 + z23)
    shll = jnp.where(is_incl_singular, 0.0, shll)
    domdt = sgs + sghl
    dnodt = shs
    domdt = jnp.where(
        jnp.abs(sinim) > 0.0,
        domdt - cosim / safe_sinim * shll,
        domdt,
    )
    dnodt = jnp.where(
        jnp.abs(sinim) > 0.0,
        dnodt + shll / safe_sinim,
        dnodt,
    )

    # Deep space secular effects
    dndt = 0.0
    theta = (gsto + tc * rptim) % _twopi_jax
    em = em + dedt * t
    inclm = inclm + didt * t
    argpm = argpm + domdt * t
    nodem = nodem + dnodt * t
    mm = mm + dmdt * t

    # Resonance terms
    is_resonant = irez > 0.5
    aonv = (nm / xke) ** x2o3

    # ====== 12-hour resonance (irez == 2) ======
    cosisq = cosim * cosim
    # Use ecco/eccsq for g-coefficients in 12-hour resonance
    em_12h = ecco
    emsq_12h = eccsq
    eoc_12h = em_12h * emsq_12h
    g201_12h = -0.306 - (em_12h - 0.64) * 0.440

    # Low eccentricity branch (em_12h <= 0.65)
    g211_lo = 3.616 - 13.2470 * em_12h + 16.2900 * emsq_12h
    g310_lo = -19.302 + 117.3900 * em_12h - 228.4190 * emsq_12h + 156.5910 * eoc_12h
    g322_lo = -18.9068 + 109.7927 * em_12h - 214.6334 * emsq_12h + 146.5816 * eoc_12h
    g410_lo = -41.122 + 242.6940 * em_12h - 471.0940 * emsq_12h + 313.9530 * eoc_12h
    g422_lo = -146.407 + 841.8800 * em_12h - 1629.014 * emsq_12h + 1083.4350 * eoc_12h
    g520_lo = -532.114 + 3017.977 * em_12h - 5740.032 * emsq_12h + 3708.2760 * eoc_12h

    # High eccentricity branch (em_12h > 0.65)
    g211_hi = -72.099 + 331.819 * em_12h - 508.738 * emsq_12h + 266.724 * eoc_12h
    g310_hi = -346.844 + 1582.851 * em_12h - 2415.925 * emsq_12h + 1246.113 * eoc_12h
    g322_hi = -342.585 + 1554.908 * em_12h - 2366.899 * emsq_12h + 1215.972 * eoc_12h
    g410_hi = -1052.797 + 4758.686 * em_12h - 7193.992 * emsq_12h + 3651.957 * eoc_12h
    g422_hi = -3581.690 + 16178.110 * em_12h - 24462.770 * emsq_12h + 12422.520 * eoc_12h
    g520_hi_hi = -5149.66 + 29936.92 * em_12h - 54087.36 * emsq_12h + 31324.56 * eoc_12h
    g520_hi_lo = 1464.74 - 4664.75 * em_12h + 3763.64 * emsq_12h
    g520_hi = jnp.where(em_12h > 0.715, g520_hi_hi, g520_hi_lo)

    is_low_ecc = em_12h <= 0.65
    g211 = jnp.where(is_low_ecc, g211_lo, g211_hi)
    g310 = jnp.where(is_low_ecc, g310_lo, g310_hi)
    g322 = jnp.where(is_low_ecc, g322_lo, g322_hi)
    g410 = jnp.where(is_low_ecc, g410_lo, g410_hi)
    g422 = jnp.where(is_low_ecc, g422_lo, g422_hi)
    g520 = jnp.where(is_low_ecc, g520_lo, g520_hi)

    # g533, g521, g532 selection
    is_low_ecc_07 = em_12h < 0.7
    g533_lo = -919.22770 + 4988.6100 * em_12h - 9064.7700 * emsq_12h + 5542.21 * eoc_12h
    g521_lo = -822.71072 + 4568.6173 * em_12h - 8491.4146 * emsq_12h + 5337.524 * eoc_12h
    g532_lo = -853.66600 + 4690.2500 * em_12h - 8624.7700 * emsq_12h + 5341.4 * eoc_12h
    g533_hi = -37995.780 + 161616.52 * em_12h - 229838.20 * emsq_12h + 109377.94 * eoc_12h
    g521_hi = -51752.104 + 218913.95 * em_12h - 309468.16 * emsq_12h + 146349.42 * eoc_12h
    g532_hi = -40023.880 + 170470.89 * em_12h - 242699.48 * emsq_12h + 115605.82 * eoc_12h
    g533 = jnp.where(is_low_ecc_07, g533_lo, g533_hi)
    g521 = jnp.where(is_low_ecc_07, g521_lo, g521_hi)
    g532 = jnp.where(is_low_ecc_07, g532_lo, g532_hi)

    sini2 = sinim * sinim
    f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq)
    f221 = 1.5 * sini2
    f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq)
    f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq)
    f441 = 35.0 * sini2 * f220
    f442 = 39.3750 * sini2 * sini2
    f522 = (
        9.84375
        * sinim
        * (
            sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq)
            + 0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq)
        )
    )
    f523 = sinim * (
        4.92187512 * sini2 * (-2.0 - 4.0 * cosim + 10.0 * cosisq)
        + 6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq)
    )
    f542 = (
        29.53125
        * sinim
        * (2.0 - 8.0 * cosim + cosisq * (-12.0 + 8.0 * cosim + 10.0 * cosisq))
    )
    f543 = (
        29.53125
        * sinim
        * (-2.0 - 8.0 * cosim + cosisq * (12.0 + 8.0 * cosim - 10.0 * cosisq))
    )

    xno2 = nm * nm
    ainv2 = aonv * aonv
    temp1_12h = 3.0 * xno2 * ainv2
    temp_12h = temp1_12h * root22
    d2201_12h = temp_12h * f220 * g201_12h
    d2211_12h = temp_12h * f221 * g211
    temp1_12h = temp1_12h * aonv
    temp_12h = temp1_12h * root32
    d3210_12h = temp_12h * f321 * g310
    d3222_12h = temp_12h * f322 * g322
    temp1_12h = temp1_12h * aonv
    temp_12h = 2.0 * temp1_12h * root44
    d4410_12h = temp_12h * f441 * g410
    d4422_12h = temp_12h * f442 * g422
    temp1_12h = temp1_12h * aonv
    temp_12h = temp1_12h * root52
    d5220_12h = temp_12h * f522 * g520
    d5232_12h = temp_12h * f523 * g532
    temp_12h = 2.0 * temp1_12h * root54
    d5421_12h = temp_12h * f542 * g521
    d5433_12h = temp_12h * f543 * g533
    xlamo_12h = (mo + nodeo + nodeo - theta - theta) % _twopi_jax
    xfact_12h = mdot + dmdt + 2.0 * (nodedot + dnodt - rptim) - no

    # ====== Synchronous resonance (irez == 1) ======
    g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq)
    g310_sync = 1.0 + 2.0 * emsq
    g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq)
    f220_sync = 0.75 * (1.0 + cosim) * (1.0 + cosim)
    f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (1.0 + cosim)
    f330 = 1.0 + cosim
    f330 = 1.875 * f330 * f330 * f330
    del1_sync = 3.0 * nm * nm * aonv * aonv
    del2_sync = 2.0 * del1_sync * f220_sync * g200 * q22
    del3_sync = 3.0 * del1_sync * f330 * g300 * q33 * aonv
    del1_sync = del1_sync * f311 * g310_sync * q31 * aonv
    xlamo_sync = (mo + nodeo + argpo - theta) % _twopi_jax
    xfact_sync = mdot + xpidot - rptim + dmdt + domdt + dnodt - no

    # Select based on resonance type
    is_12h_rez = jnp.abs(irez - 2.0) < 0.5
    is_sync_rez = jnp.abs(irez - 1.0) < 0.5

    d2201_val = jnp.where(is_12h_rez, d2201_12h, 0.0)
    d2211_val = jnp.where(is_12h_rez, d2211_12h, 0.0)
    d3210_val = jnp.where(is_12h_rez, d3210_12h, 0.0)
    d3222_val = jnp.where(is_12h_rez, d3222_12h, 0.0)
    d4410_val = jnp.where(is_12h_rez, d4410_12h, 0.0)
    d4422_val = jnp.where(is_12h_rez, d4422_12h, 0.0)
    d5220_val = jnp.where(is_12h_rez, d5220_12h, 0.0)
    d5232_val = jnp.where(is_12h_rez, d5232_12h, 0.0)
    d5421_val = jnp.where(is_12h_rez, d5421_12h, 0.0)
    d5433_val = jnp.where(is_12h_rez, d5433_12h, 0.0)
    del1_val = jnp.where(is_sync_rez, del1_sync, 0.0)
    del2_val = jnp.where(is_sync_rez, del2_sync, 0.0)
    del3_val = jnp.where(is_sync_rez, del3_sync, 0.0)

    xlamo_val = jnp.where(is_12h_rez, xlamo_12h, jnp.where(is_sync_rez, xlamo_sync, 0.0))
    xfact_val = jnp.where(is_12h_rez, xfact_12h, jnp.where(is_sync_rez, xfact_sync, 0.0))

    # Initialize integrator (only if resonant)
    xli_val = jnp.where(is_resonant, xlamo_val, 0.0)
    xni_val = jnp.where(is_resonant, no, 0.0)
    atime_val = jnp.where(is_resonant, 0.0, 0.0)
    nm = jnp.where(is_resonant, no + dndt, nm)

    return (
        em, argpm, inclm, mm, nm, nodem,
        irez, atime_val,
        d2201_val, d2211_val, d3210_val, d3222_val,
        d4410_val, d4422_val, d5220_val, d5232_val,
        d5421_val, d5433_val,
        dedt, didt, dmdt, dndt, dnodt, domdt,
        del1_val, del2_val, del3_val,
        xfact_val, xlamo_val, xli_val, xni_val,
    )


def _deep_space_init_jax(
    ecco: ArrayLike,
    argpo: ArrayLike,
    inclo: ArrayLike,
    mo: ArrayLike,
    nodeo: ArrayLike,
    no_unkozai: ArrayLike,
    epoch: ArrayLike,
    gsto: ArrayLike,
    mdot: ArrayLike,
    nodedot: ArrayLike,
    xpidot: ArrayLike,
    eccsq: ArrayLike,
    xke: float,
) -> dict[str, Array]:
    """Initialize deep-space parameters (JAX, JIT-compatible).

    Orchestrates _dscom_jax and _dsinit_jax to compute all deep-space
    coefficients needed by the propagator.

    Args:
        ecco: Eccentricity.
        argpo: Argument of perigee [rad].
        inclo: Inclination [rad].
        mo: Mean anomaly [rad].
        nodeo: RAAN [rad].
        no_unkozai: Un-Kozai mean motion [rad/min].
        epoch: Days since 1950 Jan 0.
        gsto: Greenwich sidereal time [rad].
        mdot: Mean motion rate [rad/min].
        nodedot: RAAN rate [rad/min].
        xpidot: argpdot + nodedot.
        eccsq: Eccentricity squared.
        xke: Gravity constant.

    Returns:
        Dictionary mapping parameter names to their computed values.
    """
    tc = 0.0

    # Run dscom
    (
        snodm, cnodm, sinim, cosim, sinomm, cosomm,
        day, e3, ee2, em, emsq, gam,
        peo, pgho, pho, pinco, plo,
        rtemsq_ds,
        se2, se3, sgh2, sgh3, sgh4, sh2, sh3, si2, si3, sl2, sl3, sl4,
        s1, s2, s3, s4, s5, s6, s7,
        ss1, ss2, ss3, ss4, ss5, ss6, ss7,
        sz1, sz2, sz3, sz11, sz12, sz13, sz21, sz22, sz23, sz31, sz32, sz33,
        xgh2, xgh3, xgh4, xh2, xh3, xi2, xi3, xl2, xl3, xl4,
        nm, z1, z2, z3, z11, z12, z13, z21, z22, z23, z31, z32, z33,
        zmol, zmos,
    ) = _dscom_jax(epoch, ecco, argpo, tc, inclo, nodeo, no_unkozai)

    # Run dsinit
    (
        em_ds, argpm_ds, inclm_ds, mm_ds, nm_ds, nodem_ds,
        irez, atime,
        d2201, d2211, d3210, d3222,
        d4410, d4422, d5220, d5232,
        d5421, d5433,
        dedt, didt, dmdt, dndt, dnodt, domdt,
        del1, del2, del3,
        xfact, xlamo, xli, xni,
    ) = _dsinit_jax(
        xke, cosim, emsq, argpo,
        s1, s2, s3, s4, s5, sinim,
        ss1, ss2, ss3, ss4, ss5,
        sz1, sz3, sz11, sz13, sz21, sz23, sz31, sz33,
        0.0, tc, gsto, mo, mdot,
        no_unkozai, nodeo, nodedot, xpidot,
        z1, z3, z11, z13, z21, z23, z31, z33,
        ecco, eccsq, em,
        0.0, inclo, 0.0, nm, 0.0,
    )

    return {
        # From dscom
        "e3": e3, "ee2": ee2,
        "peo": peo, "pgho": pgho, "pho": pho, "pinco": pinco, "plo": plo,
        "se2": se2, "se3": se3,
        "sgh2": sgh2, "sgh3": sgh3, "sgh4": sgh4,
        "sh2": sh2, "sh3": sh3,
        "si2": si2, "si3": si3,
        "sl2": sl2, "sl3": sl3, "sl4": sl4,
        "xgh2": xgh2, "xgh3": xgh3, "xgh4": xgh4,
        "xh2": xh2, "xh3": xh3,
        "xi2": xi2, "xi3": xi3,
        "xl2": xl2, "xl3": xl3, "xl4": xl4,
        "zmol": zmol, "zmos": zmos,
        # From dsinit
        "irez": irez, "atime": atime,
        "d2201": d2201, "d2211": d2211, "d3210": d3210, "d3222": d3222,
        "d4410": d4410, "d4422": d4422, "d5220": d5220, "d5232": d5232,
        "d5421": d5421, "d5433": d5433,
        "dedt": dedt, "didt": didt, "dmdt": dmdt,
        "dnodt": dnodt, "domdt": domdt,
        "del1": del1, "del2": del2, "del3": del3,
        "xfact": xfact, "xlamo": xlamo, "xli": xli, "xni": xni,
    }


# ---------------------------------------------------------------------------
# JAX propagation helpers
# ---------------------------------------------------------------------------


def _dpper_jax(
    params: Array,
    t: ArrayLike,
    ep: ArrayLike,
    inclp: ArrayLike,
    nodep: ArrayLike,
    argpp: ArrayLike,
    mp: ArrayLike,
    idx: dict[str, int],
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Apply deep-space periodic perturbations (JAX, init='n' only).

    Args:
        params: Flat parameter array.
        t: Time since epoch [min].
        ep: Eccentricity.
        inclp: Inclination [rad].
        nodep: RAAN [rad].
        argpp: Argument of perigee [rad].
        mp: Mean anomaly [rad].
        idx: Parameter index mapping.

    Returns:
        Tuple of (ep, inclp, nodep, argpp, mp) with perturbations applied.
    """
    twopi = 2.0 * jnp.pi
    zns = 1.19459e-5
    zes = 0.01675
    znl = 1.5835218e-4
    zel = 0.05490

    # Unpack deep-space periodic params
    e3 = params[idx["e3"]]
    ee2 = params[idx["ee2"]]
    peo = params[idx["peo"]]
    pgho = params[idx["pgho"]]
    pho = params[idx["pho"]]
    pinco = params[idx["pinco"]]
    plo = params[idx["plo"]]
    se2 = params[idx["se2"]]
    se3 = params[idx["se3"]]
    sgh2 = params[idx["sgh2"]]
    sgh3 = params[idx["sgh3"]]
    sgh4 = params[idx["sgh4"]]
    sh2 = params[idx["sh2"]]
    sh3 = params[idx["sh3"]]
    si2 = params[idx["si2"]]
    si3 = params[idx["si3"]]
    sl2 = params[idx["sl2"]]
    sl3 = params[idx["sl3"]]
    sl4 = params[idx["sl4"]]
    xgh2 = params[idx["xgh2"]]
    xgh3 = params[idx["xgh3"]]
    xgh4 = params[idx["xgh4"]]
    xh2 = params[idx["xh2"]]
    xh3 = params[idx["xh3"]]
    xi2 = params[idx["xi2"]]
    xi3 = params[idx["xi3"]]
    xl2 = params[idx["xl2"]]
    xl3 = params[idx["xl3"]]
    xl4 = params[idx["xl4"]]
    zmol = params[idx["zmol"]]
    zmos = params[idx["zmos"]]

    # Calculate time varying periodics (init='n')
    zm = zmos + zns * t
    zf = zm + 2.0 * zes * jnp.sin(zm)
    sinzf = jnp.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * jnp.cos(zf)
    ses = se2 * f2 + se3 * f3
    sis = si2 * f2 + si3 * f3
    sls = sl2 * f2 + sl3 * f3 + sl4 * sinzf
    sghs = sgh2 * f2 + sgh3 * f3 + sgh4 * sinzf
    shs = sh2 * f2 + sh3 * f3

    zm = zmol + znl * t
    zf = zm + 2.0 * zel * jnp.sin(zm)
    sinzf = jnp.sin(zf)
    f2 = 0.5 * sinzf * sinzf - 0.25
    f3 = -0.5 * sinzf * jnp.cos(zf)
    sel = ee2 * f2 + e3 * f3
    sil = xi2 * f2 + xi3 * f3
    sll = xl2 * f2 + xl3 * f3 + xl4 * sinzf
    sghl = xgh2 * f2 + xgh3 * f3 + xgh4 * sinzf
    shll = xh2 * f2 + xh3 * f3

    pe = ses + sel - peo
    pinc = sis + sil - pinco
    pl = sls + sll - plo
    pgh = sghs + sghl - pgho
    ph = shs + shll - pho

    inclp = inclp + pinc
    ep = ep + pe
    sinip = jnp.sin(inclp)
    cosip = jnp.cos(inclp)

    # Apply periodics: direct or Lyddane modification
    # Direct application (inclp >= 0.2 rad)
    ph_direct = ph / sinip
    pgh_direct = pgh - cosip * ph_direct
    argpp_direct = argpp + pgh_direct
    nodep_direct = nodep + ph_direct
    mp_direct = mp + pl

    # Lyddane modification (inclp < 0.2 rad)
    sinop = jnp.sin(nodep)
    cosop = jnp.cos(nodep)
    alfdp = sinip * sinop
    betdp = sinip * cosop
    dalf = ph * cosop + pinc * cosip * sinop
    dbet = -ph * sinop + pinc * cosip * cosop
    alfdp = alfdp + dalf
    betdp = betdp + dbet
    # Normalize nodep
    nodep_lyd = nodep % twopi
    xls = mp + argpp + pl + pgh + (cosip - pinc * sinip) * nodep_lyd
    xnoh = nodep_lyd
    nodep_lyd = jnp.arctan2(alfdp, betdp)
    # Fix quadrant
    nodep_lyd = jnp.where(
        jnp.abs(xnoh - nodep_lyd) > jnp.pi,
        jnp.where(nodep_lyd < xnoh, nodep_lyd + twopi, nodep_lyd - twopi),
        nodep_lyd,
    )
    mp_lyd = mp + pl
    argpp_lyd = xls - mp_lyd - cosip * nodep_lyd

    # Select based on inclination
    use_direct = inclp >= 0.2
    argpp = jnp.where(use_direct, argpp_direct, argpp_lyd)
    nodep = jnp.where(use_direct, nodep_direct, nodep_lyd)
    mp = jnp.where(use_direct, mp_direct, mp_lyd)

    return ep, inclp, nodep, argpp, mp


def _dspace_jax(
    params: Array,
    t: ArrayLike,
    tc: ArrayLike,
    em: ArrayLike,
    argpm: ArrayLike,
    inclm: ArrayLike,
    mm: ArrayLike,
    nodem: ArrayLike,
    nm: ArrayLike,
    idx: dict[str, int],
) -> tuple[
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    """Numerical integration of deep-space resonance effects (JAX).

    Args:
        params: Flat parameter array.
        t: Time since epoch [min].
        tc: Same as t (for compatibility).
        em: Eccentricity.
        argpm: Argument of perigee [rad].
        inclm: Inclination [rad].
        mm: Mean anomaly [rad].
        nodem: RAAN [rad].
        nm: Mean motion [rad/min].
        idx: Parameter index mapping.

    Returns:
        Tuple of (em, argpm, inclm, mm, xni, nodem, dndt, nm).
    """
    twopi = 2.0 * jnp.pi
    fasx2 = 0.13130908
    fasx4 = 2.8843198
    fasx6 = 0.37448087
    g22 = 5.7686396
    g32 = 0.95240898
    g44 = 1.8014998
    g52 = 1.0508330
    g54 = 4.4108898
    rptim = 4.37526908801129966e-3
    stepp = 720.0
    step2 = 259200.0

    irez = params[idx["irez"]]
    d2201 = params[idx["d2201"]]
    d2211 = params[idx["d2211"]]
    d3210 = params[idx["d3210"]]
    d3222 = params[idx["d3222"]]
    d4410 = params[idx["d4410"]]
    d4422 = params[idx["d4422"]]
    d5220 = params[idx["d5220"]]
    d5232 = params[idx["d5232"]]
    d5421 = params[idx["d5421"]]
    d5433 = params[idx["d5433"]]
    dedt = params[idx["dedt"]]
    del1 = params[idx["del1"]]
    del2 = params[idx["del2"]]
    del3 = params[idx["del3"]]
    didt = params[idx["didt"]]
    dmdt = params[idx["dmdt"]]
    dnodt = params[idx["dnodt"]]
    domdt = params[idx["domdt"]]
    argpo = params[idx["argpo"]]
    argpdot = params[idx["argpdot"]]
    gsto = params[idx["gsto"]]
    xfact = params[idx["xfact"]]
    xlamo = params[idx["xlamo"]]
    no = params[idx["no_unkozai"]]

    dndt = jnp.float64(0.0) if params.dtype == jnp.float64 else jnp.float32(0.0)
    theta = (gsto + tc * rptim) % twopi
    em = em + dedt * t
    inclm = inclm + didt * t
    argpm = argpm + domdt * t
    nodem = nodem + dnodt * t
    mm = mm + dmdt * t

    # Resonance integration (only if irez != 0)
    is_resonant = jnp.abs(irez) > 0.5

    # Initialize integration state
    atime_init = jnp.zeros_like(t)
    xni_init = no
    xli_init = xlamo

    delt = jnp.where(t > 0.0, stepp, -stepp)

    def _compute_dot_terms(xli_val, xni_val, atime_val):
        """Compute xndt, xldot, xnddt for both resonance types."""
        # Synchronous (irez == 1)
        xndt_sync = (
            del1 * jnp.sin(xli_val - fasx2)
            + del2 * jnp.sin(2.0 * (xli_val - fasx4))
            + del3 * jnp.sin(3.0 * (xli_val - fasx6))
        )
        xldot_sync = xni_val + xfact
        xnddt_sync = (
            del1 * jnp.cos(xli_val - fasx2)
            + 2.0 * del2 * jnp.cos(2.0 * (xli_val - fasx4))
            + 3.0 * del3 * jnp.cos(3.0 * (xli_val - fasx6))
        )
        xnddt_sync = xnddt_sync * xldot_sync

        # Half-day (irez == 2)
        xomi = argpo + argpdot * atime_val
        x2omi = xomi + xomi
        x2li = xli_val + xli_val
        xndt_hd = (
            d2201 * jnp.sin(x2omi + xli_val - g22)
            + d2211 * jnp.sin(xli_val - g22)
            + d3210 * jnp.sin(xomi + xli_val - g32)
            + d3222 * jnp.sin(-xomi + xli_val - g32)
            + d4410 * jnp.sin(x2omi + x2li - g44)
            + d4422 * jnp.sin(x2li - g44)
            + d5220 * jnp.sin(xomi + xli_val - g52)
            + d5232 * jnp.sin(-xomi + xli_val - g52)
            + d5421 * jnp.sin(xomi + x2li - g54)
            + d5433 * jnp.sin(-xomi + x2li - g54)
        )
        xldot_hd = xni_val + xfact
        xnddt_hd = (
            d2201 * jnp.cos(x2omi + xli_val - g22)
            + d2211 * jnp.cos(xli_val - g22)
            + d3210 * jnp.cos(xomi + xli_val - g32)
            + d3222 * jnp.cos(-xomi + xli_val - g32)
            + d5220 * jnp.cos(xomi + xli_val - g52)
            + d5232 * jnp.cos(-xomi + xli_val - g52)
            + 2.0
            * (
                d4410 * jnp.cos(x2omi + x2li - g44)
                + d4422 * jnp.cos(x2li - g44)
                + d5421 * jnp.cos(xomi + x2li - g54)
                + d5433 * jnp.cos(-xomi + x2li - g54)
            )
        )
        xnddt_hd = xnddt_hd * xldot_hd

        is_half_day = jnp.abs(irez - 2.0) < 0.5
        xndt = jnp.where(is_half_day, xndt_hd, xndt_sync)
        xldot = jnp.where(is_half_day, xldot_hd, xldot_sync)
        xnddt = jnp.where(is_half_day, xnddt_hd, xnddt_sync)
        return xndt, xldot, xnddt

    # Integration loop using while_loop
    def _loop_cond(state):
        atime_s, xni_s, xli_s = state
        return jnp.abs(t - atime_s) >= stepp

    def _loop_body(state):
        atime_s, xni_s, xli_s = state
        xndt, xldot, xnddt = _compute_dot_terms(xli_s, xni_s, atime_s)
        xli_s = xli_s + xldot * delt + xndt * step2
        xni_s = xni_s + xndt * delt + xnddt * step2
        atime_s = atime_s + delt
        return (atime_s, xni_s, xli_s)

    init_state = (atime_init, xni_init, xli_init)
    final_state = jax.lax.while_loop(_loop_cond, _loop_body, init_state)
    atime_f, xni_f, xli_f = final_state

    # Final interpolation to exact time t
    ft = t - atime_f
    xndt, xldot, xnddt = _compute_dot_terms(xli_f, xni_f, atime_f)

    nm_res = xni_f + xndt * ft + xnddt * ft * ft * 0.5
    xl = xli_f + xldot * ft + xndt * ft * ft * 0.5

    # Mean anomaly update (different for irez==1 vs irez==2)
    is_sync = jnp.abs(irez - 1.0) < 0.5
    mm_sync = xl - nodem - argpm + theta
    mm_half = xl - 2.0 * nodem + 2.0 * theta
    mm_res = jnp.where(is_sync, mm_sync, mm_half)

    dndt_res = nm_res - no
    nm_res = no + dndt_res

    # Apply resonance results only if resonant
    nm = jnp.where(is_resonant, nm_res, nm)
    mm = jnp.where(is_resonant, mm_res, mm)
    dndt = jnp.where(is_resonant, dndt_res, dndt)

    return em, argpm, inclm, mm, xni_f, nodem, dndt, nm


# ---------------------------------------------------------------------------
# Full deep-space propagation
# ---------------------------------------------------------------------------


def sgp4_propagate_deep_space_impl(
    params: Array,
    tsince: ArrayLike,
    idx: dict[str, int],
) -> tuple[Array, Array]:
    """Propagate deep-space satellite using SDP4 (JAX, JIT-compatible).

    This is the full propagation function for deep-space satellites,
    including secular updates, deep-space resonance integration,
    periodic perturbations, Kepler iteration, and output.

    Args:
        params: Flat parameter array from ``sgp4_init``.
        tsince: Time since epoch in minutes.
        idx: Parameter index mapping.

    Returns:
        Tuple of ``(r, v)`` where ``r`` is position [km] and ``v`` is
        velocity [km/s], both as 3-element arrays in the TEME frame.
    """
    twopi = 2.0 * jnp.pi
    x2o3 = 2.0 / 3.0
    temp4 = 1.5e-12

    p = params
    radiusearthkm = p[idx["radiusearthkm"]]
    xke = p[idx["xke"]]
    j2 = p[idx["j2"]]
    j3oj2 = p[idx["j3oj2"]]
    bstar = p[idx["bstar"]]
    ecco = p[idx["ecco"]]
    argpo = p[idx["argpo"]]
    inclo = p[idx["inclo"]]
    mo = p[idx["mo"]]
    nodeo = p[idx["nodeo"]]
    no_unkozai = p[idx["no_unkozai"]]
    con41 = p[idx["con41"]]
    cc1 = p[idx["cc1"]]
    cc4 = p[idx["cc4"]]
    argpdot = p[idx["argpdot"]]
    t2cof = p[idx["t2cof"]]
    mdot = p[idx["mdot"]]
    nodedot = p[idx["nodedot"]]
    nodecf = p[idx["nodecf"]]
    aycof = p[idx["aycof"]]

    vkmpersec = radiusearthkm * xke / 60.0
    t = tsince

    # --- Secular gravity and atmospheric drag ---
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

    # isimp is always 1 for deep space, so skip non-simplified corrections

    nm = no_unkozai
    em = ecco
    inclm = inclo

    # --- Deep space: _dspace ---
    tc = t
    (
        em,
        argpm,
        inclm,
        mm,
        xni,
        nodem,
        dndt,
        nm,
    ) = _dspace_jax(params, t, tc, em, argpm, inclm, mm, nodem, nm, idx)

    # Error check: nm <= 0
    nm_ok = nm > 0.0

    am = (xke / nm) ** x2o3 * tempa * tempa
    nm = xke / am**1.5
    em = em - tempe

    em_ok = (em < 1.0) & (em >= -0.001)
    em = jnp.clip(em, 1.0e-6, 0.999999)

    mm = mm + no_unkozai * templ
    xlm = mm + argpm + nodem

    nodem = nodem % twopi
    argpm = argpm % twopi
    xlm = xlm % twopi
    mm = (xlm - argpm - nodem) % twopi

    # --- Deep space: _dpper ---
    ep = em
    xincp = inclm
    argpp = argpm
    nodep = nodem
    mp = mm

    ep, xincp, nodep, argpp, mp = _dpper_jax(params, t, ep, xincp, nodep, argpp, mp, idx)

    # Fix negative inclination
    xincp = jnp.where(xincp < 0.0, -xincp, xincp)
    nodep = jnp.where(xincp < 0.0, nodep + jnp.pi, nodep)
    argpp = jnp.where(xincp < 0.0, argpp - jnp.pi, argpp)

    ep_ok = (ep >= 0.0) & (ep <= 1.0)

    # Recompute sinip/cosip for deep space
    sinip = jnp.sin(xincp)
    cosip = jnp.cos(xincp)

    # Recompute aycof and xlcof for deep space
    aycof = -0.5 * j3oj2 * sinip
    xlcof_ds = jnp.where(
        jnp.abs(cosip + 1.0) > 1.5e-12,
        -0.25 * j3oj2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip),
        -0.25 * j3oj2 * sinip * (3.0 + 5.0 * cosip) / temp4,
    )

    # --- Long period periodics ---
    axnl = ep * jnp.cos(argpp)
    temp_lp = 1.0 / (am * (1.0 - ep * ep))
    aynl = ep * jnp.sin(argpp) + temp_lp * aycof
    xl = mp + argpp + nodep + temp_lp * xlcof_ds * axnl

    # --- Kepler's equation ---
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

    # --- Short period quantities ---
    ecose = axnl * coseo1 + aynl * sineo1
    esine = axnl * sineo1 - aynl * coseo1
    el2 = axnl * axnl + aynl * aynl
    pl = am * (1.0 - el2)
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

    # For deep space, recompute con41, x1mth2, x7thm1
    cosisq = cosip * cosip
    con41 = 3.0 * cosisq - 1.0
    x1mth2 = 1.0 - cosisq
    x7thm1 = 7.0 * cosisq - 1.0

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

    _mr = mrt * radiusearthkm
    r = jnp.array([_mr * ux, _mr * uy, _mr * uz])
    v = jnp.array(
        [
            (mvt * ux + rvdot * vx) * vkmpersec,
            (mvt * uy + rvdot * vy) * vkmpersec,
            (mvt * uz + rvdot * vz) * vkmpersec,
        ]
    )

    valid = nm_ok & em_ok & ep_ok & pl_ok & (mrt >= 1.0)
    nan3 = jnp.full(3, jnp.nan)
    r = jnp.where(valid, r, nan3)
    v = jnp.where(valid, v, nan3)

    return r, v
