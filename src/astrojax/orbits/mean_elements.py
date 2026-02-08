"""Mean-osculating Keplerian element conversions.

First-order J2 perturbation mapping based on Brouwer-Lyddane theory.
Implements the algorithm from Schaub and Junkins, *Analytical Mechanics
of Space Systems*, Appendix F: "First-Order Mapping Between Mean and
Osculating Orbit Elements".

The transformation uses a sign convention on the perturbation parameter
gamma_2 to handle both directions (mean-to-osculating and
osculating-to-mean) with a single code path, keeping the implementation
fully JAX-traceable with no Python control flow.

All functions use JAX operations and are compatible with ``jax.jit``,
``jax.vmap``, and ``jax.grad``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.constants import J2_EARTH, R_EARTH
from astrojax.orbits.keplerian import (
    anomaly_eccentric_to_true,
    anomaly_mean_to_eccentric,
)
from astrojax.utils import from_radians, to_radians


def _transform_koe(oe: Array, sign: float) -> Array:
    """Apply the first-order Brouwer-Lyddane J2 transformation.

    Implements equations F.1-F.22 from Schaub & Junkins Appendix F.
    The sign parameter controls the direction: ``+1.0`` for
    mean-to-osculating, ``-1.0`` for osculating-to-mean.

    Args:
        oe: Keplerian elements ``[a, e, i, Omega, omega, M]`` in
            metres and radians.
        sign: ``+1.0`` for mean-to-osculating, ``-1.0`` for
            osculating-to-mean.

    Returns:
        Transformed Keplerian elements in metres and radians.
    """
    a = oe[0]
    e = oe[1]
    i = oe[2]
    raan = oe[3]
    argp = oe[4]
    m_anom = oe[5]

    # (F.1/F.2) gamma_2 = sign * (J2/2) * (R_earth/a)^2
    gamma2 = sign * (J2_EARTH / 2.0) * (R_EARTH / a) ** 2

    # eta = sqrt(1 - e^2)
    eta = jnp.sqrt(1.0 - e * e)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2
    eta6 = eta4 * eta2

    # (F.3) gamma'_2 = gamma_2 / eta^4
    gamma2_prime = gamma2 / eta4

    # (F.4) Solve Kepler's equation for eccentric anomaly E
    e_anom = anomaly_mean_to_eccentric(m_anom, e)

    # (F.5) True anomaly f
    f = anomaly_eccentric_to_true(e_anom, e)

    # (F.6) a/r = (1 + e*cos(f)) / eta^2
    a_over_r = (1.0 + e * jnp.cos(f)) / eta2

    # Precompute trig terms
    cos_i = jnp.cos(i)
    cos2_i = cos_i * cos_i
    cos4_i = cos2_i * cos2_i
    cos6_i = cos4_i * cos2_i

    cos_f = jnp.cos(f)
    sin_f = jnp.sin(f)
    cos2_f = cos_f * cos_f
    cos3_f = cos2_f * cos_f

    two_argp = 2.0 * argp
    cos_2argp = jnp.cos(two_argp)
    cos_2argp_f = jnp.cos(two_argp + f)
    sin_2argp_f = jnp.sin(two_argp + f)
    cos_2argp_2f = jnp.cos(two_argp + 2.0 * f)
    sin_2argp_2f = jnp.sin(two_argp + 2.0 * f)
    cos_2argp_3f = jnp.cos(two_argp + 3.0 * f)
    sin_2argp_3f = jnp.sin(two_argp + 3.0 * f)

    a_over_r_cubed = a_over_r ** 3

    # (1 - 5*cos^2(i)) term
    one_minus_5cos2_i = 1.0 - 5.0 * cos2_i
    one_minus_5cos2_i_sq = one_minus_5cos2_i * one_minus_5cos2_i

    # ── (F.7) Semi-major axis ──
    a_prime = a + a * gamma2 * (
        (3.0 * cos2_i - 1.0) * (a_over_r_cubed - 1.0 / eta3)
        + 3.0 * (1.0 - cos2_i) * a_over_r_cubed * cos_2argp_2f
    )

    # ── (F.8) delta_e1 ──
    delta_e1 = (gamma2_prime / 8.0) * e * eta2 * (
        1.0 - 11.0 * cos2_i - 40.0 * cos4_i / one_minus_5cos2_i
    ) * cos_2argp

    # ── (F.9) delta_e ──
    de_inner1 = ((3.0 * cos2_i - 1.0) / eta6) * (
        e * eta + e / (1.0 + eta) + 3.0 * cos_f
        + 3.0 * e * cos2_f + e * e * cos3_f
    )
    de_inner2 = (
        3.0 * ((1.0 - cos2_i) / eta6)
        * (e + 3.0 * cos_f + 3.0 * e * cos2_f + e * e * cos3_f)
        * cos_2argp_2f
    )
    de_bracket = gamma2 * (de_inner1 + de_inner2)
    de_third_term = (
        -gamma2_prime * (1.0 - cos2_i) * (3.0 * cos_2argp_f + cos_2argp_3f)
    )
    delta_e = delta_e1 + (eta2 / 2.0) * (de_bracket + de_third_term)

    # ── (F.10) delta_i ──
    sin_i = jnp.sqrt(1.0 - cos2_i)
    delta_i = -(e * delta_e1) / (eta2 * jnp.tan(i)) + (gamma2_prime / 2.0) * cos_i * sin_i * (
        3.0 * cos_2argp_2f + 3.0 * e * cos_2argp_f + e * cos_2argp_3f
    )

    # ── (F.11) M' + omega' + Omega' combined ──
    mpo_line1 = (
        m_anom + argp + raan
        + (gamma2_prime / 8.0) * eta3
        * (1.0 - 11.0 * cos2_i - 40.0 * cos4_i / one_minus_5cos2_i)
    )
    mpo_line2 = -(gamma2_prime / 16.0) * (
        2.0 + e * e
        - 11.0 * (2.0 + 3.0 * e * e) * cos2_i
        - 40.0 * (2.0 + 5.0 * e * e) * cos4_i / one_minus_5cos2_i
        - 400.0 * e * e * cos6_i / one_minus_5cos2_i_sq
    )
    mpo_line3 = (gamma2_prime / 4.0) * (
        -6.0 * one_minus_5cos2_i * (f - m_anom + e * sin_f)
        + (3.0 - 5.0 * cos2_i)
        * (3.0 * sin_2argp_2f + 3.0 * e * sin_2argp_f + e * sin_2argp_3f)
    )
    mpo_line4 = -(gamma2_prime / 8.0) * e * e * cos_i * (
        11.0 + 80.0 * cos2_i / one_minus_5cos2_i
        + 200.0 * cos4_i / one_minus_5cos2_i_sq
    )
    mpo_line5 = -(gamma2_prime / 2.0) * cos_i * (
        6.0 * (f - m_anom + e * sin_f)
        - 3.0 * sin_2argp_2f
        - 3.0 * e * sin_2argp_f
        - e * sin_2argp_3f
    )
    m_prime_plus_argp_prime_plus_raan_prime = (
        mpo_line1 + mpo_line2 + mpo_line3 + mpo_line4 + mpo_line5
    )

    # ── (F.12) e * delta_M ──
    aeta_over_r = a_over_r * eta
    aeta_over_r_sq = aeta_over_r * aeta_over_r

    edm_line1 = (gamma2_prime / 8.0) * e * eta3 * (
        1.0 - 11.0 * cos2_i - 40.0 * cos4_i / one_minus_5cos2_i
    )
    edm_term1 = (
        2.0 * (3.0 * cos2_i - 1.0) * (aeta_over_r_sq + a_over_r + 1.0) * sin_f
    )
    edm_term2 = 3.0 * (1.0 - cos2_i) * (
        (-aeta_over_r_sq - a_over_r + 1.0) * sin_2argp_f
        + (aeta_over_r_sq + a_over_r + 1.0 / 3.0) * sin_2argp_3f
    )
    e_delta_m = edm_line1 - (gamma2_prime / 4.0) * eta3 * (edm_term1 + edm_term2)

    # ── (F.13) delta_Omega ──
    do_line1 = -(gamma2_prime / 8.0) * e * e * cos_i * (
        11.0 + 80.0 * cos2_i / one_minus_5cos2_i
        + 200.0 * cos4_i / one_minus_5cos2_i_sq
    )
    do_line2 = -(gamma2_prime / 2.0) * cos_i * (
        6.0 * (f - m_anom + e * sin_f)
        - 3.0 * sin_2argp_2f
        - 3.0 * e * sin_2argp_f
        - e * sin_2argp_3f
    )
    delta_raan = do_line1 + do_line2

    # ── Final recovery (F.14-F.22) ──

    # (F.14) d1 = (e + delta_e) * sin(M) + (e*delta_M) * cos(M)
    d1 = (e + delta_e) * jnp.sin(m_anom) + e_delta_m * jnp.cos(m_anom)

    # (F.15) d2 = (e + delta_e) * cos(M) - (e*delta_M) * sin(M)
    d2 = (e + delta_e) * jnp.cos(m_anom) - e_delta_m * jnp.sin(m_anom)

    # (F.16) M' = atan2(d1, d2)
    m_prime = jnp.arctan2(d1, d2)

    # (F.17) e' = sqrt(d1^2 + d2^2)
    e_prime = jnp.sqrt(d1 * d1 + d2 * d2)

    # (F.18-F.19) d3, d4 for inclination and RAAN recovery
    half_i = i / 2.0
    sin_half_i = jnp.sin(half_i)
    cos_half_i = jnp.cos(half_i)
    d3 = (
        (sin_half_i + cos_half_i * delta_i / 2.0) * jnp.sin(raan)
        + sin_half_i * delta_raan * jnp.cos(raan)
    )
    d4 = (
        (sin_half_i + cos_half_i * delta_i / 2.0) * jnp.cos(raan)
        - sin_half_i * delta_raan * jnp.sin(raan)
    )

    # (F.20) Omega' = atan2(d3, d4)
    raan_prime = jnp.arctan2(d3, d4)

    # (F.21) i' = 2 * asin(sqrt(d3^2 + d4^2))
    i_prime = 2.0 * jnp.arcsin(jnp.sqrt(d3 * d3 + d4 * d4))

    # (F.22) omega' = (M' + omega' + Omega') - M' - Omega'
    argp_prime_raw = m_prime_plus_argp_prime_plus_raan_prime - m_prime - raan_prime

    # Normalize angles to [0, 2*pi)
    two_pi = 2.0 * jnp.pi
    m_prime_norm = (m_prime % two_pi + two_pi) % two_pi
    raan_prime_norm = (raan_prime % two_pi + two_pi) % two_pi
    argp_prime_norm = (argp_prime_raw % two_pi + two_pi) % two_pi

    return jnp.array([
        a_prime,
        e_prime,
        i_prime,
        raan_prime_norm,
        argp_prime_norm,
        m_prime_norm,
    ])


def state_koe_osc_to_mean(oe: ArrayLike, use_degrees: bool = False) -> Array:
    """Convert osculating Keplerian elements to mean Keplerian elements.

    Applies the first-order Brouwer-Lyddane transformation to convert
    osculating (instantaneous) orbital elements to mean (orbit-averaged)
    elements. The transformation accounts for short-period and
    long-period J2 perturbations.

    Args:
        oe: Osculating Keplerian elements ``[a, e, i, Omega, omega, M]``.
            Semi-major axis in metres, eccentricity dimensionless,
            angles in radians or degrees.
        use_degrees: If ``True``, angular elements are in degrees.

    Returns:
        Mean Keplerian elements in the same format as input.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH
        from astrojax.orbits import state_koe_osc_to_mean

        osc = jnp.array([R_EARTH + 500e3, 0.001, 45.0, 0.0, 0.0, 0.0])
        mean = state_koe_osc_to_mean(osc, use_degrees=True)
        ```
    """
    oe = jnp.asarray(oe, dtype=get_dtype())
    oe_rad = jnp.array([
        oe[0],
        oe[1],
        to_radians(oe[2], use_degrees),
        to_radians(oe[3], use_degrees),
        to_radians(oe[4], use_degrees),
        to_radians(oe[5], use_degrees),
    ])
    result_rad = _transform_koe(oe_rad, -1.0)
    return jnp.array([
        result_rad[0],
        result_rad[1],
        from_radians(result_rad[2], use_degrees),
        from_radians(result_rad[3], use_degrees),
        from_radians(result_rad[4], use_degrees),
        from_radians(result_rad[5], use_degrees),
    ])


def state_koe_mean_to_osc(oe: ArrayLike, use_degrees: bool = False) -> Array:
    """Convert mean Keplerian elements to osculating Keplerian elements.

    Applies the first-order Brouwer-Lyddane transformation to convert
    mean (orbit-averaged) orbital elements to osculating (instantaneous)
    elements. The transformation accounts for short-period and
    long-period J2 perturbations.

    Args:
        oe: Mean Keplerian elements ``[a, e, i, Omega, omega, M]``.
            Semi-major axis in metres, eccentricity dimensionless,
            angles in radians or degrees.
        use_degrees: If ``True``, angular elements are in degrees.

    Returns:
        Osculating Keplerian elements in the same format as input.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH
        from astrojax.orbits import state_koe_mean_to_osc

        mean = jnp.array([R_EARTH + 500e3, 0.001, 45.0, 0.0, 0.0, 0.0])
        osc = state_koe_mean_to_osc(mean, use_degrees=True)
        ```
    """
    oe = jnp.asarray(oe, dtype=get_dtype())
    oe_rad = jnp.array([
        oe[0],
        oe[1],
        to_radians(oe[2], use_degrees),
        to_radians(oe[3], use_degrees),
        to_radians(oe[4], use_degrees),
        to_radians(oe[5], use_degrees),
    ])
    result_rad = _transform_koe(oe_rad, +1.0)
    return jnp.array([
        result_rad[0],
        result_rad[1],
        from_radians(result_rad[2], use_degrees),
        from_radians(result_rad[3], use_degrees),
        from_radians(result_rad[4], use_degrees),
        from_radians(result_rad[5], use_degrees),
    ])
