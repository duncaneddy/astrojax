"""Keplerian orbital elements to Quasi-Nonsingular Relative Orbital Elements (ROE) conversions.

Provides transformations between chief/deputy Keplerian orbital elements
``[a, e, i, RAAN, omega, M]`` and Quasi-Nonsingular Relative Orbital
Elements ``[da, dλ, dex, dey, dix, diy]``.

The ROE vector components are:

| Index | Element                                    | Units         |
|-------|--------------------------------------------|---------------|
| 0     | *da* — relative semi-major axis            | dimensionless |
| 1     | *dλ* — relative mean longitude             | rad (or deg)  |
| 2     | *dex* — relative eccentricity x-component  | dimensionless |
| 3     | *dey* — relative eccentricity y-component  | dimensionless |
| 4     | *dix* — relative inclination x-component   | rad (or deg)  |
| 5     | *diy* — relative inclination y-component   | rad (or deg)  |

References:
    1. Sullivan, J. "Nonlinear Angles-Only Orbit Estimation for
       Autonomous Distributed Space Systems", 2020.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype


def _wrap_to_2pi(angle: Array) -> Array:
    """Wrap an angle to the interval [0, 2π)."""
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=get_dtype())
    return jnp.mod(angle, two_pi)


def state_oe_to_roe(
    oe_chief: ArrayLike,
    oe_deputy: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Compute Relative Orbital Elements (ROE) from chief and deputy orbital elements.

    Converts the Keplerian orbital elements of a chief and deputy
    satellite pair into quasi-nonsingular Relative Orbital Elements.

    Args:
        oe_chief: Chief satellite orbital elements
            ``[a, e, i, RAAN, omega, M]``.
            Semi-major axis in *m*, angles in *rad* (or *deg* if
            ``use_degrees=True``).
        oe_deputy: Deputy satellite orbital elements
            ``[a, e, i, RAAN, omega, M]``.
            Semi-major axis in *m*, angles in *rad* (or *deg* if
            ``use_degrees=True``).
        use_degrees: If ``True``, interpret angular inputs as degrees
            and return angular ROE components in degrees.

    Returns:
        Relative Orbital Elements ``[da, dλ, dex, dey, dix, diy]``.
            ``da`` is dimensionless, ``dex``/``dey`` are dimensionless,
            ``dλ``/``dix``/``diy`` are in *rad* (or *deg*).

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH
        from astrojax.relative_motion import state_oe_to_roe
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])
        roe = state_oe_to_roe(oe_c, oe_d, use_degrees=True)
        roe.shape
        ```

    References:
        Sullivan, J. "Nonlinear Angles-Only Orbit Estimation for
        Autonomous Distributed Space Systems", 2020.
    """
    oe_chief = jnp.asarray(oe_chief, dtype=get_dtype())
    oe_deputy = jnp.asarray(oe_deputy, dtype=get_dtype())

    ac = oe_chief[0]
    ec = oe_chief[1]
    ic = oe_chief[2]
    raan_c = oe_chief[3]
    omega_c = oe_chief[4]
    m_c = oe_chief[5]

    ad = oe_deputy[0]
    ed = oe_deputy[1]
    id_ = oe_deputy[2]
    raan_d = oe_deputy[3]
    omega_d = oe_deputy[4]
    m_d = oe_deputy[5]

    if use_degrees:
        ic = jnp.deg2rad(ic)
        raan_c = jnp.deg2rad(raan_c)
        omega_c = jnp.deg2rad(omega_c)
        m_c = jnp.deg2rad(m_c)
        id_ = jnp.deg2rad(id_)
        raan_d = jnp.deg2rad(raan_d)
        omega_d = jnp.deg2rad(omega_d)
        m_d = jnp.deg2rad(m_d)

    # Argument of latitude
    uc = m_c + omega_c
    ud = m_d + omega_d

    # Relative semi-major axis (dimensionless)
    da = (ad - ac) / ac

    # Relative mean longitude
    d_lambda = (ud - uc) + (raan_d - raan_c) * jnp.cos(ic)

    # Relative eccentricity vector
    dex = ed * jnp.cos(omega_d) - ec * jnp.cos(omega_c)
    dey = ed * jnp.sin(omega_d) - ec * jnp.sin(omega_c)

    # Relative inclination vector
    dix = id_ - ic
    diy = (raan_d - raan_c) * jnp.sin(ic)

    # Wrap angular quantities to [0, 2π)
    d_lambda = _wrap_to_2pi(d_lambda)
    dix = _wrap_to_2pi(dix)
    diy = _wrap_to_2pi(diy)

    if use_degrees:
        d_lambda = jnp.rad2deg(d_lambda)
        dix = jnp.rad2deg(dix)
        diy = jnp.rad2deg(diy)

    return jnp.array([da, d_lambda, dex, dey, dix, diy])


def state_roe_to_oe(
    oe_chief: ArrayLike,
    roe: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Compute deputy orbital elements from chief OE and Relative Orbital Elements.

    Inverts the ROE transformation to recover the deputy satellite's
    Keplerian orbital elements given the chief's elements and the ROE.

    Args:
        oe_chief: Chief satellite orbital elements
            ``[a, e, i, RAAN, omega, M]``.
            Semi-major axis in *m*, angles in *rad* (or *deg* if
            ``use_degrees=True``).
        roe: Relative Orbital Elements ``[da, dλ, dex, dey, dix, diy]``.
            ``da`` is dimensionless, ``dex``/``dey`` are dimensionless,
            ``dλ``/``dix``/``diy`` are in *rad* (or *deg* if
            ``use_degrees=True``).
        use_degrees: If ``True``, interpret angular inputs as degrees
            and return angular elements in degrees.

    Returns:
        Deputy orbital elements ``[a, e, i, RAAN, omega, M]``.
            Semi-major axis in *m*, angles in *rad* (or *deg*).

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH
        from astrojax.relative_motion import state_oe_to_roe, state_roe_to_oe
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])
        roe = state_oe_to_roe(oe_c, oe_d, use_degrees=True)
        oe_d_recovered = state_roe_to_oe(oe_c, roe, use_degrees=True)
        oe_d_recovered.shape
        ```

    References:
        Sullivan, J. "Nonlinear Angles-Only Orbit Estimation for
        Autonomous Distributed Space Systems", 2020.
    """
    oe_chief = jnp.asarray(oe_chief, dtype=get_dtype())
    roe = jnp.asarray(roe, dtype=get_dtype())

    # Chief OE
    ac = oe_chief[0]
    ec = oe_chief[1]
    ic = oe_chief[2]
    raan_c = oe_chief[3]
    omega_c = oe_chief[4]
    m_c = oe_chief[5]

    if use_degrees:
        ic = jnp.deg2rad(ic)
        raan_c = jnp.deg2rad(raan_c)
        omega_c = jnp.deg2rad(omega_c)
        m_c = jnp.deg2rad(m_c)

    # ROE components
    da = roe[0]
    d_lambda = roe[1]
    dex = roe[2]
    dey = roe[3]
    dix = roe[4]
    diy = roe[5]

    if use_degrees:
        d_lambda = jnp.deg2rad(d_lambda)
        dix = jnp.deg2rad(dix)
        diy = jnp.deg2rad(diy)

    # Compute deputy OE
    ad = ac * (1.0 + da)
    ed = jnp.sqrt(
        (dex + ec * jnp.cos(omega_c)) ** 2
        + (dey + ec * jnp.sin(omega_c)) ** 2
    )
    i_dep = dix + ic
    raan_d = raan_c + diy / jnp.sin(ic)
    omega_d = jnp.arctan2(
        dey + ec * jnp.sin(omega_c),
        dex + ec * jnp.cos(omega_c),
    )
    m_d = d_lambda - omega_d + m_c + omega_c - (raan_d - raan_c) * jnp.cos(ic)

    # Wrap angles to [0, 2π)
    raan_d = _wrap_to_2pi(raan_d)
    omega_d = _wrap_to_2pi(omega_d)
    i_dep = _wrap_to_2pi(i_dep)
    m_d = _wrap_to_2pi(m_d)

    if use_degrees:
        i_dep = jnp.rad2deg(i_dep)
        raan_d = jnp.rad2deg(raan_d)
        omega_d = jnp.rad2deg(omega_d)
        m_d = jnp.rad2deg(m_d)

    return jnp.array([ad, ed, i_dep, raan_d, omega_d, m_d])
