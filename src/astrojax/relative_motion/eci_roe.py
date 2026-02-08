"""ECI to Quasi-Nonsingular Relative Orbital Elements (ROE) conversions.

Provides direct transformations between Earth-Centered Inertial (ECI)
state vectors and Relative Orbital Elements (ROE), combining the
ECI↔KOE and OE↔ROE transformations for convenience.

References:
    1. Sullivan, J. "Nonlinear Angles-Only Orbit Estimation for
       Autonomous Distributed Space Systems", 2020.
"""

from __future__ import annotations

from jax import Array
from jax.typing import ArrayLike

from astrojax.coordinates import state_eci_to_koe, state_koe_to_eci
from astrojax.relative_motion.oe_roe import state_oe_to_roe, state_roe_to_oe


def state_eci_to_roe(
    x_chief: ArrayLike,
    x_deputy: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Compute Relative Orbital Elements (ROE) from chief and deputy ECI states.

    Converts both ECI state vectors to Keplerian orbital elements, then
    computes the quasi-nonsingular Relative Orbital Elements between them.

    Args:
        x_chief: 6-element ECI state of the chief satellite
            ``[x, y, z, vx, vy, vz]``. Units: m, m/s.
        x_deputy: 6-element ECI state of the deputy satellite
            ``[x, y, z, vx, vy, vz]``. Units: m, m/s.
        use_degrees: If ``True``, return angular ROE components in degrees.

    Returns:
        Relative Orbital Elements ``[da, dλ, dex, dey, dix, diy]``.
            ``da`` is dimensionless, ``dex``/``dey`` are dimensionless,
            ``dλ``/``dix``/``diy`` are in *rad* (or *deg*).

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH
        from astrojax.coordinates import state_koe_to_eci
        from astrojax.relative_motion import state_eci_to_roe
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])
        x_c = state_koe_to_eci(oe_c, use_degrees=True)
        x_d = state_koe_to_eci(oe_d, use_degrees=True)
        roe = state_eci_to_roe(x_c, x_d, use_degrees=True)
        roe.shape
        ```

    References:
        Sullivan, J. "Nonlinear Angles-Only Orbit Estimation for
        Autonomous Distributed Space Systems", 2020.
    """
    oe_chief = state_eci_to_koe(x_chief, use_degrees=use_degrees)
    oe_deputy = state_eci_to_koe(x_deputy, use_degrees=use_degrees)
    return state_oe_to_roe(oe_chief, oe_deputy, use_degrees=use_degrees)


def state_roe_to_eci(
    x_chief: ArrayLike,
    roe: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Compute deputy ECI state from chief ECI state and Relative Orbital Elements.

    Converts the chief ECI state to Keplerian orbital elements, applies
    the ROE to obtain deputy orbital elements, then converts back to an
    ECI state vector.

    Args:
        x_chief: 6-element ECI state of the chief satellite
            ``[x, y, z, vx, vy, vz]``. Units: m, m/s.
        roe: Relative Orbital Elements ``[da, dλ, dex, dey, dix, diy]``.
            ``da`` is dimensionless, ``dex``/``dey`` are dimensionless,
            ``dλ``/``dix``/``diy`` are in *rad* (or *deg* if
            ``use_degrees=True``).
        use_degrees: If ``True``, interpret angular ROE components as degrees.

    Returns:
        6-element ECI state of the deputy satellite
            ``[x, y, z, vx, vy, vz]`` in *m* and *m/s*.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH
        from astrojax.coordinates import state_koe_to_eci
        from astrojax.relative_motion import state_roe_to_eci
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        x_c = state_koe_to_eci(oe_c, use_degrees=True)
        roe = jnp.array([0.000142857, 0.05, 0.0005, -0.0003, 0.01, -0.02])
        x_d = state_roe_to_eci(x_c, roe, use_degrees=True)
        x_d.shape
        ```

    References:
        Sullivan, J. "Nonlinear Angles-Only Orbit Estimation for
        Autonomous Distributed Space Systems", 2020.
    """
    oe_chief = state_eci_to_koe(x_chief, use_degrees=use_degrees)
    oe_deputy = state_roe_to_oe(oe_chief, roe, use_degrees=use_degrees)
    return state_koe_to_eci(oe_deputy, use_degrees=use_degrees)
