"""Configurable orbit dynamics factory.

Composes individual force model building blocks into a single
``dynamics(t, state) -> derivative`` closure compatible with all
astrojax integrators.

The factory captures static configuration at Python trace time — boolean
toggles like ``config.drag`` become Python ``if`` branches that are
resolved during ``jax.jit`` tracing, producing an optimized computation
graph with no runtime branching overhead.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.constants import P_SUN
from astrojax.epoch import Epoch
from astrojax.frames import rotation_eci_to_ecef
from astrojax.orbit_dynamics.config import ForceModelConfig
from astrojax.orbit_dynamics.density import density_harris_priester
from astrojax.orbit_dynamics.drag import accel_drag
from astrojax.orbit_dynamics.ephemerides import sun_position
from astrojax.orbit_dynamics.gravity import (
    accel_gravity,
    accel_gravity_spherical_harmonics,
)
from astrojax.orbit_dynamics.srp import (
    accel_srp,
    eclipse_conical,
    eclipse_cylindrical,
)
from astrojax.orbit_dynamics.third_body import (
    accel_third_body_moon,
    accel_third_body_sun,
)


def create_orbit_dynamics(
    epoch_0: Epoch,
    config: ForceModelConfig | None = None,
) -> Callable[[ArrayLike, ArrayLike], Array]:
    """Create a configurable orbit dynamics function.

    Returns a closure ``dynamics(t, state) -> derivative`` that computes
    the time-derivative of a 6-element ECI state vector, composing the
    selected force models from *config*.

    The returned function is compatible with all astrojax integrators
    (``rk4_step``, ``rkf45_step``, ``dp54_step``, ``rkn1210_step``).

    Args:
        epoch_0: Reference epoch.  The integrator time *t* is interpreted
            as seconds since this epoch.
        config: Force model configuration.  Defaults to point-mass
            two-body gravity (``ForceModelConfig.two_body()``).

    Returns:
        A callable ``dynamics(t, state) -> derivative`` where:

        - *t*: seconds since *epoch_0* (scalar).
        - *state*: ``[x, y, z, vx, vy, vz]`` in ECI [m, m/s].
        - *derivative*: ``[vx, vy, vz, ax, ay, az]`` [m/s, m/s^2].

    Raises:
        ValueError: If *gravity_type* is ``"spherical_harmonics"`` but no
            *gravity_model* is provided.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax import Epoch
        from astrojax.orbit_dynamics.factory import create_orbit_dynamics
        from astrojax.orbit_dynamics.config import ForceModelConfig
        from astrojax.integrators import rk4_step
        epoch_0 = Epoch(2024, 6, 15, 12, 0, 0)
        dynamics = create_orbit_dynamics(epoch_0)
        x0 = jnp.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
        result = rk4_step(dynamics, 0.0, x0, 60.0)
        ```
    """
    if config is None:
        config = ForceModelConfig.two_body()

    # Validate spherical harmonics configuration
    use_sh = config.gravity_type == "spherical_harmonics"
    if use_sh and config.gravity_model is None:
        raise ValueError(
            "gravity_model must be provided when gravity_type is "
            "'spherical_harmonics'"
        )

    # Capture static configuration into local variables for the closure.
    # Python `if` on these booleans is resolved at trace time.
    _use_sh = use_sh
    _gravity_model = config.gravity_model
    _n_max = config.gravity_degree
    _m_max = config.gravity_order

    _drag = config.drag
    _srp = config.srp
    _third_body_sun = config.third_body_sun
    _third_body_moon = config.third_body_moon

    _eclipse_model = config.eclipse_model

    _mass = config.spacecraft.mass
    _drag_area = config.spacecraft.drag_area
    _srp_area = config.spacecraft.srp_area
    _cd = config.spacecraft.cd
    _cr = config.spacecraft.cr

    # Precompute whether we need the ECI-to-ECEF rotation or sun position,
    # so we can share intermediate computations.
    _needs_R = _use_sh or _drag
    _needs_r_sun = _drag or _srp or _third_body_sun

    def dynamics(t: ArrayLike, state: ArrayLike) -> Array:
        """Orbit dynamics: state derivative in ECI.

        Args:
            t: Seconds since epoch_0 (scalar).
            state: ``[x, y, z, vx, vy, vz]`` in ECI [m, m/s].

        Returns:
            jax.Array: ``[vx, vy, vz, ax, ay, az]`` [m/s, m/s^2].
        """
        r = state[:3]
        v = state[3:6]

        # Current epoch
        epc = epoch_0 + t

        # --- Shared intermediates ---
        R_eci_ecef = rotation_eci_to_ecef(epc) if _needs_R else None
        r_sun = sun_position(epc) if _needs_r_sun else None

        # --- Gravity ---
        if _use_sh:
            a = accel_gravity_spherical_harmonics(
                r, R_eci_ecef, _gravity_model, _n_max, _m_max
            )
        else:
            a = accel_gravity(r)

        # --- Third-body perturbations ---
        if _third_body_sun:
            a = a + accel_third_body_sun(epc, r)

        if _third_body_moon:
            a = a + accel_third_body_moon(epc, r)

        # --- Atmospheric drag ---
        if _drag:
            r_ecef = R_eci_ecef @ r
            rho = density_harris_priester(r_ecef, r_sun)
            a = a + accel_drag(state, rho, _mass, _drag_area, _cd, R_eci_ecef)

        # --- Solar radiation pressure ---
        if _srp:
            if _eclipse_model == "conical":
                nu = eclipse_conical(r, r_sun)
            elif _eclipse_model == "cylindrical":
                nu = eclipse_cylindrical(r, r_sun)
            else:
                nu = 1.0
            a = a + nu * accel_srp(r, r_sun, _mass, _cr, _srp_area, P_SUN)

        return jnp.concatenate([v, a])

    return dynamics
