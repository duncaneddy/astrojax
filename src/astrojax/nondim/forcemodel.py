"""Nondim adapters for astrojax orbit-dynamics force models.

Most existing force functions (``accel_point_mass``, ``accel_srp``,
``accel_drag``, ``accel_gravity_spherical_harmonics``) are already
parameterized so that passing nondim inputs and a nondim
``GravityModel`` yields nondim outputs with no code changes.

Provides:
- ``to_nondim_gravity_model``: scales a GravityModel for nondim use.
- ``accel_third_body_sun_nondim`` / ``accel_third_body_moon_nondim``:
  nondim wrappers for third-body acceleration.
- ``nondim_orbit_dynamics``: configurable orbit-dynamics factory that
  pre-scales SI constants and dispatches to nondim-aware force terms.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.constants import GM_EARTH, P_SUN
from astrojax.eop._types import EOPData
from astrojax.epoch import Epoch
from astrojax.frames import rotation_eci_to_ecef
from astrojax.frames.gcrf_itrf import bias_precession_nutation
from astrojax.nondim.converters import (
    from_nondim_position,
    from_nondim_velocity,
    to_nondim_accel,
)
from astrojax.nondim.system import UnitSystem
from astrojax.orbit_dynamics.config import ForceModelConfig
from astrojax.orbit_dynamics.density import density_harris_priester
from astrojax.orbit_dynamics.drag import accel_drag
from astrojax.orbit_dynamics.ephemerides import sun_position
from astrojax.orbit_dynamics.gravity import (
    GravityModel,
    accel_gravity_spherical_harmonics,
    accel_point_mass,
)
from astrojax.orbit_dynamics.nrlmsise00 import density_nrlmsise00
from astrojax.orbit_dynamics.srp import (
    accel_srp,
    eclipse_conical,
    eclipse_cylindrical,
)
from astrojax.orbit_dynamics.third_body import (
    accel_third_body_moon,
    accel_third_body_sun,
)
from astrojax.space_weather._types import SpaceWeatherData


def to_nondim_gravity_model(model: GravityModel, units: UnitSystem) -> GravityModel:
    """Return a GravityModel with ``gm`` and ``radius`` scaled to nondim.

    Dimensionless Stokes coefficients (the ``data`` matrix) are
    unchanged; the new instance re-runs ``_precompute_coefficients`` on
    its own.

    Args:
        model: An SI ``GravityModel`` (e.g. ``GravityModel.from_type("JGM3")``).
        units: The ``UnitSystem`` to scale into.

    Returns:
        A new ``GravityModel`` whose acceleration outputs, when called
        with nondim position inputs, will be in nondim units.
    """
    return GravityModel(
        model_name=f"{model.model_name}_nondim",
        gm=model.gm / units.mu,
        radius=model.radius / units.LU,
        n_max=model.n_max,
        m_max=model.m_max,
        data=model.data,
        tide_system=model.tide_system,
        normalization=model.normalization,
    )


def accel_third_body_sun_nondim(epc: Epoch, r_nd: ArrayLike, units: UnitSystem) -> Array:
    """Third-body solar acceleration with nondim I/O.

    Internally converts ``r_nd`` to SI, calls the SI routine (which
    queries planetary ephemerides in SI), and converts the resulting
    acceleration back to nondim.
    """
    r_si = from_nondim_position(r_nd, units)
    a_si = accel_third_body_sun(epc, r_si)
    return to_nondim_accel(a_si, units)


def accel_third_body_moon_nondim(epc: Epoch, r_nd: ArrayLike, units: UnitSystem) -> Array:
    """Third-body lunar acceleration with nondim I/O."""
    r_si = from_nondim_position(r_nd, units)
    a_si = accel_third_body_moon(epc, r_si)
    return to_nondim_accel(a_si, units)


def nondim_orbit_dynamics(
    eop: EOPData,
    epoch_0: Epoch,
    units: UnitSystem,
    config: ForceModelConfig | None = None,
    space_weather: SpaceWeatherData | None = None,
) -> Callable[[ArrayLike, ArrayLike], Array]:
    """Construct a nondimensional orbit-dynamics closure.

    Mirrors :func:`~astrojax.orbit_dynamics.factory.create_orbit_dynamics`
    but pre-scales all SI constants to *units* at closure-build time so
    the returned function operates entirely in nondim units.  Only force
    terms that genuinely require absolute SI quantities (ephemerides,
    atmospheric density, frame rotations) momentarily cross the unit
    boundary; their results are converted back to nondim before being
    summed into the acceleration.

    Args:
        eop: Earth orientation parameters for ECI-ECEF frame rotations.
            Use ``zero_eop()`` when no Earth orientation corrections are
            needed.
        epoch_0: Reference epoch.  The integrator time ``t_nd`` is
            interpreted as nondim time since this epoch (i.e. SI seconds
            since *epoch_0* divided by ``units.TU``).
        units: The :class:`UnitSystem` defining all nondim scales.
        config: Force model configuration.  Defaults to point-mass
            two-body gravity (``ForceModelConfig.two_body()``).
        space_weather: Space weather data for NRLMSISE-00 density model.
            Required when ``config.drag`` is True and
            ``config.density_model == "nrlmsise00"``.

    Returns:
        A callable ``dynamics_nd(t_nd, state_nd) -> deriv_nd`` where:

        - ``t_nd``: nondim time since *epoch_0* (= SI seconds / TU).
        - ``state_nd``: ``[r/LU, v/VU]`` (6-vector).
        - ``deriv_nd``: ``[v_nd, a_nd]`` (6-vector) in nondim units.

    Raises:
        ValueError: If *gravity_type* is ``"spherical_harmonics"`` but no
            *gravity_model* is provided.
        ValueError: If *density_model* is ``"nrlmsise00"`` but no
            *space_weather* is provided.
    """
    if config is None:
        config = ForceModelConfig.two_body()

    _eop = eop

    # Validate spherical harmonics configuration
    use_sh = config.gravity_type == "spherical_harmonics"
    if use_sh and config.gravity_model is None:
        raise ValueError(
            "gravity_model must be provided when gravity_type is 'spherical_harmonics'"
        )

    # Validate NRLMSISE-00 configuration
    _density_model = config.density_model
    if config.drag and _density_model == "nrlmsise00" and space_weather is None:
        raise ValueError("space_weather must be provided when density_model is 'nrlmsise00'")
    _sw = space_weather

    # ─── Pre-scale SI constants to nondim ─────────────────────────────
    # Point-mass gravity uses Earth's gravitational parameter, scaled
    # once at closure-build time.  For ``UnitSystem.from_orbit(a, GM_EARTH)``
    # this equals 1.0 by construction.
    _mu_nd = GM_EARTH / units.mu

    # Spherical-harmonic gravity model is pre-scaled (gm, radius) once;
    # dimensionless Stokes coefficients are unchanged.
    _gravity_model_nd: GravityModel | None = None
    if use_sh:
        _gravity_model_nd = to_nondim_gravity_model(config.gravity_model, units)
    _use_sh = use_sh
    _n_max = config.gravity_degree
    _m_max = config.gravity_order

    # Capture static perturbation toggles into the closure
    _drag = config.drag
    _srp = config.srp
    _third_body_sun = config.third_body_sun
    _third_body_moon = config.third_body_moon
    _eclipse_model = config.eclipse_model

    # Spacecraft physical parameters (SI, captured for SI-side calls)
    _mass = config.spacecraft.mass
    _drag_area = config.spacecraft.drag_area
    _srp_area = config.spacecraft.srp_area
    _cd = config.spacecraft.cd
    _cr = config.spacecraft.cr

    # Determine which shared intermediates are needed (matches factory.py)
    _needs_R = _use_sh or _drag
    _needs_BPN = _drag and _density_model == "harris_priester"
    _needs_r_sun = (_drag and _density_model == "harris_priester") or _srp or _third_body_sun
    _needs_r_si = _drag or _srp
    _needs_v_si = _drag

    def dynamics_nd(t_nd: ArrayLike, state_nd: ArrayLike) -> Array:
        """Nondim orbit dynamics: state derivative in nondim units.

        Args:
            t_nd: Nondim time since ``epoch_0`` (= SI seconds / units.TU).
            state_nd: ``[r/LU, v/VU]`` (6-vector, nondim).

        Returns:
            jax.Array: ``[v_nd, a_nd]`` (6-vector, nondim).
        """
        r_nd = state_nd[:3]
        v_nd = state_nd[3:6]

        # Reconstitute SI epoch only for terms that need it
        # (ephemerides, frame rotation, atmospheric density).
        t_si = t_nd * units.TU
        epc = epoch_0 + t_si

        # --- Shared intermediates ---
        R_eci_ecef = rotation_eci_to_ecef(_eop, epc) if _needs_R else None
        BPN = bias_precession_nutation(_eop, epc) if _needs_BPN else None
        r_sun_si = sun_position(epc) if _needs_r_sun else None
        r_si = from_nondim_position(r_nd, units) if _needs_r_si else None
        v_si = from_nondim_velocity(v_nd, units) if _needs_v_si else None

        # --- Gravity (always present) ---
        if _use_sh:
            # Spherical-harmonic gravity is dimensionally homogeneous: a
            # nondim model with nondim position yields nondim accel.
            a_nd = accel_gravity_spherical_harmonics(
                r_nd, R_eci_ecef, _gravity_model_nd, _n_max, _m_max
            )
        else:
            # Point-mass gravity in nondim units: pass the pre-scaled mu.
            a_nd = accel_point_mass(r_nd, jnp.zeros(3, dtype=r_nd.dtype), _mu_nd)

        # --- Third-body perturbations ---
        if _third_body_sun:
            a_nd = a_nd + accel_third_body_sun_nondim(epc, r_nd, units)

        if _third_body_moon:
            a_nd = a_nd + accel_third_body_moon_nondim(epc, r_nd, units)

        # --- Atmospheric drag ---
        # Drag involves OMEGA_EARTH (SI rad/s) inside ``accel_drag``, so
        # we evaluate it in SI and convert the result back to nondim.
        if _drag:
            state_si = jnp.concatenate([r_si, v_si])
            if _density_model == "harris_priester":
                r_tod = BPN @ r_si
                r_sun_tod = BPN @ r_sun_si
                rho = density_harris_priester(r_tod, r_sun_tod)
            elif _density_model == "nrlmsise00":
                r_ecef = R_eci_ecef @ r_si
                rho = density_nrlmsise00(_sw, epc, r_ecef)
            a_drag_si = accel_drag(state_si, rho, _mass, _drag_area, _cd, R_eci_ecef)
            a_nd = a_nd + to_nondim_accel(a_drag_si, units)

        # --- Solar radiation pressure ---
        # ``accel_srp`` references AU (SI), so evaluate in SI then scale.
        if _srp:
            if _eclipse_model == "conical":
                nu = eclipse_conical(r_si, r_sun_si)
            elif _eclipse_model == "cylindrical":
                nu = eclipse_cylindrical(r_si, r_sun_si)
            else:
                nu = 1.0
            a_srp_si = nu * accel_srp(r_si, r_sun_si, _mass, _cr, _srp_area, P_SUN)
            a_nd = a_nd + to_nondim_accel(a_srp_si, units)

        return jnp.concatenate([v_nd, a_nd])

    return dynamics_nd
