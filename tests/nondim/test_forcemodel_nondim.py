"""Nondim wrappers for force models: algebraic equivalence with SI."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from astrojax.config import get_dtype, set_dtype
from astrojax.constants import GM_EARTH
from astrojax.eop import zero_eop
from astrojax.epoch import Epoch
from astrojax.nondim import (
    UnitSystem,
    accel_third_body_moon_nondim,
    accel_third_body_sun_nondim,
    from_nondim_accel,
    from_nondim_velocity,
    nondim_orbit_dynamics,
    to_nondim_gravity_model,
    to_nondim_position,
    to_nondim_state,
)
from astrojax.orbit_dynamics import ForceModelConfig, create_orbit_dynamics
from astrojax.orbit_dynamics.gravity import (
    GravityModel,
    accel_gravity_spherical_harmonics,
)
from astrojax.orbit_dynamics.third_body import (
    accel_third_body_moon,
    accel_third_body_sun,
)


@pytest.fixture(autouse=True)
def restore_dtype():
    original = get_dtype()
    yield
    set_dtype(original)


class TestGravityModelAdapter:
    def test_gm_scales(self):
        set_dtype(jnp.float64)
        model = GravityModel.from_type("JGM3")
        units = UnitSystem.from_orbit(7000e3, GM_EARTH)
        model_nd = to_nondim_gravity_model(model, units)
        assert model_nd.gm == pytest.approx(model.gm / units.mu)

    def test_radius_scales(self):
        set_dtype(jnp.float64)
        model = GravityModel.from_type("JGM3")
        units = UnitSystem.from_orbit(7000e3, GM_EARTH)
        model_nd = to_nondim_gravity_model(model, units)
        assert model_nd.radius == pytest.approx(model.radius / units.LU)

    def test_coefficients_unchanged(self):
        set_dtype(jnp.float64)
        model = GravityModel.from_type("JGM3")
        units = UnitSystem.from_orbit(7000e3, GM_EARTH)
        model_nd = to_nondim_gravity_model(model, units)
        # C̄ / S̄ coefficients are dimensionless -- must not change.
        assert jnp.allclose(model_nd.coeff_c, model.coeff_c)
        assert jnp.allclose(model_nd.coeff_s, model.coeff_s)

    def test_acceleration_equivalent_after_denorm(self):
        """SI accel ≈ from_nondim_accel(nondim_accel)."""
        set_dtype(jnp.float64)
        model = GravityModel.from_type("JGM3")
        units = UnitSystem.from_orbit(7000e3, GM_EARTH)
        model_nd = to_nondim_gravity_model(model, units)

        r_si = jnp.array([7100e3, 0.0, 0.0])
        R = jnp.eye(3)  # identity rotation for the test
        a_si = accel_gravity_spherical_harmonics(r_si, R, model, 5, 5)

        r_nd = to_nondim_position(r_si, units)
        a_nd = accel_gravity_spherical_harmonics(r_nd, R, model_nd, 5, 5)
        a_si_from_nd = from_nondim_accel(a_nd, units)

        assert jnp.allclose(a_si, a_si_from_nd, rtol=1e-10), f"SI={a_si} vs from_nd={a_si_from_nd}"

        # Also verify with an off-axis position to exercise non-zonal terms.
        r_si_off = jnp.array([5000e3, 4000e3, 3000e3])
        a_si_off = accel_gravity_spherical_harmonics(r_si_off, R, model, 5, 5)
        r_nd_off = to_nondim_position(r_si_off, units)
        a_nd_off = accel_gravity_spherical_harmonics(r_nd_off, R, model_nd, 5, 5)
        a_si_from_nd_off = from_nondim_accel(a_nd_off, units)
        assert jnp.allclose(a_si_off, a_si_from_nd_off, rtol=1e-10), (
            f"off-axis: SI={a_si_off} vs from_nd={a_si_from_nd_off}"
        )


class TestThirdBodyNondimWrappers:
    @pytest.fixture
    def epoch(self):
        return Epoch(2024, 1, 1, 12, 0, 0.0)

    def test_sun_equivalent(self, epoch):
        set_dtype(jnp.float64)
        units = UnitSystem.from_orbit(7000e3, GM_EARTH)
        r_si = jnp.array([7100e3, 0.0, 0.0])
        r_nd = to_nondim_position(r_si, units)

        a_si = accel_third_body_sun(epoch, r_si)
        a_si_from_nd = from_nondim_accel(
            accel_third_body_sun_nondim(epoch, r_nd, units),
            units,
        )
        assert jnp.allclose(a_si, a_si_from_nd, rtol=1e-10)

    def test_moon_equivalent(self, epoch):
        set_dtype(jnp.float64)
        units = UnitSystem.from_orbit(7000e3, GM_EARTH)
        r_si = jnp.array([7100e3, 0.0, 0.0])
        r_nd = to_nondim_position(r_si, units)

        a_si = accel_third_body_moon(epoch, r_si)
        a_si_from_nd = from_nondim_accel(
            accel_third_body_moon_nondim(epoch, r_nd, units),
            units,
        )
        assert jnp.allclose(a_si, a_si_from_nd, rtol=1e-10)


class TestNondimOrbitDynamics:
    def test_two_body_matches_si_factory(self):
        """For pure two-body, nondim_orbit_dynamics returns a derivative
        whose denormalized output equals create_orbit_dynamics' output.
        """
        set_dtype(jnp.float64)
        eop = zero_eop()
        epoch_0 = Epoch(2024, 1, 1, 12, 0, 0.0)
        config = ForceModelConfig.two_body()

        units = UnitSystem.from_orbit(7000e3, GM_EARTH)
        dynamics_si = create_orbit_dynamics(eop, epoch_0, config)
        dynamics_nd = nondim_orbit_dynamics(eop, epoch_0, units, config)

        state_si = jnp.array([7100e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        t_si = 0.0
        state_nd = to_nondim_state(state_si, units)
        t_nd = 0.0

        d_si = dynamics_si(t_si, state_si)
        d_nd = dynamics_nd(t_nd, state_nd)

        # d_si is [v, a] SI. d_nd is [v_nd, a_nd] nondim.
        d_nd_denorm = jnp.concatenate(
            [
                from_nondim_velocity(d_nd[:3], units),
                from_nondim_accel(d_nd[3:], units),
            ]
        )
        assert jnp.allclose(d_si, d_nd_denorm, rtol=1e-10)
