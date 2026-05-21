"""Converter functions: round-trip correctness across all dtypes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from astrojax.config import get_dtype, set_dtype
from astrojax.constants import GM_EARTH
from astrojax.nondim import (
    UnitSystem,
    from_nondim_accel,
    from_nondim_density,
    from_nondim_force,
    from_nondim_position,
    from_nondim_state,
    from_nondim_time,
    from_nondim_velocity,
    to_nondim_accel,
    to_nondim_density,
    to_nondim_force,
    to_nondim_mu,
    to_nondim_position,
    to_nondim_state,
    to_nondim_time,
    to_nondim_velocity,
)


@pytest.fixture
def units() -> UnitSystem:
    return UnitSystem.from_orbit(7000e3, GM_EARTH)


@pytest.fixture(autouse=True)
def restore_dtype():
    original = get_dtype()
    yield
    set_dtype(original)


class TestRoundTripFloat32:
    """Round-trip in float32 (the astrojax default)."""

    def test_position(self, units):
        set_dtype(jnp.float32)
        r_si = jnp.array([7100e3, 0.0, 0.0], dtype=jnp.float32)
        r_nd = to_nondim_position(r_si, units)
        r_back = from_nondim_position(r_nd, units)
        assert jnp.allclose(r_si, r_back, rtol=1e-6)

    def test_state_round_trip(self, units):
        set_dtype(jnp.float32)
        x_si = jnp.array([7100e3, 0.0, 0.0, 0.0, 7.5e3, 0.0], dtype=jnp.float32)
        x_nd = to_nondim_state(x_si, units)
        x_back = from_nondim_state(x_nd, units)
        assert jnp.allclose(x_si, x_back, rtol=1e-5)

    def test_state_nondim_magnitude_is_order_one(self, units):
        """For from_orbit(a, μ), |r_nd| and |v_nd| are both ~ 1."""
        set_dtype(jnp.float32)
        x_si = jnp.array([7000e3, 0.0, 0.0, 0.0, 7.546e3, 0.0], dtype=jnp.float32)
        x_nd = to_nondim_state(x_si, units)
        assert 0.5 < float(jnp.linalg.norm(x_nd[:3])) < 2.0
        assert 0.5 < float(jnp.linalg.norm(x_nd[3:])) < 2.0

    def test_mu(self, units):
        set_dtype(jnp.float32)
        mu_nd = to_nondim_mu(jnp.float32(GM_EARTH), units)
        assert float(mu_nd) == pytest.approx(1.0, rel=1e-5)

    def test_accel(self, units):
        set_dtype(jnp.float32)
        a_si = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        a_back = from_nondim_accel(to_nondim_accel(a_si, units), units)
        assert jnp.allclose(a_si, a_back, rtol=1e-6)

    def test_density(self, units):
        set_dtype(jnp.float32)
        rho_si = jnp.float32(1.0e-12)
        rho_back = from_nondim_density(to_nondim_density(rho_si, units), units)
        assert float(rho_back) == pytest.approx(1.0e-12, rel=1e-5)

    def test_time(self, units):
        set_dtype(jnp.float32)
        t_si = jnp.float32(3600.0)
        t_back = from_nondim_time(to_nondim_time(t_si, units), units)
        assert float(t_back) == pytest.approx(3600.0, rel=1e-5)

    def test_force(self, units):
        set_dtype(jnp.float32)
        f_si = jnp.array([1e-6, 2e-6, 3e-6], dtype=jnp.float32)
        f_back = from_nondim_force(to_nondim_force(f_si, units), units)
        assert jnp.allclose(f_si, f_back, rtol=1e-5)


class TestRoundTripBfloat16:
    def test_position_round_trip(self, units):
        set_dtype(jnp.bfloat16)
        r_si = jnp.array([7100e3, 0.0, 0.0], dtype=jnp.bfloat16)
        r_nd = to_nondim_position(r_si, units)
        r_back = from_nondim_position(r_nd, units)
        # bfloat16 has only ~3 decimal digits; tolerate a few percent.
        assert jnp.allclose(r_si.astype(jnp.float32), r_back.astype(jnp.float32), rtol=5e-2)

    def test_velocity_round_trip(self, units):
        set_dtype(jnp.bfloat16)
        v_si = jnp.array([0.0, 7.5e3, 0.0], dtype=jnp.bfloat16)
        v_nd = to_nondim_velocity(v_si, units)
        v_back = from_nondim_velocity(v_nd, units)
        assert jnp.allclose(v_si.astype(jnp.float32), v_back.astype(jnp.float32), rtol=5e-2)


class TestJaxCompatibility:
    def test_jit_position(self, units):
        set_dtype(jnp.float32)
        r_si = jnp.array([7100e3, 0.0, 0.0], dtype=jnp.float32)
        jitted = jax.jit(lambda r: to_nondim_position(r, units))
        out = jitted(r_si)
        assert out.shape == (3,)

    def test_vmap_state(self, units):
        set_dtype(jnp.float32)
        states = jnp.tile(
            jnp.array([7100e3, 0.0, 0.0, 0.0, 7.5e3, 0.0], dtype=jnp.float32),
            (4, 1),
        )
        out = jax.vmap(lambda x: to_nondim_state(x, units))(states)
        assert out.shape == (4, 6)

    def test_grad_through_converter(self, units):
        set_dtype(jnp.float32)

        def loss(r_si):
            return jnp.sum(to_nondim_position(r_si, units) ** 2)

        r = jnp.array([7100e3, 0.0, 0.0], dtype=jnp.float32)
        g = jax.grad(loss)(r)
        # ∂loss/∂r_si = 2 * r_si / LU^2
        expected = 2.0 * r / units.LU**2
        assert jnp.allclose(g, expected, rtol=1e-4)


class TestOrderingForVelocityScale:
    """Velocity scale is LU/TU — confirms VU is used (not LU)."""

    def test_velocity_uses_VU(self, units):
        set_dtype(jnp.float32)
        v_si = jnp.array([0.0, units.VU, 0.0], dtype=jnp.float32)
        v_nd = to_nondim_velocity(v_si, units)
        assert jnp.allclose(v_nd, jnp.array([0.0, 1.0, 0.0]), atol=1e-5)
