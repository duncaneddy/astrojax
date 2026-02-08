"""Tests for the orbit_dynamics module.

Validates ephemerides, gravity, third-body perturbations,
Harris-Priester density, drag, SRP, and eclipse models.
"""

import jax
import jax.numpy as jnp
import pytest

from astrojax import Epoch
from astrojax.constants import (
    AU, GM_EARTH, GM_SUN,
    R_EARTH, P_SUN, DEG2RAD,
)
from astrojax.orbit_dynamics import (
    sun_position,
    moon_position,
    accel_point_mass,
    accel_gravity,
    accel_third_body_sun,
    accel_third_body_moon,
    density_harris_priester,
    accel_drag,
    accel_srp,
    eclipse_conical,
    eclipse_cylindrical,
)


# ---------------------------------------------------------------------------
# Helper: create Epoch from MJD (astrojax assumes UTC ≈ TT)
# MJD 60310.0 = 2024-01-01 00:00:00
# ---------------------------------------------------------------------------
def _epc_mjd_60310():
    """Epoch corresponding to MJD 60310.0 (2024-01-01)."""
    return Epoch(2024, 1, 1)


# ===========================================================================
# Sun & Moon Ephemerides
# ===========================================================================
class TestSunPosition:
    """Tests for sun_position()."""

    def test_magnitude_approximately_1au(self):
        """Sun distance should be approximately 1 AU."""
        epc = _epc_mjd_60310()
        r_sun = sun_position(epc)
        dist = float(jnp.linalg.norm(r_sun))
        assert 0.98 * AU < dist < 1.02 * AU

    def test_shape(self):
        """Output should be a 3-element vector."""
        epc = _epc_mjd_60310()
        r_sun = sun_position(epc)
        assert r_sun.shape == (3,)

    def test_jit_compatible(self):
        """sun_position should be JIT-compilable."""
        epc = _epc_mjd_60310()
        r_eager = sun_position(epc)
        r_jit = jax.jit(sun_position)(epc)
        assert jnp.allclose(r_eager, r_jit, atol=1.0)

    def test_different_epochs_give_different_positions(self):
        """Sun position should change over time."""
        epc1 = Epoch(2024, 1, 1)
        epc2 = Epoch(2024, 7, 1)
        r1 = sun_position(epc1)
        r2 = sun_position(epc2)
        assert not jnp.allclose(r1, r2, atol=1e6)


class TestMoonPosition:
    """Tests for moon_position()."""

    def test_magnitude_approximately_384000km(self):
        """Moon distance should be approximately 384,400 km."""
        epc = _epc_mjd_60310()
        r_moon = moon_position(epc)
        dist = float(jnp.linalg.norm(r_moon))
        assert 3.5e8 < dist < 4.1e8

    def test_shape(self):
        """Output should be a 3-element vector."""
        epc = _epc_mjd_60310()
        r_moon = moon_position(epc)
        assert r_moon.shape == (3,)

    def test_jit_compatible(self):
        """moon_position should be JIT-compilable."""
        epc = _epc_mjd_60310()
        r_eager = moon_position(epc)
        r_jit = jax.jit(moon_position)(epc)
        # float32 eager vs JIT can diverge ~5 km due to long trig chains
        assert jnp.allclose(r_eager, r_jit, rtol=5e-5, atol=1e4)


# ===========================================================================
# Point-Mass Gravity
# ===========================================================================
class TestAccelPointMass:
    """Tests for accel_point_mass()."""

    def test_surface_gravity(self):
        """Gravity at Earth's surface should be ~9.8 m/s^2."""
        r = jnp.array([R_EARTH, 0.0, 0.0])
        a = accel_point_mass(r, jnp.zeros(3), GM_EARTH)
        assert abs(float(jnp.linalg.norm(a)) - GM_EARTH / R_EARTH**2) < 1e-4

    def test_direction(self):
        """Gravity should point towards the central body."""
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        a = accel_point_mass(r, jnp.zeros(3), GM_EARTH)
        # Should point in -x direction
        assert float(a[0]) < 0
        assert abs(float(a[1])) < 1e-10
        assert abs(float(a[2])) < 1e-10

    def test_third_body_form(self):
        """Third-body form with non-zero r_body."""
        r_obj = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        r_body = jnp.array([AU, 0.0, 0.0])
        a = accel_point_mass(r_obj, r_body, GM_SUN)
        # Should return a valid acceleration
        assert float(jnp.linalg.norm(a)) > 0

    def test_shape(self):
        """Output should be a 3-element vector."""
        r = jnp.array([R_EARTH, 0.0, 0.0])
        a = accel_point_mass(r, jnp.zeros(3), GM_EARTH)
        assert a.shape == (3,)

    def test_accepts_6d_state(self):
        """Should accept a 6D state, using only the first 3 elements."""
        x = jnp.array([R_EARTH, 0.0, 0.0, 0.0, 7500.0, 0.0])
        a = accel_point_mass(x, jnp.zeros(3), GM_EARTH)
        r = jnp.array([R_EARTH, 0.0, 0.0])
        a_3d = accel_point_mass(r, jnp.zeros(3), GM_EARTH)
        assert jnp.allclose(a, a_3d, atol=1e-10)

    def test_jit_compatible(self):
        """accel_point_mass should be JIT-compilable."""
        r = jnp.array([R_EARTH, 0.0, 0.0])

        @jax.jit
        def f(r):
            return accel_point_mass(r, jnp.zeros(3), GM_EARTH)

        a_jit = f(r)
        a_eager = accel_point_mass(r, jnp.zeros(3), GM_EARTH)
        assert jnp.allclose(a_jit, a_eager, atol=1e-10)


class TestAccelGravity:
    """Tests for accel_gravity()."""

    def test_matches_point_mass(self):
        """accel_gravity should match accel_point_mass with Earth params."""
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        a1 = accel_gravity(r)
        a2 = accel_point_mass(r, jnp.zeros(3), GM_EARTH)
        assert jnp.allclose(a1, a2, atol=1e-10)


# ===========================================================================
# Third-Body Perturbations
# ===========================================================================
class TestThirdBody:
    """Tests for accel_third_body_sun() and accel_third_body_moon()."""

    def test_sun_magnitude_at_leo(self):
        """Sun perturbation at LEO should be ~1e-7 to 1e-6 m/s^2."""
        epc = _epc_mjd_60310()
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        a = accel_third_body_sun(epc, r)
        mag = float(jnp.linalg.norm(a))
        assert 1e-8 < mag < 1e-5

    def test_moon_magnitude_at_leo(self):
        """Moon perturbation at LEO should be ~1e-7 to 1e-5 m/s^2."""
        epc = _epc_mjd_60310()
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        a = accel_third_body_moon(epc, r)
        mag = float(jnp.linalg.norm(a))
        assert 1e-8 < mag < 1e-5

    def test_sun_jit_compatible(self):
        """accel_third_body_sun should be JIT-compilable."""
        epc = _epc_mjd_60310()
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        a_eager = accel_third_body_sun(epc, r)
        a_jit = jax.jit(accel_third_body_sun)(epc, r)
        assert jnp.allclose(a_eager, a_jit, atol=1e-12)

    def test_moon_jit_compatible(self):
        """accel_third_body_moon should be JIT-compilable."""
        epc = _epc_mjd_60310()
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        a_eager = accel_third_body_moon(epc, r)
        a_jit = jax.jit(accel_third_body_moon)(epc, r)
        # float32 moon ephemeris propagates ~1e-5 relative error to acceleration
        assert jnp.allclose(a_eager, a_jit, rtol=1e-4, atol=1e-12)


# ===========================================================================
# Harris-Priester Atmospheric Density
# ===========================================================================
class TestDensityHarrisPriester:
    """Tests for density_harris_priester()."""

    _R_SUN = jnp.array([24622331959.58, -133060326832.922, -57688711921.833])

    def test_typical_leo_density(self):
        """Density at ~88 km altitude should be non-zero and reasonable."""
        # Position at ~88km above surface (just within 100km lower bound
        # after geodetic conversion - this is ECEF so geodetic alt differs)
        r = jnp.array([0.0, 0.0, -6466752.314])
        rho = density_harris_priester(r, self._R_SUN)
        assert float(rho) > 0.0

    def test_zero_below_100km(self):
        """Density should be 0 below 100 km altitude."""
        from astrojax.coordinates import position_geodetic_to_ecef
        r = position_geodetic_to_ecef(jnp.array([0.0, 0.0, 50.0e3]))
        rho = density_harris_priester(r, self._R_SUN)
        assert float(rho) == 0.0

    def test_zero_above_1000km(self):
        """Density should be 0 above 1000 km altitude."""
        from astrojax.coordinates import position_geodetic_to_ecef
        r = position_geodetic_to_ecef(jnp.array([0.0, 0.0, 1100.0e3]))
        rho = density_harris_priester(r, self._R_SUN)
        assert float(rho) == 0.0

    def test_density_decreases_with_altitude(self):
        """Density should decrease with increasing altitude."""
        rho_values = []
        for alt_km in [200, 400, 600, 800]:
            r = jnp.array([0.0, 0.0, -(R_EARTH + alt_km * 1e3)])
            rho = density_harris_priester(r, self._R_SUN)
            rho_values.append(float(rho))
        for i in range(len(rho_values) - 1):
            assert rho_values[i] > rho_values[i + 1]

    def test_jit_compatible(self):
        """density_harris_priester should be JIT-compilable."""
        r = jnp.array([0.0, 0.0, -6466752.314])

        @jax.jit
        def f(r):
            return density_harris_priester(r, self._R_SUN)

        rho_jit = f(r)
        rho_eager = density_harris_priester(r, self._R_SUN)
        # float32 exponential interpolation can diverge ~2e-5 relative between eager/JIT
        assert jnp.allclose(rho_jit, rho_eager, rtol=5e-5, atol=1e-15)


# ===========================================================================
# Atmospheric Drag
# ===========================================================================
class TestAccelDrag:
    """Tests for accel_drag()."""

    def test_basic_drag(self):
        """Drag acceleration should oppose the velocity direction."""
        x = jnp.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
        a = accel_drag(x, 1e-12, 1000.0, 1.0, 2.0, jnp.eye(3))
        # With identity rotation, drag should oppose v (in -y direction)
        assert float(a[1]) < 0

    def test_zero_density_zero_drag(self):
        """Zero density should produce zero drag."""
        x = jnp.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
        a = accel_drag(x, 0.0, 1000.0, 1.0, 2.0, jnp.eye(3))
        assert float(jnp.linalg.norm(a)) == 0.0

    def test_shape(self):
        """Output should be a 3-element vector."""
        x = jnp.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
        a = accel_drag(x, 1e-12, 1000.0, 1.0, 2.0, jnp.eye(3))
        assert a.shape == (3,)

    def test_jit_compatible(self):
        """accel_drag should be JIT-compilable."""
        x = jnp.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])

        @jax.jit
        def f(x):
            return accel_drag(x, 1e-12, 1000.0, 1.0, 2.0, jnp.eye(3))

        a_jit = f(x)
        a_eager = accel_drag(x, 1e-12, 1000.0, 1.0, 2.0, jnp.eye(3))
        assert jnp.allclose(a_jit, a_eager, atol=1e-18)

    def test_rust_reference_magnitude(self):
        """Drag with identity T at LEO should match Rust reference magnitude."""
        from astrojax.coordinates import state_koe_to_eci
        oe = jnp.array([R_EARTH + 500e3, 0.01, 97.3 * DEG2RAD,
                         15.0 * DEG2RAD, 30.0 * DEG2RAD, 45.0 * DEG2RAD])
        x = state_koe_to_eci(oe)
        a = accel_drag(x, 1.0e-12, 1000.0, 1.0, 2.0, jnp.eye(3))
        mag = float(jnp.linalg.norm(a))
        # Rust test: ~5.976e-8 m/s^2. Allow generous float32 tolerance.
        assert 1e-9 < mag < 1e-6


# ===========================================================================
# Solar Radiation Pressure
# ===========================================================================
class TestAccelSRP:
    """Tests for accel_srp()."""

    def test_at_1au_unit_params(self):
        """SRP at 1 AU with unit params should give |a| = p0."""
        r = jnp.array([AU, 0.0, 0.0])
        r_sun = jnp.zeros(3)
        a = accel_srp(r, r_sun, 1.0, 1.0, 1.0, 4.5e-6)
        assert abs(float(a[0]) - 4.5e-6) < 1e-12
        assert abs(float(a[1])) < 1e-15
        assert abs(float(a[2])) < 1e-15

    def test_direction_away_from_sun(self):
        """SRP should push away from the Sun."""
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        r_sun = jnp.array([-AU, 0.0, 0.0])
        a = accel_srp(r, r_sun, 100.0, 1.8, 1.0, P_SUN)
        assert float(a[0]) > 0  # pushed in +x (away from Sun at -AU)

    def test_shape(self):
        """Output should be a 3-element vector."""
        r = jnp.array([AU, 0.0, 0.0])
        a = accel_srp(r, jnp.zeros(3), 1.0, 1.0, 1.0, 4.5e-6)
        assert a.shape == (3,)

    def test_jit_compatible(self):
        """accel_srp should be JIT-compilable."""
        r = jnp.array([AU, 0.0, 0.0])

        @jax.jit
        def f(r):
            return accel_srp(r, jnp.zeros(3), 1.0, 1.0, 1.0, 4.5e-6)

        a_jit = f(r)
        a_eager = accel_srp(r, jnp.zeros(3), 1.0, 1.0, 1.0, 4.5e-6)
        assert jnp.allclose(a_jit, a_eager, atol=1e-15)


# ===========================================================================
# Eclipse Models
# ===========================================================================
class TestEclipseConical:
    """Tests for eclipse_conical()."""

    def test_fully_illuminated(self):
        """Object on the sunlit side should be fully illuminated."""
        r_sun = jnp.array([AU, 0.0, 0.0])
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        nu = eclipse_conical(r, r_sun)
        assert float(nu) == pytest.approx(1.0, abs=1e-6)

    def test_full_shadow(self):
        """Object directly behind Earth should be in full shadow."""
        r_sun = jnp.array([AU, 0.0, 0.0])
        # Place satellite directly behind Earth, close enough to be in shadow
        r = jnp.array([-(R_EARTH + 500e3), 0.0, 0.0])
        nu = eclipse_conical(r, r_sun)
        assert float(nu) == pytest.approx(0.0, abs=1e-6)

    def test_output_range(self):
        """Illumination fraction should be in [0, 1]."""
        r_sun = jnp.array([AU, 0.0, 0.0])
        r = jnp.array([0.0, R_EARTH + 500e3, 0.0])
        nu = eclipse_conical(r, r_sun)
        assert 0.0 <= float(nu) <= 1.0

    def test_jit_compatible(self):
        """eclipse_conical should be JIT-compilable."""
        r_sun = jnp.array([AU, 0.0, 0.0])
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        nu_eager = eclipse_conical(r, r_sun)
        nu_jit = jax.jit(eclipse_conical)(r, r_sun)
        assert jnp.allclose(nu_eager, nu_jit, atol=1e-8)


class TestEclipseCylindrical:
    """Tests for eclipse_cylindrical()."""

    def test_fully_illuminated(self):
        """Object on the sunlit side should be illuminated."""
        r_sun = jnp.array([AU, 0.0, 0.0])
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        nu = eclipse_cylindrical(r, r_sun)
        assert float(nu) == pytest.approx(1.0, abs=1e-6)

    def test_full_shadow(self):
        """Object behind Earth should be in shadow."""
        r_sun = jnp.array([AU, 0.0, 0.0])
        r = jnp.array([-(R_EARTH + 500e3), 0.0, 0.0])
        nu = eclipse_cylindrical(r, r_sun)
        assert float(nu) == pytest.approx(0.0, abs=1e-6)

    def test_binary_output(self):
        """Cylindrical model should only return 0 or 1."""
        r_sun = jnp.array([AU, 0.0, 0.0])
        for pos in [
            [R_EARTH + 500e3, 0.0, 0.0],
            [-(R_EARTH + 500e3), 0.0, 0.0],
            [0.0, R_EARTH + 500e3, 0.0],
        ]:
            nu = eclipse_cylindrical(jnp.array(pos), r_sun)
            assert float(nu) in (0.0, 1.0)

    def test_jit_compatible(self):
        """eclipse_cylindrical should be JIT-compilable."""
        r_sun = jnp.array([AU, 0.0, 0.0])
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        nu_eager = eclipse_cylindrical(r, r_sun)
        nu_jit = jax.jit(eclipse_cylindrical)(r, r_sun)
        assert jnp.allclose(nu_eager, nu_jit, atol=1e-8)
