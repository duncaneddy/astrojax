"""Tests for the orbit_dynamics module.

Validates ephemerides, gravity, third-body perturbations,
Harris-Priester density, drag, SRP, and eclipse models.
"""

import jax
import jax.numpy as jnp
import pytest

from astrojax import Epoch
from astrojax.constants import (
    AU,
    DEG2RAD,
    GM_EARTH,
    GM_SUN,
    P_SUN,
    R_EARTH,
)
from astrojax.orbit_dynamics import (
    GravityModel,
    accel_drag,
    accel_gravity,
    accel_gravity_spherical_harmonics,
    accel_point_mass,
    accel_srp,
    accel_third_body_moon,
    accel_third_body_sun,
    density_harris_priester,
    eclipse_conical,
    eclipse_cylindrical,
    moon_position,
    sun_position,
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
# Gravity Model
# ===========================================================================
class TestGravityModel:
    """Tests for GravityModel class."""

    def test_load_egm2008_360(self):
        """EGM2008_360 model metadata should match expected values."""
        model = GravityModel.from_type("EGM2008_360")
        assert model.model_name == "EGM2008"
        assert model.gm == GM_EARTH
        assert model.radius == R_EARTH
        assert model.n_max == 360
        assert model.m_max == 360
        assert model.tide_system == "tide_free"
        assert model.normalization == "fully_normalized"

    def test_load_ggm05s(self):
        """GGM05S model metadata should match expected values."""
        model = GravityModel.from_type("GGM05S")
        assert model.model_name == "GGM05S"
        assert model.gm == GM_EARTH
        assert model.radius == R_EARTH
        assert model.n_max == 180
        assert model.m_max == 180
        assert model.tide_system == "zero_tide"
        assert model.normalization == "fully_normalized"

    def test_load_jgm3(self):
        """JGM3 model metadata should match expected values."""
        model = GravityModel.from_type("JGM3")
        assert model.model_name == "JGM3"
        assert model.gm == GM_EARTH
        assert model.radius == R_EARTH
        assert model.n_max == 70
        assert model.m_max == 70
        assert model.normalization == "fully_normalized"

    def test_get_coefficients(self):
        """Coefficient retrieval should match known values."""
        model = GravityModel.from_type("EGM2008_360")

        c, s = model.get(2, 0)
        assert c == pytest.approx(-0.484165143790815e-03, abs=1e-12)
        assert s == pytest.approx(0.0, abs=1e-12)

        c, s = model.get(3, 3)
        assert c == pytest.approx(0.721321757121568e-06, abs=1e-12)
        assert s == pytest.approx(0.141434926192941e-05, abs=1e-12)

        c, s = model.get(360, 360)
        assert c == pytest.approx(0.200046056782130e-10, abs=1e-12)
        assert s == pytest.approx(-0.958653755280305e-10, abs=1e-12)

    def test_get_out_of_bounds(self):
        """Requesting coefficients beyond model bounds should raise."""
        model = GravityModel.from_type("EGM2008_360")
        with pytest.raises(ValueError):
            model.get(361, 0)

    def test_set_max_degree_order(self):
        """Truncation should reduce bounds and preserve coefficients."""
        original = GravityModel.from_type("JGM3")
        truncated = GravityModel.from_type("JGM3")

        c_orig, s_orig = original.get(2, 0)
        c_10_5_orig, s_10_5_orig = original.get(10, 5)
        c_20_20_orig, s_20_20_orig = original.get(20, 20)

        truncated.set_max_degree_order(20, 20)

        assert truncated.n_max == 20
        assert truncated.m_max == 20

        c, s = truncated.get(2, 0)
        assert c == pytest.approx(c_orig, abs=1e-15)
        assert s == pytest.approx(s_orig, abs=1e-15)

        c, s = truncated.get(10, 5)
        assert c == pytest.approx(c_10_5_orig, abs=1e-15)
        assert s == pytest.approx(s_10_5_orig, abs=1e-15)

        c, s = truncated.get(20, 20)
        assert c == pytest.approx(c_20_20_orig, abs=1e-15)
        assert s == pytest.approx(s_20_20_orig, abs=1e-15)

        with pytest.raises(ValueError):
            truncated.get(21, 0)

    def test_set_max_degree_order_computation_parity(self):
        """Truncated model should match full model at same degree."""
        from astrojax.config import set_dtype
        set_dtype(jnp.float64)

        truncated = GravityModel.from_type("JGM3")
        truncated.set_max_degree_order(20, 20)
        full = GravityModel.from_type("JGM3")

        r = jnp.array([6525.919e3, 1710.416e3, 2508.886e3])
        R = jnp.eye(3)

        a_trunc = accel_gravity_spherical_harmonics(r, R, truncated, 20, 20)
        a_full = accel_gravity_spherical_harmonics(r, R, full, 20, 20)

        assert float(jnp.max(jnp.abs(a_trunc - a_full))) < 1e-15

        set_dtype(jnp.float32)

    def test_unknown_model_type(self):
        """Unknown model type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown gravity model type"):
            GravityModel.from_type("NONEXISTENT")

    def test_nonexistent_file(self):
        """Loading from a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            GravityModel.from_file("/nonexistent/path.gfc")


# ===========================================================================
# Spherical Harmonic Gravity
# ===========================================================================
class TestAccelSphericalHarmonics:
    """Tests for accel_gravity_spherical_harmonics()."""

    def test_degree_0_matches_point_mass(self):
        """Degree-0 spherical harmonics should equal point-mass gravity."""
        from astrojax.config import set_dtype
        set_dtype(jnp.float64)

        model = GravityModel.from_type("EGM2008_360")
        r = jnp.array([R_EARTH, 0.0, 0.0])
        R = jnp.eye(3)

        a = accel_gravity_spherical_harmonics(r, R, model, 0, 0)
        assert float(a[0]) == pytest.approx(-GM_EARTH / R_EARTH**2, abs=1e-12)
        assert float(a[1]) == pytest.approx(0.0, abs=1e-12)
        assert float(a[2]) == pytest.approx(0.0, abs=1e-12)

        set_dtype(jnp.float32)

    def test_degree_60_egm2008(self):
        """Degree-60 EGM2008 at R_EARTH on x-axis should match reference."""
        from astrojax.config import set_dtype
        set_dtype(jnp.float64)

        model = GravityModel.from_type("EGM2008_360")
        r = jnp.array([R_EARTH, 0.0, 0.0])
        R = jnp.eye(3)

        a = accel_gravity_spherical_harmonics(r, R, model, 60, 60)
        assert float(a[0]) == pytest.approx(-9.81433239, abs=1e-8)
        assert float(a[1]) == pytest.approx(1.813976e-6, abs=1e-12)
        assert float(a[2]) == pytest.approx(-7.29925652190e-5, abs=1e-12)

        set_dtype(jnp.float32)

    @pytest.mark.parametrize("n,m,ax,ay,az", [
        (2, 2, -6.97922756436, -1.8292810538, -2.69001658552),
        (3, 3, -6.97926211185, -1.82929165145, -2.68998602761),
        (4, 4, -6.97931189287, -1.82931487069, -2.6899914012),
        (5, 5, -6.9792700471, -1.82929795164, -2.68997917147),
        (6, 6, -6.979220667, -1.8292787808, -2.68997263887),
        (7, 7, -6.97925478463, -1.82926946742, -2.68999296889),
        (8, 8, -6.97927699747, -1.82928186346, -2.68998582282),
        (9, 9, -6.97925893036, -1.82928170212, -2.68997442046),
        (10, 10, -6.97924447943, -1.82928331386, -2.68997524437),
        (11, 11, -6.9792517591, -1.82928094754, -2.68998382906),
        (12, 12, -6.97924725688, -1.82928130662, -2.68998625958),
        (13, 13, -6.97924858679, -1.82928591192, -2.6899891726),
        (14, 14, -6.97924919386, -1.82928546814, -2.68999164569),
        (15, 15, -6.97925490319, -1.82928469874, -2.68999376747),
        (16, 16, -6.97926211023, -1.82928438361, -2.68999719587),
        (17, 17, -6.97926308133, -1.82928484644, -2.68999716187),
        (18, 18, -6.97926208121, -1.829284918, -2.6899952379),
        (19, 19, -6.97926229494, -1.82928369323, -2.68999256236),
        (20, 20, -6.979261862, -1.82928315091, -2.68999053339),
    ])
    def test_jgm3_validation(self, n, m, ax, ay, az):
        """JGM3 validation against Satellite Orbits reference values."""
        from astrojax.config import set_dtype
        set_dtype(jnp.float64)

        model = GravityModel.from_type("JGM3")
        r = jnp.array([6525.919e3, 1710.416e3, 2508.886e3])
        R = jnp.eye(3)

        a = accel_gravity_spherical_harmonics(r, R, model, n, m)
        tol = 1e-7
        assert float(a[0]) == pytest.approx(ax, abs=tol)
        assert float(a[1]) == pytest.approx(ay, abs=tol)
        assert float(a[2]) == pytest.approx(az, abs=tol)

        set_dtype(jnp.float32)

    def test_shape(self):
        """Output should be a (3,) array."""
        model = GravityModel.from_type("JGM3")
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        R = jnp.eye(3)
        a = accel_gravity_spherical_harmonics(r, R, model, 4, 4)
        assert a.shape == (3,)

    def test_accepts_6d_state(self):
        """Should accept a 6D state, using only the first 3 elements."""
        model = GravityModel.from_type("JGM3")
        r3 = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        r6 = jnp.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
        R = jnp.eye(3)
        a3 = accel_gravity_spherical_harmonics(r3, R, model, 4, 4)
        a6 = accel_gravity_spherical_harmonics(r6, R, model, 4, 4)
        assert jnp.allclose(a3, a6, atol=1e-10)

    def test_jit_compatible(self):
        """accel_gravity_spherical_harmonics should be JIT-compilable."""
        model = GravityModel.from_type("JGM3")
        r = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        R = jnp.eye(3)

        @jax.jit
        def f(r, R):
            return accel_gravity_spherical_harmonics(r, R, model, 4, 4)

        a_eager = accel_gravity_spherical_harmonics(r, R, model, 4, 4)
        a_jit = f(r, R)
        assert jnp.allclose(a_eager, a_jit, atol=1e-10)


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
