"""Cross-validation tests comparing astrojax.orbit_dynamics against brahe 1.0+.

Both libraries implement the same Montenbruck & Gill low-precision
ephemeris and the same Harris-Priester density model.  Brahe applies an
additional GCRF frame bias rotation (~17 mas) to ephemeris output that
astrojax omits (outputting EME2000 directly), so ephemeris-dependent
quantities carry a small systematic offset.

Tolerance hierarchy:
  - Point-mass gravity: exact match (same formula, no frame dependence)
  - Density (same Sun vector): ~1e-7 relative (slight Sun position diff)
  - Third-body Sun acceleration: ~1e-4 relative (ephemeris diff propagates)
  - Third-body Moon acceleration: ~5e-4 relative (larger Moon position diff)
  - SRP / eclipse: ~1e-4 relative (Sun position diff)
  - Drag: ~1e-4 relative (density depends on Sun position)
"""

import brahe as bh
import jax.numpy as jnp
import numpy as np
import pytest

from astrojax import Epoch
from astrojax.config import set_dtype
from astrojax.constants import GM_EARTH, GM_SUN
from astrojax.orbit_dynamics import (
    GravityModel,
    accel_drag,
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


@pytest.fixture(autouse=True)
def _use_float64():
    """Set float64 for each test, restore float32 afterward."""
    set_dtype(jnp.float64)
    yield
    set_dtype(jnp.float32)


# Tolerances — account for GCRF frame bias and float64 precision
_EPHEM_SUN_ATOL = 5e6       # m  (Sun position diff due to GCRF bias ~2 km)
_EPHEM_MOON_ATOL = 1e5      # m  (Moon position diff ~67 km)
_ACCEL_SUN_RTOL = 5e-4      # relative (third-body Sun, ephemeris diff)
_ACCEL_MOON_RTOL = 5e-3     # relative (third-body Moon, larger ephemeris diff)
_GRAVITY_ATOL = 1e-10       # m/s^2 (exact same formula)
_DENSITY_RTOL = 1e-5        # relative (density depends on Sun position)
_SRP_RTOL = 2e-4            # relative (SRP depends on Sun position)
_DRAG_RTOL = 5e-4           # relative (drag via density via Sun position)


# ──────────────────────────────────────────────
# Epoch helpers
# ──────────────────────────────────────────────

def _make_epochs(year, month, day, hour=0, minute=0, second=0.0):
    """Create matched brahe and astrojax Epoch objects."""
    epc_bh = bh.Epoch(year, month, day, hour, minute, second)
    epc_aj = Epoch(year, month, day, hour, minute, second)
    return epc_bh, epc_aj


# ──────────────────────────────────────────────
# Sun Position
# ──────────────────────────────────────────────

class TestSunPositionVsBrahe:
    """Validate sun_position against brahe.sun_position."""

    @pytest.mark.parametrize("year,month,day,hour", [
        (2024, 1, 1, 0),
        (2024, 1, 1, 12),
        (2024, 1, 2, 0),
        (2024, 6, 15, 12),
        (2024, 12, 21, 0),
    ])
    def test_sun_position(self, year, month, day, hour):
        """Sun position should agree with brahe within frame bias tolerance."""
        epc_bh, epc_aj = _make_epochs(year, month, day, hour)
        expected = bh.sun_position(epc_bh)
        actual = np.array(sun_position(epc_aj))
        np.testing.assert_allclose(
            actual, expected,
            atol=_EPHEM_SUN_ATOL, rtol=1e-4,
            err_msg=f"sun_position mismatch at {year}-{month:02d}-{day:02d}T{hour:02d}",
        )


# ──────────────────────────────────────────────
# Moon Position
# ──────────────────────────────────────────────

class TestMoonPositionVsBrahe:
    """Validate moon_position against brahe.moon_position."""

    @pytest.mark.parametrize("year,month,day,hour", [
        (2024, 1, 1, 0),
        (2024, 1, 1, 12),
        (2024, 1, 2, 0),
        (2024, 6, 15, 12),
        (2024, 12, 21, 0),
    ])
    def test_moon_position(self, year, month, day, hour):
        """Moon position should agree with brahe within frame bias tolerance."""
        epc_bh, epc_aj = _make_epochs(year, month, day, hour)
        expected = bh.moon_position(epc_bh)
        actual = np.array(moon_position(epc_aj))
        np.testing.assert_allclose(
            actual, expected,
            atol=_EPHEM_MOON_ATOL, rtol=1e-3,
            err_msg=f"moon_position mismatch at {year}-{month:02d}-{day:02d}T{hour:02d}",
        )


# ──────────────────────────────────────────────
# Point-Mass Gravity
# ──────────────────────────────────────────────

class TestPointMassGravityVsBrahe:
    """Validate accel_point_mass against brahe.accel_point_mass_gravity."""

    @pytest.mark.parametrize("alt_km", [200, 500, 800, 2000, 35786])
    def test_gravity_radial(self, alt_km):
        """Point-mass gravity on x-axis matches brahe."""
        r = np.array([bh.R_EARTH + alt_km * 1e3, 0.0, 0.0])
        r_body = np.zeros(3)
        expected = bh.accel_point_mass_gravity(r, r_body, bh.GM_EARTH)
        actual = np.array(accel_point_mass(jnp.array(r), jnp.zeros(3), GM_EARTH))
        np.testing.assert_allclose(actual, expected, atol=_GRAVITY_ATOL)

    def test_gravity_off_axis(self):
        """Point-mass gravity at arbitrary position matches brahe."""
        r = np.array([4000e3, 3000e3, 2000e3])
        r_body = np.zeros(3)
        expected = bh.accel_point_mass_gravity(r, r_body, bh.GM_EARTH)
        actual = np.array(accel_point_mass(jnp.array(r), jnp.zeros(3), GM_EARTH))
        np.testing.assert_allclose(actual, expected, atol=_GRAVITY_ATOL)

    def test_third_body_form(self):
        """Third-body form with non-zero central body position matches brahe."""
        r = np.array([6878e3, 0.0, 0.0])
        epc_bh = bh.Epoch(2024, 1, 1)
        r_sun = bh.sun_position(epc_bh)
        expected = bh.accel_point_mass_gravity(r, r_sun, bh.GM_SUN)
        actual = np.array(accel_point_mass(jnp.array(r), jnp.array(r_sun), GM_SUN))
        np.testing.assert_allclose(actual, expected, atol=1e-14)


# ──────────────────────────────────────────────
# Third-Body Perturbations
# ──────────────────────────────────────────────

_LEO_POSITIONS = [
    [6878e3, 0.0, 0.0],
    [0.0, 6878e3, 0.0],
    [0.0, 0.0, 6878e3],
    [4884992.3, 4553508.5, 1330313.6],
    [-6891477.8, -227551.8, 663813.9],
    [6790850.7, 45505.4, -727399.8],
    [-4983679.0, -4645498.6, -1357188.6],
    [2383524.6, -5900075.6, -2680956.2],
]


class TestThirdBodySunVsBrahe:
    """Validate accel_third_body_sun against brahe."""

    @pytest.mark.parametrize("r_eci", _LEO_POSITIONS)
    def test_accel_third_body_sun(self, r_eci):
        """Third-body Sun acceleration should agree with brahe."""
        epc_bh, epc_aj = _make_epochs(2024, 1, 1)
        r = np.array(r_eci)
        expected = bh.accel_third_body_sun(epc_bh, r)
        actual = np.array(accel_third_body_sun(epc_aj, jnp.array(r)))
        np.testing.assert_allclose(
            actual, expected,
            rtol=_ACCEL_SUN_RTOL, atol=1e-14,
            err_msg=f"third_body_sun mismatch for r={r_eci}",
        )


class TestThirdBodyMoonVsBrahe:
    """Validate accel_third_body_moon against brahe."""

    @pytest.mark.parametrize("r_eci", _LEO_POSITIONS)
    def test_accel_third_body_moon(self, r_eci):
        """Third-body Moon acceleration should agree with brahe."""
        epc_bh, epc_aj = _make_epochs(2024, 1, 1)
        r = np.array(r_eci)
        expected = bh.accel_third_body_moon(epc_bh, r)
        actual = np.array(accel_third_body_moon(epc_aj, jnp.array(r)))
        np.testing.assert_allclose(
            actual, expected,
            rtol=_ACCEL_MOON_RTOL, atol=1e-13,
            err_msg=f"third_body_moon mismatch for r={r_eci}",
        )


# ──────────────────────────────────────────────
# Harris-Priester Atmospheric Density
# ──────────────────────────────────────────────

_HP_POSITIONS = [
    [0.0, 0.0, -(bh.R_EARTH + 100e3)],     # 100 km altitude
    [0.0, 0.0, -(bh.R_EARTH + 200e3)],     # 200 km
    [0.0, 0.0, -(bh.R_EARTH + 400e3)],     # 400 km
    [0.0, 0.0, -(bh.R_EARTH + 600e3)],     # 600 km
    [0.0, 0.0, -(bh.R_EARTH + 800e3)],     # 800 km
    [0.0, 0.0, -(bh.R_EARTH + 1000e3)],    # 1000 km
    [4595372.625, 0.0, -4565130.155],       # off-axis ~88 km
    [4666083.303, 0.0, -4635840.833],       # off-axis ~188 km
    [4807504.659, 0.0, -4777262.189],       # off-axis ~388 km
]


class TestHarrisPriesterVsBrahe:
    """Validate density_harris_priester against brahe."""

    @pytest.mark.parametrize("r_tod", _HP_POSITIONS)
    def test_density_harris_priester(self, r_tod):
        """Harris-Priester density should agree with brahe.

        Both are called with the same Sun position vector from brahe
        to isolate the density algorithm from ephemeris differences.
        """
        epc_bh = bh.Epoch(2024, 1, 1)
        r_sun_bh = bh.sun_position(epc_bh)

        r = np.array(r_tod)
        expected = bh.density_harris_priester(r, r_sun_bh)

        # Use brahe's Sun position for both to isolate density algorithm
        actual = float(density_harris_priester(
            jnp.array(r), jnp.array(r_sun_bh)
        ))

        if expected == 0.0:
            assert actual == 0.0
        else:
            np.testing.assert_allclose(
                actual, expected,
                rtol=_DENSITY_RTOL,
                err_msg=f"HP density mismatch for r={r_tod}",
            )


# ──────────────────────────────────────────────
# Atmospheric Drag
# ──────────────────────────────────────────────

_DRAG_STATES = [
    [6878e3, 0.0, 0.0, 0.0, 7500.0, 0.0],
    [4884992.3, 4553508.5, 1330313.6, -5300.0, 4925.7, 2601.9],
    [0.0, 6878e3, 0.0, -7500.0, 0.0, 0.0],
]


class TestDragVsBrahe:
    """Validate accel_drag against brahe.

    Uses brahe's density_harris_priester with identity rotation
    to isolate the drag algorithm from density differences.
    """

    @pytest.mark.parametrize("state", _DRAG_STATES)
    def test_accel_drag(self, state):
        """Drag acceleration should agree with brahe."""
        epc_bh = bh.Epoch(2024, 1, 1)
        r_sun_bh = bh.sun_position(epc_bh)

        x = np.array(state)
        r = x[:3]
        T = np.eye(3)

        # Use brahe's density for both to isolate drag algorithm
        rho = bh.density_harris_priester(r, r_sun_bh)
        if rho == 0.0:
            return  # skip if no atmosphere

        mass, area, cd = 100.0, 1.0, 2.3
        expected = bh.accel_drag(x, rho, mass, area, cd, T)
        actual = np.array(accel_drag(
            jnp.array(x), rho, mass, area, cd, jnp.eye(3)
        ))

        np.testing.assert_allclose(
            actual, expected,
            rtol=1e-10, atol=1e-16,
            err_msg=f"drag mismatch for state={state}",
        )


# ──────────────────────────────────────────────
# Solar Radiation Pressure
# ──────────────────────────────────────────────

class TestSRPVsBrahe:
    """Validate accel_srp against brahe.accel_solar_radiation_pressure."""

    @pytest.mark.parametrize("r_eci", _LEO_POSITIONS)
    def test_accel_srp(self, r_eci):
        """SRP acceleration should agree with brahe."""
        epc_bh = bh.Epoch(2024, 1, 1)
        r_sun_bh = bh.sun_position(epc_bh)

        r = np.array(r_eci)
        mass, cr, area, p0 = 100.0, 1.8, 1.0, 4.56e-6

        expected = bh.accel_solar_radiation_pressure(r, r_sun_bh, mass, cr, area, p0)
        # Use brahe's Sun position for both to isolate SRP algorithm
        actual = np.array(accel_srp(
            jnp.array(r), jnp.array(r_sun_bh), mass, cr, area, p0
        ))

        np.testing.assert_allclose(
            actual, expected,
            rtol=1e-10, atol=1e-16,
            err_msg=f"SRP mismatch for r={r_eci}",
        )


# ──────────────────────────────────────────────
# Eclipse Models
# ──────────────────────────────────────────────

_ECLIPSE_POSITIONS = [
    [6878e3, 0.0, 0.0],                       # sunlit
    [-(6378e3 + 500e3), 0.0, 0.0],           # shadow (behind Earth)
    [0.0, 6878e3, 0.0],                       # orthogonal
    [-6891477.8, -227551.8, 663813.9],         # LEO arbitrary
    [6790850.7, 45505.4, -727399.8],           # LEO arbitrary
]


class TestEclipseCylindricalVsBrahe:
    """Validate eclipse_cylindrical against brahe."""

    @pytest.mark.parametrize("r_eci", _ECLIPSE_POSITIONS)
    def test_eclipse_cylindrical(self, r_eci):
        """Cylindrical eclipse model should match brahe."""
        epc_bh = bh.Epoch(2024, 1, 1)
        r_sun_bh = bh.sun_position(epc_bh)

        r = np.array(r_eci)
        expected = bh.eclipse_cylindrical(r, r_sun_bh)
        # Use brahe's Sun position for both
        actual = float(eclipse_cylindrical(
            jnp.array(r), jnp.array(r_sun_bh)
        ))

        assert actual == pytest.approx(expected, abs=1e-10), \
            f"eclipse_cylindrical: {actual} vs {expected} for r={r_eci}"


class TestEclipseConicalVsBrahe:
    """Validate eclipse_conical against brahe."""

    @pytest.mark.parametrize("r_eci", _ECLIPSE_POSITIONS)
    def test_eclipse_conical(self, r_eci):
        """Conical eclipse model should match brahe."""
        epc_bh = bh.Epoch(2024, 1, 1)
        r_sun_bh = bh.sun_position(epc_bh)

        r = np.array(r_eci)
        expected = bh.eclipse_conical(r, r_sun_bh)
        # Use brahe's Sun position for both
        actual = float(eclipse_conical(
            jnp.array(r), jnp.array(r_sun_bh)
        ))

        assert actual == pytest.approx(expected, abs=1e-6), \
            f"eclipse_conical: {actual} vs {expected} for r={r_eci}"


# ──────────────────────────────────────────────
# End-to-End: Same-epoch comparison
# ──────────────────────────────────────────────

class TestEndToEndVsBrahe:
    """End-to-end comparison using each library's own ephemeris.

    These tests use each library's native Sun/Moon positions (which
    differ by the GCRF frame bias) and verify the acceleration
    magnitudes agree within the expected tolerance.
    """

    @pytest.mark.parametrize("alt_km", [500, 800])
    def test_third_body_sun_native(self, alt_km):
        """Third-body Sun with native ephemeris agrees in magnitude."""
        epc_bh, epc_aj = _make_epochs(2024, 1, 1)
        r = np.array([bh.R_EARTH + alt_km * 1e3, 0.0, 0.0])

        expected = np.linalg.norm(bh.accel_third_body_sun(epc_bh, r))
        actual = float(jnp.linalg.norm(accel_third_body_sun(epc_aj, jnp.array(r))))

        np.testing.assert_allclose(actual, expected, rtol=_ACCEL_SUN_RTOL)

    @pytest.mark.parametrize("alt_km", [500, 800])
    def test_third_body_moon_native(self, alt_km):
        """Third-body Moon with native ephemeris agrees in magnitude."""
        epc_bh, epc_aj = _make_epochs(2024, 1, 1)
        r = np.array([bh.R_EARTH + alt_km * 1e3, 0.0, 0.0])

        expected = np.linalg.norm(bh.accel_third_body_moon(epc_bh, r))
        actual = float(jnp.linalg.norm(accel_third_body_moon(epc_aj, jnp.array(r))))

        np.testing.assert_allclose(actual, expected, rtol=_ACCEL_MOON_RTOL)


# ──────────────────────────────────────────────
# Spherical Harmonic Gravity
# ──────────────────────────────────────────────

_SH_POSITIONS = [
    [6878e3, 0.0, 0.0],
    [0.0, 6878e3, 0.0],
    [0.0, 0.0, 6878e3],
    [6525.919e3, 1710.416e3, 2508.886e3],
    [4884992.3, 4553508.5, 1330313.6],
]

_SH_DEGREE_ORDERS = [
    (4, 4),
    (10, 10),
    (20, 20),
    (40, 40),
]


class TestSphericalHarmonicsVsBrahe:
    """Validate accel_gravity_spherical_harmonics against brahe.

    Uses identity rotation for both libraries to isolate the algorithm
    from frame transformation differences.
    """

    @pytest.mark.parametrize("r_eci", _SH_POSITIONS)
    @pytest.mark.parametrize("n_max,m_max", _SH_DEGREE_ORDERS)
    def test_accel_spherical_harmonics(self, r_eci, n_max, m_max):
        """Spherical harmonic acceleration should agree with brahe."""
        r = np.array(r_eci)
        R_identity = np.eye(3)

        # brahe uses GravityModelType enum for loading
        bh_model = bh.GravityModel.from_model_type(bh.GravityModelType.JGM3)
        expected = bh.accel_gravity_spherical_harmonics(
            r, R_identity, bh_model, n_max, m_max,
        )

        aj_model = GravityModel.from_type("JGM3")
        actual = np.array(accel_gravity_spherical_harmonics(
            jnp.array(r), jnp.eye(3), aj_model, n_max, m_max,
        ))

        np.testing.assert_allclose(
            actual, expected,
            rtol=1e-9, atol=1e-12,
            err_msg=f"SH mismatch for r={r_eci}, n={n_max}, m={m_max}",
        )
