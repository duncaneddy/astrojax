"""Cross-validation tests comparing astrojax.orbits output against brahe 1.0+.

Brahe uses float64 internally while astrojax uses float32. Tolerances
are set to accommodate the precision difference (~7 significant digits
for float32 vs ~15 for float64).
"""

import numpy as np
import pytest

import brahe as bh

import jax.numpy as jnp

from astrojax.orbits import (
    anomaly_eccentric_to_mean,
    anomaly_eccentric_to_true,
    anomaly_mean_to_eccentric,
    anomaly_mean_to_true,
    anomaly_true_to_eccentric,
    anomaly_true_to_mean,
    apoapsis_distance,
    apogee_altitude,
    apogee_velocity,
    geo_sma,
    mean_motion,
    orbital_period,
    orbital_period_from_state,
    perigee_altitude,
    perigee_velocity,
    periapsis_distance,
    semimajor_axis,
    semimajor_axis_from_orbital_period,
    state_koe_mean_to_osc,
    state_koe_osc_to_mean,
    sun_synchronous_inclination,
)

DEGREES = bh.AngleFormat.DEGREES
RADIANS = bh.AngleFormat.RADIANS

# Float32 vs float64 tolerances
_REL_TOL = 5e-6   # relative tolerance (~float32 precision)
_PERIOD_ATOL = 1.0       # seconds
_DISTANCE_ATOL = 10.0    # metres
_VELOCITY_ATOL = 0.1     # m/s
_ANGLE_DEG_ATOL = 0.01   # degrees
_ANGLE_RAD_ATOL = 2e-4   # radians


# ──────────────────────────────────────────────
# Orbital period
# ──────────────────────────────────────────────

class TestOrbitalPeriodVsBrahe:
    @pytest.mark.parametrize("alt_km", [200, 500, 800, 2000, 35786])
    def test_orbital_period(self, alt_km):
        """orbital_period matches brahe across altitude range."""
        a = bh.R_EARTH + alt_km * 1e3
        expected = bh.orbital_period(a)
        actual = float(orbital_period(a))
        assert abs(actual - expected) < _PERIOD_ATOL + abs(expected) * _REL_TOL

    def test_orbital_period_from_state_circular(self):
        """orbital_period_from_state matches brahe for circular orbit."""
        a = bh.R_EARTH + 500e3
        v = (bh.GM_EARTH / a) ** 0.5
        state = np.array([a, 0.0, 0.0, 0.0, v, 0.0])

        expected = bh.orbital_period_from_state(state, bh.GM_EARTH)
        actual = float(orbital_period_from_state(jnp.array(state)))
        assert abs(actual - expected) < _PERIOD_ATOL + abs(expected) * _REL_TOL

    def test_orbital_period_from_state_elliptical(self):
        """orbital_period_from_state matches brahe for elliptical orbit."""
        a = bh.R_EARTH + 500e3
        e = 0.1
        r_p = a * (1.0 - e)
        v_p = (bh.GM_EARTH * (2.0 / r_p - 1.0 / a)) ** 0.5
        state = np.array([r_p, 0.0, 0.0, 0.0, v_p, 0.0])

        expected = bh.orbital_period_from_state(state, bh.GM_EARTH)
        actual = float(orbital_period_from_state(jnp.array(state)))
        assert abs(actual - expected) < _PERIOD_ATOL + abs(expected) * _REL_TOL


# ──────────────────────────────────────────────
# Semi-major axis
# ──────────────────────────────────────────────

class TestSemimajorAxisVsBrahe:
    def test_semimajor_axis_from_period(self):
        """semimajor_axis_from_orbital_period matches brahe."""
        T = bh.orbital_period(bh.R_EARTH + 500e3)
        expected = bh.semimajor_axis_from_orbital_period(T)
        actual = float(semimajor_axis_from_orbital_period(T))
        assert abs(actual - expected) < _DISTANCE_ATOL + abs(expected) * _REL_TOL

    def test_semimajor_axis_from_mean_motion_radians(self):
        """semimajor_axis(n, radians) matches brahe."""
        a = bh.R_EARTH + 500e3
        n = bh.mean_motion(a, RADIANS)
        expected = bh.semimajor_axis(n, RADIANS)
        actual = float(semimajor_axis(n, use_degrees=False))
        assert abs(actual - expected) < _DISTANCE_ATOL + abs(expected) * _REL_TOL

    def test_semimajor_axis_from_mean_motion_degrees(self):
        """semimajor_axis(n, degrees) matches brahe."""
        a = bh.R_EARTH + 500e3
        n = bh.mean_motion(a, DEGREES)
        expected = bh.semimajor_axis(n, DEGREES)
        actual = float(semimajor_axis(n, use_degrees=True))
        assert abs(actual - expected) < _DISTANCE_ATOL + abs(expected) * _REL_TOL


# ──────────────────────────────────────────────
# Mean motion
# ──────────────────────────────────────────────

class TestMeanMotionVsBrahe:
    @pytest.mark.parametrize("alt_km", [200, 500, 800, 2000])
    def test_mean_motion_radians(self, alt_km):
        """mean_motion in radians matches brahe across altitudes."""
        a = bh.R_EARTH + alt_km * 1e3
        expected = bh.mean_motion(a, RADIANS)
        actual = float(mean_motion(a, use_degrees=False))
        assert abs(actual - expected) < abs(expected) * _REL_TOL

    @pytest.mark.parametrize("alt_km", [200, 500, 800, 2000])
    def test_mean_motion_degrees(self, alt_km):
        """mean_motion in degrees matches brahe across altitudes."""
        a = bh.R_EARTH + alt_km * 1e3
        expected = bh.mean_motion(a, DEGREES)
        actual = float(mean_motion(a, use_degrees=True))
        assert abs(actual - expected) < abs(expected) * _REL_TOL


# ──────────────────────────────────────────────
# Velocities at apsides
# ──────────────────────────────────────────────

class TestVelocityVsBrahe:
    @pytest.mark.parametrize("e", [0.0, 0.001, 0.01, 0.1, 0.5])
    def test_perigee_velocity(self, e):
        """perigee_velocity matches brahe across eccentricities."""
        a = bh.R_EARTH + 500e3
        expected = bh.perigee_velocity(a, e)
        actual = float(perigee_velocity(a, e))
        assert abs(actual - expected) < _VELOCITY_ATOL + abs(expected) * _REL_TOL

    @pytest.mark.parametrize("e", [0.0, 0.001, 0.01, 0.1, 0.5])
    def test_apogee_velocity(self, e):
        """apogee_velocity matches brahe across eccentricities."""
        a = bh.R_EARTH + 500e3
        expected = bh.apogee_velocity(a, e)
        actual = float(apogee_velocity(a, e))
        assert abs(actual - expected) < _VELOCITY_ATOL + abs(expected) * _REL_TOL


# ──────────────────────────────────────────────
# Distances and altitudes
# ──────────────────────────────────────────────

class TestDistanceAltitudeVsBrahe:
    @pytest.mark.parametrize("e", [0.0, 0.01, 0.1, 0.5])
    def test_periapsis_distance(self, e):
        """periapsis_distance matches brahe."""
        a = bh.R_EARTH + 500e3
        expected = bh.periapsis_distance(a, e)
        actual = float(periapsis_distance(a, e))
        assert abs(actual - expected) < _DISTANCE_ATOL

    @pytest.mark.parametrize("e", [0.0, 0.01, 0.1, 0.5])
    def test_apoapsis_distance(self, e):
        """apoapsis_distance matches brahe."""
        a = bh.R_EARTH + 500e3
        expected = bh.apoapsis_distance(a, e)
        actual = float(apoapsis_distance(a, e))
        assert abs(actual - expected) < _DISTANCE_ATOL

    @pytest.mark.parametrize("e", [0.0, 0.001, 0.01, 0.1])
    def test_perigee_altitude(self, e):
        """perigee_altitude matches brahe."""
        a = bh.R_EARTH + 500e3
        expected = bh.perigee_altitude(a, e)
        actual = float(perigee_altitude(a, e))
        assert abs(actual - expected) < _DISTANCE_ATOL + abs(expected) * _REL_TOL

    @pytest.mark.parametrize("e", [0.0, 0.001, 0.01, 0.1])
    def test_apogee_altitude(self, e):
        """apogee_altitude matches brahe."""
        a = bh.R_EARTH + 500e3
        expected = bh.apogee_altitude(a, e)
        actual = float(apogee_altitude(a, e))
        assert abs(actual - expected) < _DISTANCE_ATOL + abs(expected) * _REL_TOL


# ──────────────────────────────────────────────
# Special orbits
# ──────────────────────────────────────────────

class TestSpecialOrbitsVsBrahe:
    @pytest.mark.parametrize("alt_km", [400, 500, 600, 800])
    def test_sun_synchronous_inclination(self, alt_km):
        """sun_synchronous_inclination matches brahe across altitudes."""
        a = bh.R_EARTH + alt_km * 1e3
        expected = bh.sun_synchronous_inclination(
            a, 0.001, angle_format=DEGREES
        )
        actual = float(sun_synchronous_inclination(a, 0.001, use_degrees=True))
        assert abs(actual - expected) < _ANGLE_DEG_ATOL + abs(expected) * _REL_TOL

    def test_geo_sma(self):
        """geo_sma matches brahe."""
        expected = bh.geo_sma()
        actual = float(geo_sma())
        assert abs(actual - expected) < 100.0 + abs(expected) * _REL_TOL


# ──────────────────────────────────────────────
# Anomaly conversions — value tests
# ──────────────────────────────────────────────

class TestAnomalyConversionsVsBrahe:
    @pytest.mark.parametrize("E_deg", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0])
    @pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.5, 0.7])
    def test_eccentric_to_mean_degrees(self, E_deg, e):
        """anomaly_eccentric_to_mean matches brahe in degrees."""
        expected = bh.anomaly_eccentric_to_mean(
            E_deg, e, angle_format=DEGREES
        )
        actual = float(anomaly_eccentric_to_mean(E_deg, e, use_degrees=True))
        assert abs(actual - expected) < _ANGLE_DEG_ATOL + abs(expected) * _REL_TOL

    @pytest.mark.parametrize("M_deg", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0])
    @pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.5, 0.7])
    def test_mean_to_eccentric_degrees(self, M_deg, e):
        """anomaly_mean_to_eccentric matches brahe in degrees."""
        expected = bh.anomaly_mean_to_eccentric(
            M_deg, e, angle_format=DEGREES
        )
        actual = float(anomaly_mean_to_eccentric(M_deg, e, use_degrees=True))
        assert abs(actual - expected) < _ANGLE_DEG_ATOL + abs(expected) * _REL_TOL

    @pytest.mark.parametrize("nu_deg", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0])
    @pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.5, 0.7])
    def test_true_to_eccentric_degrees(self, nu_deg, e):
        """anomaly_true_to_eccentric matches brahe in degrees."""
        expected = bh.anomaly_true_to_eccentric(
            nu_deg, e, angle_format=DEGREES
        )
        actual = float(anomaly_true_to_eccentric(nu_deg, e, use_degrees=True))
        assert abs(actual - expected) < _ANGLE_DEG_ATOL + abs(expected) * _REL_TOL

    @pytest.mark.parametrize("E_deg", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0])
    @pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.5, 0.7])
    def test_eccentric_to_true_degrees(self, E_deg, e):
        """anomaly_eccentric_to_true matches brahe in degrees."""
        expected = bh.anomaly_eccentric_to_true(
            E_deg, e, angle_format=DEGREES
        )
        actual = float(anomaly_eccentric_to_true(E_deg, e, use_degrees=True))
        assert abs(actual - expected) < _ANGLE_DEG_ATOL + abs(expected) * _REL_TOL

    @pytest.mark.parametrize("nu_deg", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0])
    @pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.5, 0.7])
    def test_true_to_mean_degrees(self, nu_deg, e):
        """anomaly_true_to_mean matches brahe in degrees."""
        expected = bh.anomaly_true_to_mean(
            nu_deg, e, angle_format=DEGREES
        )
        actual = float(anomaly_true_to_mean(nu_deg, e, use_degrees=True))
        assert abs(actual - expected) < _ANGLE_DEG_ATOL + abs(expected) * _REL_TOL

    @pytest.mark.parametrize("M_deg", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0])
    @pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.5, 0.7])
    def test_mean_to_true_degrees(self, M_deg, e):
        """anomaly_mean_to_true matches brahe in degrees."""
        expected = bh.anomaly_mean_to_true(
            M_deg, e, angle_format=DEGREES
        )
        actual = float(anomaly_mean_to_true(M_deg, e, use_degrees=True))
        assert abs(actual - expected) < _ANGLE_DEG_ATOL + abs(expected) * _REL_TOL


# ──────────────────────────────────────────────
# Anomaly conversions — radians
# ──────────────────────────────────────────────

class TestAnomalyRadiansVsBrahe:
    @pytest.mark.parametrize("E_rad", [0.0, 0.5, 1.0, 1.5707963, 2.5, 3.0])
    def test_eccentric_to_mean_radians(self, E_rad):
        """anomaly_eccentric_to_mean matches brahe in radians."""
        e = 0.1
        expected = bh.anomaly_eccentric_to_mean(
            E_rad, e, angle_format=RADIANS
        )
        actual = float(anomaly_eccentric_to_mean(E_rad, e, use_degrees=False))
        assert abs(actual - expected) < _ANGLE_RAD_ATOL + abs(expected) * _REL_TOL

    @pytest.mark.parametrize("M_rad", [0.0, 0.5, 1.0, 1.5707963, 2.5, 3.0])
    def test_mean_to_eccentric_radians(self, M_rad):
        """anomaly_mean_to_eccentric matches brahe in radians."""
        e = 0.1
        expected = bh.anomaly_mean_to_eccentric(
            M_rad, e, angle_format=RADIANS
        )
        actual = float(anomaly_mean_to_eccentric(M_rad, e, use_degrees=False))
        assert abs(actual - expected) < _ANGLE_RAD_ATOL + abs(expected) * _REL_TOL


# ──────────────────────────────────────────────
# Mean-Osculating element conversions
# ──────────────────────────────────────────────

# Tolerances for mean-osculating comparisons (float32 vs float64)
_MO_SMA_ATOL = 50.0       # metres
_MO_ECC_ATOL = 1e-5       # dimensionless
_MO_ANGLE_RAD_ATOL = 0.01  # radians (~0.6 deg, accommodates float32 at M=pi)
_MO_ANGLE_DEG_ATOL = 0.6   # degrees


def _angle_diff_rad(a, b):
    """Minimum angular distance in radians, handling 0/2pi wraparound."""
    d = abs(a - b) % (2.0 * np.pi)
    return min(d, 2.0 * np.pi - d)


def _angle_diff_deg(a, b):
    """Minimum angular distance in degrees, handling 0/360 wraparound."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _compare_koe_rad(actual_jax, expected_brahe, sma_atol=_MO_SMA_ATOL,
                     ecc_atol=_MO_ECC_ATOL, angle_atol=_MO_ANGLE_RAD_ATOL):
    """Compare two KOE vectors element-by-element (radians)."""
    actual = np.array(actual_jax)
    expected = np.array(expected_brahe)
    assert abs(actual[0] - expected[0]) < sma_atol, (
        f"SMA: {actual[0]:.3f} vs {expected[0]:.3f}, diff={abs(actual[0]-expected[0]):.3f}"
    )
    assert abs(actual[1] - expected[1]) < ecc_atol, (
        f"ecc: {actual[1]:.8f} vs {expected[1]:.8f}, diff={abs(actual[1]-expected[1]):.8f}"
    )
    for idx, name in [(2, "inc"), (3, "raan"), (4, "argp"), (5, "M")]:
        diff = _angle_diff_rad(float(actual[idx]), float(expected[idx]))
        assert diff < angle_atol, (
            f"{name}: {actual[idx]:.8f} vs {expected[idx]:.8f}, diff={diff:.8f}"
        )


def _compare_koe_deg(actual_jax, expected_brahe, sma_atol=_MO_SMA_ATOL,
                     ecc_atol=_MO_ECC_ATOL, angle_atol=_MO_ANGLE_DEG_ATOL):
    """Compare two KOE vectors element-by-element (degrees)."""
    actual = np.array(actual_jax)
    expected = np.array(expected_brahe)
    assert abs(actual[0] - expected[0]) < sma_atol
    assert abs(actual[1] - expected[1]) < ecc_atol
    for idx in range(2, 6):
        diff = _angle_diff_deg(float(actual[idx]), float(expected[idx]))
        assert diff < angle_atol


class TestMeanToOscVsBrahe:
    def test_leo_radians(self):
        """state_koe_mean_to_osc matches brahe for LEO (radians)."""
        oe = np.array([
            bh.R_EARTH + 500e3, 0.01,
            np.radians(45.0), np.radians(30.0),
            np.radians(60.0), np.radians(90.0),
        ])
        expected = bh.state_koe_mean_to_osc(oe, RADIANS)
        actual = state_koe_mean_to_osc(jnp.array(oe))
        _compare_koe_rad(actual, expected)

    def test_leo_degrees(self):
        """state_koe_mean_to_osc matches brahe for LEO (degrees)."""
        oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
        expected = bh.state_koe_mean_to_osc(oe, DEGREES)
        actual = state_koe_mean_to_osc(jnp.array(oe), use_degrees=True)
        _compare_koe_deg(actual, expected)

    @pytest.mark.parametrize("m_deg", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    def test_mean_to_osc_various_M(self, m_deg):
        """state_koe_mean_to_osc matches brahe across mean anomaly values."""
        oe = np.array([
            bh.R_EARTH + 500e3, 0.01,
            np.radians(45.0), np.radians(30.0),
            np.radians(60.0), np.radians(m_deg),
        ])
        expected = bh.state_koe_mean_to_osc(oe, RADIANS)
        actual = state_koe_mean_to_osc(jnp.array(oe))
        _compare_koe_rad(actual, expected)

    @pytest.mark.parametrize("ecc", [0.0001, 0.001, 0.01, 0.1])
    def test_mean_to_osc_various_e(self, ecc):
        """state_koe_mean_to_osc matches brahe across eccentricity values."""
        oe = np.array([
            bh.R_EARTH + 500e3, ecc,
            np.radians(45.0), np.radians(30.0),
            np.radians(60.0), np.radians(90.0),
        ])
        expected = bh.state_koe_mean_to_osc(oe, RADIANS)
        actual = state_koe_mean_to_osc(jnp.array(oe))
        _compare_koe_rad(actual, expected)

    def test_geo(self):
        """state_koe_mean_to_osc matches brahe for GEO orbit."""
        oe = np.array([
            42164e3, 0.0001,
            np.radians(0.1), np.radians(45.0), 0.0, 0.0,
        ])
        expected = bh.state_koe_mean_to_osc(oe, RADIANS)
        actual = state_koe_mean_to_osc(jnp.array(oe))
        _compare_koe_rad(actual, expected, sma_atol=10.0)

    def test_sun_synchronous(self):
        """state_koe_mean_to_osc matches brahe for sun-sync orbit."""
        oe = np.array([
            bh.R_EARTH + 700e3, 0.001,
            np.radians(98.0), np.radians(45.0),
            np.radians(90.0), np.radians(270.0),
        ])
        expected = bh.state_koe_mean_to_osc(oe, RADIANS)
        actual = state_koe_mean_to_osc(jnp.array(oe))
        _compare_koe_rad(actual, expected)


class TestOscToMeanVsBrahe:
    def test_leo_radians(self):
        """state_koe_osc_to_mean matches brahe for LEO (radians)."""
        oe = np.array([
            bh.R_EARTH + 500e3, 0.01,
            np.radians(45.0), np.radians(30.0),
            np.radians(60.0), np.radians(90.0),
        ])
        expected = bh.state_koe_osc_to_mean(oe, RADIANS)
        actual = state_koe_osc_to_mean(jnp.array(oe))
        _compare_koe_rad(actual, expected)

    def test_leo_degrees(self):
        """state_koe_osc_to_mean matches brahe for LEO (degrees)."""
        oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
        expected = bh.state_koe_osc_to_mean(oe, DEGREES)
        actual = state_koe_osc_to_mean(jnp.array(oe), use_degrees=True)
        _compare_koe_deg(actual, expected)

    @pytest.mark.parametrize("m_deg", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    def test_osc_to_mean_various_M(self, m_deg):
        """state_koe_osc_to_mean matches brahe across mean anomaly values."""
        oe = np.array([
            bh.R_EARTH + 500e3, 0.01,
            np.radians(45.0), np.radians(30.0),
            np.radians(60.0), np.radians(m_deg),
        ])
        expected = bh.state_koe_osc_to_mean(oe, RADIANS)
        actual = state_koe_osc_to_mean(jnp.array(oe))
        _compare_koe_rad(actual, expected)
