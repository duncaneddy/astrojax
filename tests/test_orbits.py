import jax
import jax.numpy as jnp
import pytest

from astrojax.constants import GM_EARTH, R_EARTH
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

# Tolerances for float32 arithmetic
_PERIOD_TOL = 1.0        # seconds
_DISTANCE_TOL = 1.0      # metres
_VELOCITY_TOL = 0.1      # m/s
_ANOMALY_RAD_TOL = 1e-4  # radians
_ANOMALY_DEG_TOL = 0.01  # degrees
_ROUNDTRIP_DEG_TOL = 0.1 # degrees (anomaly roundtrips)
_SMA_TOL = 10.0          # metres
_SSI_DEG_TOL = 0.1       # degrees
_GEO_SMA_TOL = 100.0     # metres

# Standard test orbit: 500 km circular LEO
_SMA_500 = R_EARTH + 500e3


# ──────────────────────────────────────────────
# Orbital period
# ──────────────────────────────────────────────

class TestOrbitalPeriod:
    def test_orbital_period_500km(self):
        """Period of a 500 km LEO orbit matches reference value."""
        T = orbital_period(_SMA_500)
        assert jnp.abs(T - 5676.977) < _PERIOD_TOL

    def test_orbital_period_from_state_circular(self):
        """Period from circular orbit state vector matches direct computation."""
        r = _SMA_500
        v = jnp.sqrt(GM_EARTH / r)
        state = jnp.array([r, 0.0, 0.0, 0.0, v, 0.0])
        T_state = orbital_period_from_state(state)
        T_direct = orbital_period(r)
        assert jnp.abs(T_state - T_direct) < _PERIOD_TOL

    def test_orbital_period_from_state_elliptical(self):
        """Period from elliptical orbit state vector matches period from SMA."""
        a = _SMA_500
        e = 0.1
        r_perigee = a * (1.0 - e)
        v_perigee = jnp.sqrt(GM_EARTH * (2.0 / r_perigee - 1.0 / a))
        state = jnp.array([r_perigee, 0.0, 0.0, 0.0, v_perigee, 0.0])
        T_state = orbital_period_from_state(state)
        T_direct = orbital_period(a)
        assert jnp.abs(T_state - T_direct) < _PERIOD_TOL


# ──────────────────────────────────────────────
# Semi-major axis
# ──────────────────────────────────────────────

class TestSemimajorAxis:
    def test_semimajor_axis_from_period_roundtrip(self):
        """Period -> SMA -> Period roundtrip preserves value."""
        T = orbital_period(_SMA_500)
        a = semimajor_axis_from_orbital_period(T)
        assert jnp.abs(a - _SMA_500) < _SMA_TOL

    def test_semimajor_axis_from_mean_motion_radians(self):
        """SMA from mean motion in radians matches expected orbit."""
        n = mean_motion(_SMA_500)
        a = semimajor_axis(n)
        assert jnp.abs(a - _SMA_500) < _SMA_TOL

    def test_semimajor_axis_from_mean_motion_degrees(self):
        """SMA from mean motion in degrees matches expected orbit."""
        n = mean_motion(_SMA_500, use_degrees=True)
        a = semimajor_axis(n, use_degrees=True)
        assert jnp.abs(a - _SMA_500) < _SMA_TOL


# ──────────────────────────────────────────────
# Mean motion
# ──────────────────────────────────────────────

class TestMeanMotion:
    def test_mean_motion_radians(self):
        """Mean motion of 500 km orbit in radians matches reference."""
        n = mean_motion(_SMA_500)
        assert jnp.abs(n - 0.0011067836) < 1e-7

    def test_mean_motion_degrees(self):
        """Mean motion of 500 km orbit in degrees matches reference."""
        n = mean_motion(_SMA_500, use_degrees=True)
        assert jnp.abs(n - 0.0634140299) < 1e-6


# ──────────────────────────────────────────────
# Velocity at apsides
# ──────────────────────────────────────────────

class TestVelocity:
    def test_perigee_velocity(self):
        """Perigee velocity at 500 km, e=0.001 matches reference."""
        vp = perigee_velocity(_SMA_500, 0.001)
        assert jnp.abs(vp - 7620.225) < _VELOCITY_TOL

    def test_apogee_velocity(self):
        """Apogee velocity at 500 km, e=0.001 matches reference."""
        va = apogee_velocity(_SMA_500, 0.001)
        assert jnp.abs(va - 7605.0) < _VELOCITY_TOL

    def test_circular_orbit_perigee_equals_apogee(self):
        """For a circular orbit (e=0), perigee and apogee velocities are equal."""
        vp = perigee_velocity(_SMA_500, 0.0)
        va = apogee_velocity(_SMA_500, 0.0)
        assert jnp.abs(vp - va) < 1e-3


# ──────────────────────────────────────────────
# Distances and altitudes
# ──────────────────────────────────────────────

class TestDistanceAltitude:
    def test_periapsis_distance_circular(self):
        """Periapsis distance of circular orbit equals SMA."""
        rp = periapsis_distance(_SMA_500, 0.0)
        assert jnp.abs(rp - _SMA_500) < _DISTANCE_TOL

    def test_periapsis_distance_eccentric(self):
        """Periapsis distance of eccentric orbit: a*(1-e)."""
        rp = periapsis_distance(500e3, 0.1)
        assert jnp.abs(rp - 450e3) < _DISTANCE_TOL

    def test_apoapsis_distance_circular(self):
        """Apoapsis distance of circular orbit equals SMA."""
        ra = apoapsis_distance(_SMA_500, 0.0)
        assert jnp.abs(ra - _SMA_500) < _DISTANCE_TOL

    def test_apoapsis_distance_eccentric(self):
        """Apoapsis distance of eccentric orbit: a*(1+e)."""
        ra = apoapsis_distance(500e3, 0.1)
        assert jnp.abs(ra - 550e3) < _DISTANCE_TOL

    def test_perigee_altitude_iss(self):
        """Perigee altitude of ISS-like orbit is in expected range."""
        a = R_EARTH + 420e3
        alt = perigee_altitude(a, 0.0005)
        assert 416e3 < float(alt) < 420e3

    def test_apogee_altitude_molniya(self):
        """Apogee altitude of Molniya-type orbit is very high."""
        a = 26554e3
        alt = apogee_altitude(a, 0.7)
        assert float(alt) > 30_000e3

    def test_circular_orbit_perigee_equals_apogee_altitude(self):
        """For circular orbit, perigee and apogee altitudes match."""
        alt_p = perigee_altitude(_SMA_500, 0.0)
        alt_a = apogee_altitude(_SMA_500, 0.0)
        assert jnp.abs(alt_p - alt_a) < _DISTANCE_TOL


# ──────────────────────────────────────────────
# Special orbits
# ──────────────────────────────────────────────

class TestSpecialOrbits:
    def test_sun_synchronous_inclination(self):
        """Sun-sync inclination at 500 km, e=0.001 matches reference."""
        inc = sun_synchronous_inclination(_SMA_500, 0.001, use_degrees=True)
        assert jnp.abs(inc - 97.4017) < _SSI_DEG_TOL

    def test_geo_sma(self):
        """Geostationary SMA is approximately 42164 km."""
        a = geo_sma()
        assert jnp.abs(a - 42164172.0) < _GEO_SMA_TOL


# ──────────────────────────────────────────────
# Anomaly conversions — value tests
# ──────────────────────────────────────────────

class TestAnomalyEccentricMean:
    def test_eccentric_to_mean_zero(self):
        """E=0 -> M=0 for any eccentricity."""
        M = anomaly_eccentric_to_mean(0.0, 0.0)
        assert jnp.abs(M) < _ANOMALY_RAD_TOL

    def test_eccentric_to_mean_pi(self):
        """E=pi -> M=pi for e=0."""
        M = anomaly_eccentric_to_mean(jnp.pi, 0.0)
        assert jnp.abs(M - jnp.pi) < _ANOMALY_RAD_TOL

    def test_eccentric_to_mean_90_deg(self):
        """E=90 deg, e=0.1 matches reference value."""
        M = anomaly_eccentric_to_mean(90.0, 0.1, use_degrees=True)
        assert jnp.abs(M - 84.2704) < _ANOMALY_DEG_TOL

    def test_eccentric_to_mean_90_rad(self):
        """E=pi/2, e=0.1 matches reference value."""
        M = anomaly_eccentric_to_mean(jnp.pi / 2.0, 0.1)
        assert jnp.abs(M - 1.47080) < _ANOMALY_RAD_TOL


class TestAnomalyMeanEccentric:
    def test_mean_to_eccentric_zero(self):
        """M=0 -> E=0."""
        E = anomaly_mean_to_eccentric(0.0, 0.0)
        assert jnp.abs(E) < _ANOMALY_RAD_TOL

    def test_mean_to_eccentric_pi(self):
        """M=pi -> E=pi for e=0."""
        E = anomaly_mean_to_eccentric(jnp.pi, 0.0)
        assert jnp.abs(E - jnp.pi) < _ANOMALY_RAD_TOL

    def test_mean_to_eccentric_90_deg(self):
        """M=84.2704 deg, e=0.1 -> E≈90 deg."""
        E = anomaly_mean_to_eccentric(84.27042, 0.1, use_degrees=True)
        assert jnp.abs(E - 90.0) < _ANOMALY_DEG_TOL

    def test_mean_to_eccentric_90_rad(self):
        """M=1.4708 rad, e=0.1 -> E≈pi/2."""
        E = anomaly_mean_to_eccentric(1.4707963, 0.1)
        assert jnp.abs(E - jnp.pi / 2.0) < _ANOMALY_RAD_TOL


class TestAnomalyTrueEccentric:
    def test_true_to_eccentric_zero(self):
        """nu=0 -> E=0."""
        E = anomaly_true_to_eccentric(0.0, 0.0)
        assert jnp.abs(E) < _ANOMALY_RAD_TOL

    def test_true_to_eccentric_pi(self):
        """nu=pi -> E=pi for e=0 (atan2 may return -pi, which is equivalent)."""
        E = anomaly_true_to_eccentric(jnp.pi, 0.0)
        assert jnp.abs(jnp.abs(E) - jnp.pi) < _ANOMALY_RAD_TOL

    def test_true_to_eccentric_90_deg(self):
        """nu=90 deg, e=0.1 -> E≈84.2608 deg."""
        E = anomaly_true_to_eccentric(90.0, 0.1, use_degrees=True)
        assert jnp.abs(E - 84.2608) < _ANOMALY_DEG_TOL

    def test_true_to_eccentric_90_rad(self):
        """nu=pi/2, e=0.1 -> E≈1.4706 rad."""
        E = anomaly_true_to_eccentric(jnp.pi / 2.0, 0.1)
        assert jnp.abs(E - 1.47063) < _ANOMALY_RAD_TOL


class TestAnomalyEccentricTrue:
    def test_eccentric_to_true_zero(self):
        """E=0 -> nu=0."""
        nu = anomaly_eccentric_to_true(0.0, 0.0)
        assert jnp.abs(nu) < _ANOMALY_RAD_TOL

    def test_eccentric_to_true_pi(self):
        """E=pi -> nu=pi for e=0 (atan2 may return -pi, which is equivalent)."""
        nu = anomaly_eccentric_to_true(jnp.pi, 0.0)
        assert jnp.abs(jnp.abs(nu) - jnp.pi) < _ANOMALY_RAD_TOL

    def test_eccentric_to_true_90_deg(self):
        """E=90 deg, e=0.1 -> nu≈95.739 deg."""
        nu = anomaly_eccentric_to_true(90.0, 0.1, use_degrees=True)
        assert jnp.abs(nu - 95.739) < _ANOMALY_DEG_TOL

    def test_eccentric_to_true_90_rad(self):
        """E=pi/2, e=0.1 -> nu≈1.6710 rad."""
        nu = anomaly_eccentric_to_true(jnp.pi / 2.0, 0.1)
        assert jnp.abs(nu - 1.67096) < _ANOMALY_RAD_TOL


class TestAnomalyTrueMean:
    def test_true_to_mean_zero(self):
        """nu=0, e=0 -> M=0."""
        M = anomaly_true_to_mean(0.0, 0.0)
        assert jnp.abs(M) < _ANOMALY_RAD_TOL

    def test_true_to_mean_90_deg(self):
        """nu=90 deg, e=0.1 -> M≈78.56 deg."""
        M = anomaly_true_to_mean(90.0, 0.1, use_degrees=True)
        assert jnp.abs(M - 78.56) < _ANOMALY_DEG_TOL

    def test_true_to_mean_90_rad(self):
        """nu=pi/2, e=0.1 -> M≈1.3711 rad."""
        M = anomaly_true_to_mean(jnp.pi / 2.0, 0.1)
        assert jnp.abs(M - 1.3711) < _ANOMALY_RAD_TOL


class TestAnomalyMeanTrue:
    def test_mean_to_true_zero(self):
        """M=0, e=0 -> nu=0."""
        nu = anomaly_mean_to_true(0.0, 0.0)
        assert jnp.abs(nu) < _ANOMALY_RAD_TOL

    def test_mean_to_true_90_deg(self):
        """M=90 deg, e=0.1 -> nu≈101.38 deg."""
        nu = anomaly_mean_to_true(90.0, 0.1, use_degrees=True)
        assert jnp.abs(nu - 101.38) < _ANOMALY_DEG_TOL

    def test_mean_to_true_90_rad(self):
        """M=pi/2, e=0.1 -> nu≈1.7695 rad."""
        nu = anomaly_mean_to_true(jnp.pi / 2.0, 0.1)
        assert jnp.abs(nu - 1.7695) < _ANOMALY_RAD_TOL


# ──────────────────────────────────────────────
# Anomaly roundtrip bijectivity
# ──────────────────────────────────────────────

class TestAnomalyRoundtrip:
    @pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.5, 0.7])
    @pytest.mark.parametrize("theta", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0])
    def test_eccentric_mean_roundtrip(self, e, theta):
        """E -> M -> E roundtrip preserves value."""
        M = anomaly_eccentric_to_mean(theta, e, use_degrees=True)
        E_back = anomaly_mean_to_eccentric(M, e, use_degrees=True)
        assert jnp.abs(E_back - theta) < _ROUNDTRIP_DEG_TOL

    @pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.5, 0.7])
    @pytest.mark.parametrize("theta", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0])
    def test_mean_eccentric_roundtrip(self, e, theta):
        """M -> E -> M roundtrip preserves value."""
        E = anomaly_mean_to_eccentric(theta, e, use_degrees=True)
        M_back = anomaly_eccentric_to_mean(E, e, use_degrees=True)
        assert jnp.abs(M_back - theta) < _ROUNDTRIP_DEG_TOL

    @pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.5, 0.7])
    @pytest.mark.parametrize("theta", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0])
    def test_true_eccentric_roundtrip(self, e, theta):
        """nu -> E -> nu roundtrip preserves value."""
        E = anomaly_true_to_eccentric(theta, e, use_degrees=True)
        nu_back = anomaly_eccentric_to_true(E, e, use_degrees=True)
        assert jnp.abs(nu_back - theta) < _ROUNDTRIP_DEG_TOL

    @pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.5, 0.7])
    @pytest.mark.parametrize("theta", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 179.0])
    def test_true_mean_roundtrip(self, e, theta):
        """nu -> M -> nu roundtrip preserves value."""
        M = anomaly_true_to_mean(theta, e, use_degrees=True)
        nu_back = anomaly_mean_to_true(M, e, use_degrees=True)
        assert jnp.abs(nu_back - theta) < _ROUNDTRIP_DEG_TOL


# ──────────────────────────────────────────────
# JAX compatibility
# ──────────────────────────────────────────────

class TestJAXCompatibility:
    def test_jit_orbital_period(self):
        """orbital_period is JIT-compilable."""
        T_eager = orbital_period(_SMA_500)
        T_jit = jax.jit(orbital_period)(_SMA_500)
        assert jnp.abs(T_eager - T_jit) < 1e-3

    def test_jit_anomaly_mean_to_eccentric(self):
        """anomaly_mean_to_eccentric is JIT-compilable (fori_loop)."""
        E_eager = anomaly_mean_to_eccentric(90.0, 0.1, use_degrees=True)
        E_jit = jax.jit(anomaly_mean_to_eccentric, static_argnums=2)(
            90.0, 0.1, True
        )
        assert jnp.abs(E_eager - E_jit) < 1e-3

    def test_vmap_orbital_period(self):
        """orbital_period works with vmap over a batch of SMA values."""
        smas = jnp.array([R_EARTH + 400e3, R_EARTH + 500e3, R_EARTH + 600e3])
        periods = jax.vmap(orbital_period)(smas)
        assert periods.shape == (3,)
        # Verify monotonically increasing (higher orbit = longer period)
        assert float(periods[0]) < float(periods[1]) < float(periods[2])

    def test_vmap_anomaly_mean_to_eccentric(self):
        """anomaly_mean_to_eccentric works with vmap over anomaly values."""
        Ms = jnp.array([30.0, 60.0, 90.0])
        Es = jax.vmap(anomaly_mean_to_eccentric, in_axes=(0, None, None))(
            Ms, 0.1, True
        )
        assert Es.shape == (3,)

    def test_grad_orbital_period(self):
        """orbital_period supports gradient computation."""
        grad_fn = jax.grad(lambda a: orbital_period(a))
        g = grad_fn(jnp.float32(_SMA_500))
        assert g.shape == ()
        assert float(g) > 0.0  # Period increases with SMA

    def test_grad_anomaly_mean_to_eccentric(self):
        """anomaly_mean_to_eccentric supports gradient (through fori_loop)."""
        def scalar_fn(M):
            return anomaly_mean_to_eccentric(M, 0.1, False)

        g = jax.grad(scalar_fn)(jnp.float32(jnp.pi / 2.0))
        assert g.shape == ()
        assert jnp.isfinite(g)

    def test_grad_sun_synchronous_inclination(self):
        """sun_synchronous_inclination supports gradient w.r.t. SMA."""
        def scalar_fn(a):
            return sun_synchronous_inclination(a, 0.001, use_degrees=False)

        g = jax.grad(scalar_fn)(jnp.float32(_SMA_500))
        assert g.shape == ()
        assert jnp.isfinite(g)


# ──────────────────────────────────────────────
# Mean-Osculating element conversions
# ──────────────────────────────────────────────

# Tolerances for mean-osculating roundtrips (first-order theory, J2^2 errors)
_MO_SMA_TOL = 100.0       # metres
_MO_ECC_TOL = 1e-4         # dimensionless
_MO_ANGLE_RAD_TOL = 0.01   # radians (~0.6 degrees)
_MO_ANGLE_DEG_TOL = 0.6    # degrees


class TestMeanOsculatingRoundtrip:
    def test_mean_to_osc_to_mean_radians(self):
        """mean -> osc -> mean roundtrip preserves elements (radians)."""
        mean = jnp.array([
            R_EARTH + 500e3,
            0.01,
            jnp.deg2rad(45.0),
            jnp.deg2rad(30.0),
            jnp.deg2rad(60.0),
            jnp.deg2rad(90.0),
        ])
        osc = state_koe_mean_to_osc(mean)
        recovered = state_koe_osc_to_mean(osc)

        assert jnp.abs(mean[0] - recovered[0]) < _MO_SMA_TOL
        assert jnp.abs(mean[1] - recovered[1]) < _MO_ECC_TOL
        for idx in range(2, 6):
            assert jnp.abs(mean[idx] - recovered[idx]) < _MO_ANGLE_RAD_TOL

    def test_mean_to_osc_to_mean_degrees(self):
        """mean -> osc -> mean roundtrip preserves elements (degrees)."""
        mean = jnp.array([R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
        osc = state_koe_mean_to_osc(mean, use_degrees=True)
        recovered = state_koe_osc_to_mean(osc, use_degrees=True)

        assert jnp.abs(mean[0] - recovered[0]) < _MO_SMA_TOL
        assert jnp.abs(mean[1] - recovered[1]) < _MO_ECC_TOL
        for idx in range(2, 6):
            assert jnp.abs(mean[idx] - recovered[idx]) < _MO_ANGLE_DEG_TOL

    def test_osc_to_mean_to_osc(self):
        """osc -> mean -> osc roundtrip preserves elements."""
        osc = jnp.array([
            R_EARTH + 600e3,
            0.02,
            jnp.deg2rad(60.0),
            jnp.deg2rad(45.0),
            jnp.deg2rad(120.0),
            jnp.deg2rad(180.0),
        ])
        mean = state_koe_osc_to_mean(osc)
        recovered = state_koe_mean_to_osc(mean)

        assert jnp.abs(osc[0] - recovered[0]) < _MO_SMA_TOL
        assert jnp.abs(osc[1] - recovered[1]) < _MO_ECC_TOL
        for idx in range(2, 6):
            assert jnp.abs(osc[idx] - recovered[idx]) < _MO_ANGLE_RAD_TOL


class TestMeanOsculatingOrbitTypes:
    def test_near_circular(self):
        """Near-circular orbit (e=0.0001) roundtrip succeeds."""
        mean = jnp.array([
            R_EARTH + 400e3, 0.0001, jnp.deg2rad(28.5), 0.0, 0.0, 0.0,
        ])
        osc = state_koe_mean_to_osc(mean)
        recovered = state_koe_osc_to_mean(osc)
        assert jnp.abs(mean[0] - recovered[0]) < _MO_SMA_TOL

    def test_sun_synchronous(self):
        """Sun-synchronous orbit (i=98 deg) roundtrip succeeds."""
        mean = jnp.array([
            R_EARTH + 700e3,
            0.001,
            jnp.deg2rad(98.0),
            jnp.deg2rad(45.0),
            jnp.deg2rad(90.0),
            jnp.deg2rad(270.0),
        ])
        osc = state_koe_mean_to_osc(mean)
        recovered = state_koe_osc_to_mean(osc)
        assert jnp.abs(mean[0] - recovered[0]) < _MO_SMA_TOL
        assert jnp.abs(mean[1] - recovered[1]) < 1e-3

    def test_moderate_eccentricity(self):
        """Moderate eccentricity (e=0.1) roundtrip succeeds."""
        mean = jnp.array([
            R_EARTH + 500e3,
            0.1,
            jnp.deg2rad(45.0),
            jnp.deg2rad(30.0),
            jnp.deg2rad(60.0),
            jnp.deg2rad(90.0),
        ])
        osc = state_koe_mean_to_osc(mean)
        recovered = state_koe_osc_to_mean(osc)
        assert jnp.abs(mean[0] - recovered[0]) < _MO_SMA_TOL
        assert jnp.abs(mean[1] - recovered[1]) < _MO_ECC_TOL

    def test_geo(self):
        """GEO orbit (42164 km) roundtrip — J2 effects smaller at GEO."""
        mean = jnp.array([
            42164e3, 0.0001, jnp.deg2rad(0.1), jnp.deg2rad(45.0), 0.0, 0.0,
        ])
        osc = state_koe_mean_to_osc(mean)
        recovered = state_koe_osc_to_mean(osc)
        assert jnp.abs(mean[0] - recovered[0]) < 10.0

    @pytest.mark.parametrize("m_deg", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
    def test_various_mean_anomalies(self, m_deg):
        """Roundtrip across different mean anomaly values."""
        mean = jnp.array([
            R_EARTH + 500e3,
            0.01,
            jnp.deg2rad(45.0),
            jnp.deg2rad(30.0),
            jnp.deg2rad(60.0),
            jnp.deg2rad(m_deg),
        ])
        osc = state_koe_mean_to_osc(mean)
        recovered = state_koe_osc_to_mean(osc)
        assert jnp.abs(mean[0] - recovered[0]) < _MO_SMA_TOL


class TestMeanOsculatingBehavior:
    def test_osc_differs_from_mean(self):
        """Osculating elements differ from mean (J2 perturbation effect)."""
        mean = jnp.array([
            R_EARTH + 500e3,
            0.01,
            jnp.deg2rad(45.0),
            jnp.deg2rad(30.0),
            jnp.deg2rad(60.0),
            jnp.deg2rad(90.0),
        ])
        osc = state_koe_mean_to_osc(mean)
        assert jnp.abs(osc[0] - mean[0]) > 1.0  # SMA differs by > 1 metre

    def test_degrees_consistency(self):
        """Degrees output stays in degree range (not radian range)."""
        mean = jnp.array([R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
        osc = state_koe_mean_to_osc(mean, use_degrees=True)
        # Inclination should be in degree range
        assert float(osc[2]) > 7.0
        assert float(osc[2]) < 180.0


class TestMeanOsculatingJAXCompat:
    def test_jit_mean_to_osc(self):
        """state_koe_mean_to_osc is JIT-compilable."""
        mean = jnp.array([
            R_EARTH + 500e3, 0.01, jnp.deg2rad(45.0),
            jnp.deg2rad(30.0), jnp.deg2rad(60.0), jnp.deg2rad(90.0),
        ])
        osc_eager = state_koe_mean_to_osc(mean)
        osc_jit = jax.jit(state_koe_mean_to_osc)(mean)
        assert jnp.allclose(osc_eager, osc_jit, atol=1e-3)

    def test_jit_osc_to_mean(self):
        """state_koe_osc_to_mean is JIT-compilable."""
        osc = jnp.array([
            R_EARTH + 500e3, 0.01, jnp.deg2rad(45.0),
            jnp.deg2rad(30.0), jnp.deg2rad(60.0), jnp.deg2rad(90.0),
        ])
        mean_eager = state_koe_osc_to_mean(osc)
        mean_jit = jax.jit(state_koe_osc_to_mean)(osc)
        assert jnp.allclose(mean_eager, mean_jit, atol=1e-3)

    def test_vmap_mean_to_osc(self):
        """state_koe_mean_to_osc works with vmap over a batch."""
        batch = jnp.array([
            [R_EARTH + 400e3, 0.01, jnp.deg2rad(45.0), 0.0, 0.0, 0.0],
            [R_EARTH + 500e3, 0.01, jnp.deg2rad(45.0), 0.0, 0.0, 0.0],
            [R_EARTH + 600e3, 0.01, jnp.deg2rad(45.0), 0.0, 0.0, 0.0],
        ])
        result = jax.vmap(state_koe_mean_to_osc)(batch)
        assert result.shape == (3, 6)

    def test_grad_mean_to_osc(self):
        """state_koe_mean_to_osc supports gradient computation."""
        mean = jnp.array([
            R_EARTH + 500e3, 0.01, jnp.deg2rad(45.0),
            jnp.deg2rad(30.0), jnp.deg2rad(60.0), jnp.deg2rad(90.0),
        ])

        def scalar_fn(oe):
            return state_koe_mean_to_osc(oe)[0]  # SMA component

        g = jax.grad(scalar_fn)(mean)
        assert g.shape == (6,)
        assert jnp.all(jnp.isfinite(g))
