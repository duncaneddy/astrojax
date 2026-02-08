"""Tests for the astrojax.coordinates module.

Covers geocentric, geodetic, and Keplerian coordinate transformations
with float32-appropriate tolerances, round-trip validation, cardinal-point
checks, use_degrees flag, and JAX compatibility (jit, vmap).
"""

import jax
import jax.numpy as jnp

from astrojax.constants import DEG2RAD, GM_EARTH, R_EARTH, WGS84_a, WGS84_f
from astrojax.coordinates import (
    position_ecef_to_geocentric,
    position_ecef_to_geodetic,
    position_geocentric_to_ecef,
    position_geodetic_to_ecef,
    state_eci_to_koe,
    state_koe_to_eci,
)

# ──────────────────────────────────────────────
# Tolerance constants (float32-appropriate)
# ──────────────────────────────────────────────

_POS_TOL = 1.0  # metres
_GEOD_ROUNDTRIP_TOL = 1.0  # metres
_GEOC_ROUNDTRIP_TOL = 1.0  # metres (float32 on ~6.5e6 m magnitudes)
_ANGLE_TOL = 1e-4  # radians (~0.006 degrees, float32 roundtrip precision)
_SMA_TOL = 10.0  # metres
_ECC_TOL = 5e-4  # dimensionless (float32 vis-viva roundoff)
_ANG_DEG_TOL = 5e-3  # degrees (float32 roundtrip precision)
_VEL_TOL = 1e-3  # m/s


# ──────────────────────────────────────────────
# Geocentric transformations
# ──────────────────────────────────────────────


class TestGeocentricToECEF:
    def test_origin_equator(self):
        """lon=0, lat=0, alt=0 → [WGS84_a, 0, 0]."""
        x_geoc = jnp.array([0.0, 0.0, 0.0])
        x_ecef = position_geocentric_to_ecef(x_geoc)

        assert jnp.abs(x_ecef[0] - WGS84_a) < _POS_TOL
        assert jnp.abs(x_ecef[1]) < _POS_TOL
        assert jnp.abs(x_ecef[2]) < _POS_TOL

    def test_90deg_lon(self):
        """lon=90°, lat=0, alt=0 → [0, WGS84_a, 0]."""
        x_geoc = jnp.array([90.0, 0.0, 0.0])
        x_ecef = position_geocentric_to_ecef(x_geoc, use_degrees=True)

        assert jnp.abs(x_ecef[0]) < _POS_TOL
        assert jnp.abs(x_ecef[1] - WGS84_a) < _POS_TOL
        assert jnp.abs(x_ecef[2]) < _POS_TOL

    def test_north_pole(self):
        """lon=0, lat=90°, alt=0 → [0, 0, WGS84_a]."""
        x_geoc = jnp.array([0.0, 90.0, 0.0])
        x_ecef = position_geocentric_to_ecef(x_geoc, use_degrees=True)

        assert jnp.abs(x_ecef[0]) < _POS_TOL
        assert jnp.abs(x_ecef[1]) < _POS_TOL
        assert jnp.abs(x_ecef[2] - WGS84_a) < _POS_TOL

    def test_with_altitude(self):
        """Altitude shifts the radial distance."""
        alt = 500e3
        x_geoc = jnp.array([0.0, 0.0, alt])
        x_ecef = position_geocentric_to_ecef(x_geoc)

        assert jnp.abs(x_ecef[0] - (WGS84_a + alt)) < _POS_TOL

    def test_use_degrees_consistent(self):
        """Degrees and radians inputs give the same ECEF output."""
        lon_deg, lat_deg = 45.0, 30.0
        lon_rad = lon_deg * DEG2RAD
        lat_rad = lat_deg * DEG2RAD

        ecef_deg = position_geocentric_to_ecef(
            jnp.array([lon_deg, lat_deg, 0.0]), use_degrees=True
        )
        ecef_rad = position_geocentric_to_ecef(
            jnp.array([lon_rad, lat_rad, 0.0]), use_degrees=False
        )
        assert jnp.allclose(ecef_deg, ecef_rad, atol=_POS_TOL)


class TestECEFToGeocentric:
    def test_origin_equator(self):
        """[WGS84_a, 0, 0] → lon=0, lat=0, alt=0."""
        x_ecef = jnp.array([WGS84_a, 0.0, 0.0])
        geoc = position_ecef_to_geocentric(x_ecef)

        assert jnp.abs(geoc[0]) < _ANGLE_TOL  # lon
        assert jnp.abs(geoc[1]) < _ANGLE_TOL  # lat
        assert jnp.abs(geoc[2]) < _POS_TOL  # alt

    def test_north_pole(self):
        """[0, 0, WGS84_a] → lon=0, lat=90°, alt=0."""
        x_ecef = jnp.array([0.0, 0.0, WGS84_a])
        geoc = position_ecef_to_geocentric(x_ecef, use_degrees=True)

        assert jnp.abs(geoc[0]) < _ANG_DEG_TOL  # lon
        assert jnp.abs(geoc[1] - 90.0) < _ANG_DEG_TOL  # lat
        assert jnp.abs(geoc[2]) < _POS_TOL  # alt

    def test_use_degrees_output(self):
        """use_degrees=True returns degrees."""
        x_ecef = jnp.array([0.0, WGS84_a, 0.0])
        geoc = position_ecef_to_geocentric(x_ecef, use_degrees=True)

        assert jnp.abs(geoc[0] - 90.0) < _ANG_DEG_TOL


class TestGeocentricRoundTrip:
    def test_roundtrip_cardinal_points(self):
        """Forward → inverse ≈ identity for cardinal points."""
        for geoc_deg in [
            [0.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
            [-45.0, 30.0, 100e3],
        ]:
            x_geoc = jnp.array(geoc_deg)
            ecef = position_geocentric_to_ecef(x_geoc, use_degrees=True)
            back = position_ecef_to_geocentric(ecef, use_degrees=True)

            assert jnp.allclose(back[:2], x_geoc[:2], atol=_ANG_DEG_TOL), (
                f"Angle roundtrip failed for {geoc_deg}"
            )
            assert jnp.abs(back[2] - x_geoc[2]) < _GEOC_ROUNDTRIP_TOL, (
                f"Altitude roundtrip failed for {geoc_deg}"
            )

    def test_roundtrip_pole(self):
        """Pole roundtrip: latitude and altitude recover, longitude is undefined."""
        x_geoc = jnp.array([0.0, 90.0, 0.0])
        ecef = position_geocentric_to_ecef(x_geoc, use_degrees=True)
        back = position_ecef_to_geocentric(ecef, use_degrees=True)

        assert jnp.abs(back[1] - 90.0) < _ANG_DEG_TOL  # lat
        assert jnp.abs(back[2]) < _GEOC_ROUNDTRIP_TOL  # alt

    def test_roundtrip_random_point(self):
        """Round-trip for a non-trivial point (matches Rust test)."""
        x_geoc = jnp.array([77.875, 20.9752, 0.0])
        ecef = position_geocentric_to_ecef(x_geoc, use_degrees=True)
        back = position_ecef_to_geocentric(ecef, use_degrees=True)

        assert jnp.allclose(back[:2], x_geoc[:2], atol=_ANG_DEG_TOL)
        assert jnp.abs(back[2] - x_geoc[2]) < _GEOC_ROUNDTRIP_TOL


# ──────────────────────────────────────────────
# Geodetic transformations
# ──────────────────────────────────────────────


class TestGeodeticToECEF:
    def test_origin_equator(self):
        """lon=0, lat=0, alt=0 → [WGS84_a, 0, 0]."""
        x_geod = jnp.array([0.0, 0.0, 0.0])
        x_ecef = position_geodetic_to_ecef(x_geod)

        assert jnp.abs(x_ecef[0] - WGS84_a) < _POS_TOL
        assert jnp.abs(x_ecef[1]) < _POS_TOL
        assert jnp.abs(x_ecef[2]) < _POS_TOL

    def test_90deg_lon(self):
        """lon=90°, lat=0, alt=0 → [0, WGS84_a, 0]."""
        x_geod = jnp.array([90.0, 0.0, 0.0])
        x_ecef = position_geodetic_to_ecef(x_geod, use_degrees=True)

        assert jnp.abs(x_ecef[0]) < _POS_TOL
        assert jnp.abs(x_ecef[1] - WGS84_a) < _POS_TOL
        assert jnp.abs(x_ecef[2]) < _POS_TOL

    def test_north_pole(self):
        """lon=0, lat=90°, alt=0 → [0, 0, WGS84_a*(1-f)] (semi-minor axis)."""
        x_geod = jnp.array([0.0, 90.0, 0.0])
        x_ecef = position_geodetic_to_ecef(x_geod, use_degrees=True)
        b = WGS84_a * (1.0 - WGS84_f)  # semi-minor axis

        assert jnp.abs(x_ecef[0]) < _POS_TOL
        assert jnp.abs(x_ecef[1]) < _POS_TOL
        assert jnp.abs(x_ecef[2] - b) < _POS_TOL

    def test_use_degrees_consistent(self):
        """Degrees and radians inputs give the same ECEF output."""
        lon_deg, lat_deg = 45.0, 30.0
        lon_rad = lon_deg * DEG2RAD
        lat_rad = lat_deg * DEG2RAD

        ecef_deg = position_geodetic_to_ecef(
            jnp.array([lon_deg, lat_deg, 0.0]), use_degrees=True
        )
        ecef_rad = position_geodetic_to_ecef(
            jnp.array([lon_rad, lat_rad, 0.0]), use_degrees=False
        )
        assert jnp.allclose(ecef_deg, ecef_rad, atol=_POS_TOL)


class TestECEFToGeodetic:
    def test_origin_equator(self):
        """[WGS84_a, 0, 0] → lon=0, lat=0, alt=0."""
        x_ecef = jnp.array([WGS84_a, 0.0, 0.0])
        geod = position_ecef_to_geodetic(x_ecef)

        assert jnp.abs(geod[0]) < _ANGLE_TOL  # lon
        assert jnp.abs(geod[1]) < _ANGLE_TOL  # lat
        assert jnp.abs(geod[2]) < _POS_TOL  # alt

    def test_north_pole(self):
        """[0, 0, b] → lon=0, lat=90°, alt≈0."""
        b = WGS84_a * (1.0 - WGS84_f)
        x_ecef = jnp.array([0.0, 0.0, b])
        geod = position_ecef_to_geodetic(x_ecef, use_degrees=True)

        assert jnp.abs(geod[0]) < _ANG_DEG_TOL  # lon
        assert jnp.abs(geod[1] - 90.0) < _ANG_DEG_TOL  # lat
        assert jnp.abs(geod[2]) < _GEOD_ROUNDTRIP_TOL  # alt


class TestGeodeticRoundTrip:
    def test_roundtrip_cardinal_points(self):
        """Forward → inverse ≈ identity for cardinal points."""
        for geod_deg in [
            [0.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
            [-45.0, 30.0, 100e3],
        ]:
            x_geod = jnp.array(geod_deg)
            ecef = position_geodetic_to_ecef(x_geod, use_degrees=True)
            back = position_ecef_to_geodetic(ecef, use_degrees=True)

            assert jnp.allclose(back[:2], x_geod[:2], atol=_ANG_DEG_TOL), (
                f"Angle roundtrip failed for {geod_deg}"
            )
            assert jnp.abs(back[2] - x_geod[2]) < _GEOD_ROUNDTRIP_TOL, (
                f"Altitude roundtrip failed for {geod_deg}"
            )

    def test_roundtrip_pole(self):
        """Pole roundtrip: latitude and altitude recover, longitude is undefined."""
        x_geod = jnp.array([0.0, 90.0, 0.0])
        ecef = position_geodetic_to_ecef(x_geod, use_degrees=True)
        back = position_ecef_to_geodetic(ecef, use_degrees=True)

        assert jnp.abs(back[1] - 90.0) < _ANG_DEG_TOL  # lat
        assert jnp.abs(back[2]) < _GEOD_ROUNDTRIP_TOL  # alt

    def test_roundtrip_random_point(self):
        """Round-trip for a non-trivial point (matches Rust test)."""
        x_geod = jnp.array([77.875, 20.9752, 0.0])
        ecef = position_geodetic_to_ecef(x_geod, use_degrees=True)
        back = position_ecef_to_geodetic(ecef, use_degrees=True)

        assert jnp.allclose(back[:2], x_geod[:2], atol=_ANG_DEG_TOL)
        assert jnp.abs(back[2] - x_geod[2]) < _GEOD_ROUNDTRIP_TOL


# ──────────────────────────────────────────────
# Keplerian orbital element conversions
# ──────────────────────────────────────────────


def _circular_equatorial_eci(sma=R_EARTH + 500e3):
    """Circular equatorial LEO state: r along x, v along y."""
    v_circ = jnp.sqrt(GM_EARTH / sma)
    return jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])


class TestKOEToECI:
    def test_circular_equatorial(self):
        """Circular equatorial orbit: r=[a,0,0], v=[0,v_circ,0]."""
        sma = R_EARTH + 500e3
        oe = jnp.array([sma, 0.0, 0.0, 0.0, 0.0, 0.0])
        state = state_koe_to_eci(oe)

        v_circ = float(jnp.sqrt(GM_EARTH / sma))
        assert jnp.abs(state[0] - sma) < _POS_TOL
        assert jnp.abs(state[1]) < _POS_TOL
        assert jnp.abs(state[2]) < _POS_TOL
        assert jnp.abs(state[3]) < _VEL_TOL
        assert jnp.abs(state[4] - v_circ) < _VEL_TOL
        assert jnp.abs(state[5]) < _VEL_TOL

    def test_polar_orbit(self):
        """i=90° orbit: r along x, v along z."""
        sma = R_EARTH + 500e3
        oe = jnp.array([sma, 0.0, 90.0, 0.0, 0.0, 0.0])
        state = state_koe_to_eci(oe, use_degrees=True)

        v_circ = float(jnp.sqrt(GM_EARTH / sma))
        assert jnp.abs(state[0] - sma) < _POS_TOL
        assert jnp.abs(state[1]) < _POS_TOL
        assert jnp.abs(state[2]) < _POS_TOL
        assert jnp.abs(state[3]) < _VEL_TOL
        assert jnp.abs(state[4]) < _VEL_TOL
        assert jnp.abs(state[5] - v_circ) < _VEL_TOL

    def test_use_degrees_consistent(self):
        """Degrees and radians inputs yield same ECI state."""
        sma = R_EARTH + 500e3
        oe_deg = jnp.array([sma, 0.001, 98.0, 15.0, 30.0, 45.0])
        oe_rad = jnp.array([
            sma, 0.001,
            98.0 * DEG2RAD, 15.0 * DEG2RAD,
            30.0 * DEG2RAD, 45.0 * DEG2RAD,
        ])

        state_deg = state_koe_to_eci(oe_deg, use_degrees=True)
        state_rad = state_koe_to_eci(oe_rad, use_degrees=False)

        assert jnp.allclose(state_deg[:3], state_rad[:3], atol=_POS_TOL)
        assert jnp.allclose(state_deg[3:], state_rad[3:], atol=_VEL_TOL)


class TestECIToKOE:
    def test_circular_equatorial(self):
        """Circular equatorial orbit recovers e=0, i=0."""
        state = _circular_equatorial_eci()
        oe = state_eci_to_koe(state)

        assert jnp.abs(oe[0] - (R_EARTH + 500e3)) < _SMA_TOL  # a
        assert jnp.abs(oe[1]) < _ECC_TOL  # e
        assert jnp.abs(oe[2]) < _ANGLE_TOL  # i

    def test_polar_orbit(self):
        """Polar orbit: v along z gives i ≈ 90°."""
        sma = R_EARTH + 500e3
        v_circ = jnp.sqrt(GM_EARTH / sma)
        state = jnp.array([sma, 0.0, 0.0, 0.0, 0.0, v_circ])
        oe = state_eci_to_koe(state, use_degrees=True)

        assert jnp.abs(oe[0] - sma) < _SMA_TOL
        assert jnp.abs(oe[1]) < _ECC_TOL
        assert jnp.abs(oe[2] - 90.0) < _ANG_DEG_TOL


class TestKeplerianRoundTrip:
    """Round-trip: KOE → ECI → KOE ≈ identity across orbit types."""

    def test_leo_near_circular(self):
        """LEO near-circular sun-synchronous-like orbit."""
        oe = jnp.array([7000e3, 0.001, 98.0, 15.0, 30.0, 45.0])
        cart = state_koe_to_eci(oe, use_degrees=True)
        oe_back = state_eci_to_koe(cart, use_degrees=True)

        assert jnp.abs(oe_back[0] - oe[0]) < _SMA_TOL
        assert jnp.abs(oe_back[1] - oe[1]) < _ECC_TOL
        assert jnp.abs(oe_back[2] - oe[2]) < _ANG_DEG_TOL
        assert jnp.abs(oe_back[3] - oe[3]) < _ANG_DEG_TOL
        assert jnp.abs(oe_back[4] - oe[4]) < _ANG_DEG_TOL
        assert jnp.abs(oe_back[5] - oe[5]) < _ANG_DEG_TOL

    def test_molniya(self):
        """Highly eccentric Molniya orbit (e=0.74)."""
        oe = jnp.array([26560e3, 0.74, 63.4, 250.0, 90.0, 180.0])
        cart = state_koe_to_eci(oe, use_degrees=True)
        oe_back = state_eci_to_koe(cart, use_degrees=True)

        assert jnp.abs(oe_back[0] - oe[0]) < _SMA_TOL
        assert jnp.abs(oe_back[1] - oe[1]) < _ECC_TOL
        assert jnp.abs(oe_back[2] - oe[2]) < _ANG_DEG_TOL
        assert jnp.abs(oe_back[3] - oe[3]) < _ANG_DEG_TOL
        assert jnp.abs(oe_back[4] - oe[4]) < _ANG_DEG_TOL
        assert jnp.abs(oe_back[5] - oe[5]) < _ANG_DEG_TOL

    def test_geo(self):
        """Geostationary-like circular equatorial orbit."""
        oe = jnp.array([42164e3, 0.0, 0.0, 0.0, 0.0, 0.0])
        cart = state_koe_to_eci(oe, use_degrees=True)
        oe_back = state_eci_to_koe(cart, use_degrees=True)

        assert jnp.abs(oe_back[0] - oe[0]) < _SMA_TOL
        assert jnp.abs(oe_back[1] - oe[1]) < _ECC_TOL

    def test_eccentric_inclined(self):
        """Eccentric inclined orbit."""
        oe = jnp.array([7000e3, 0.5, 45.0, 120.0, 270.0, 300.0])
        cart = state_koe_to_eci(oe, use_degrees=True)
        oe_back = state_eci_to_koe(cart, use_degrees=True)

        assert jnp.abs(oe_back[0] - oe[0]) < _SMA_TOL
        assert jnp.abs(oe_back[1] - oe[1]) < _ECC_TOL
        assert jnp.abs(oe_back[2] - oe[2]) < _ANG_DEG_TOL
        assert jnp.abs(oe_back[3] - oe[3]) < _ANG_DEG_TOL
        assert jnp.abs(oe_back[4] - oe[4]) < _ANG_DEG_TOL
        assert jnp.abs(oe_back[5] - oe[5]) < _ANG_DEG_TOL

    def test_roundtrip_radians(self):
        """Round-trip in radians mode."""
        oe = jnp.array([
            7000e3, 0.001,
            98.0 * DEG2RAD, 15.0 * DEG2RAD,
            30.0 * DEG2RAD, 45.0 * DEG2RAD,
        ])
        cart = state_koe_to_eci(oe)
        oe_back = state_eci_to_koe(cart)

        assert jnp.abs(oe_back[0] - oe[0]) < _SMA_TOL
        assert jnp.abs(oe_back[1] - oe[1]) < _ECC_TOL
        assert jnp.abs(oe_back[2] - oe[2]) < _ANGLE_TOL
        assert jnp.abs(oe_back[3] - oe[3]) < _ANGLE_TOL
        assert jnp.abs(oe_back[4] - oe[4]) < _ANGLE_TOL
        assert jnp.abs(oe_back[5] - oe[5]) < _ANGLE_TOL


# ──────────────────────────────────────────────
# JAX compatibility
# ──────────────────────────────────────────────


class TestJAXCompatibility:
    def test_jit_geocentric_to_ecef(self):
        """position_geocentric_to_ecef is JIT-compilable."""
        x = jnp.array([0.5, 0.3, 100e3])
        eager = position_geocentric_to_ecef(x)
        jitted = jax.jit(position_geocentric_to_ecef)(x)
        assert jnp.allclose(eager, jitted, atol=1e-5)

    def test_jit_ecef_to_geocentric(self):
        """position_ecef_to_geocentric is JIT-compilable."""
        x = jnp.array([WGS84_a, 1e6, 0.5e6])
        eager = position_ecef_to_geocentric(x)
        jitted = jax.jit(position_ecef_to_geocentric)(x)
        assert jnp.allclose(eager, jitted, atol=1e-5)

    def test_jit_geodetic_to_ecef(self):
        """position_geodetic_to_ecef is JIT-compilable."""
        x = jnp.array([0.5, 0.3, 100e3])
        eager = position_geodetic_to_ecef(x)
        jitted = jax.jit(position_geodetic_to_ecef)(x)
        assert jnp.allclose(eager, jitted, atol=1e-5)

    def test_jit_ecef_to_geodetic(self):
        """position_ecef_to_geodetic is JIT-compilable."""
        x = jnp.array([WGS84_a, 1e6, 0.5e6])
        eager = position_ecef_to_geodetic(x)
        jitted = jax.jit(position_ecef_to_geodetic)(x)
        assert jnp.allclose(eager, jitted, atol=1e-5)

    def test_jit_koe_to_eci(self):
        """state_koe_to_eci is JIT-compilable."""
        oe = jnp.array([7000e3, 0.001, 0.5, 0.3, 0.2, 0.8])
        eager = state_koe_to_eci(oe)
        jitted = jax.jit(state_koe_to_eci)(oe)
        assert jnp.allclose(eager, jitted, atol=1e-5)

    def test_jit_eci_to_koe(self):
        """state_eci_to_koe is JIT-compilable."""
        # Use a non-degenerate orbit to avoid e≈0 float32 sensitivity
        oe = jnp.array([7000e3, 0.01, 0.5, 0.3, 0.2, 0.8])
        state = state_koe_to_eci(oe)
        eager = state_eci_to_koe(state)
        jitted = jax.jit(state_eci_to_koe)(state)
        assert jnp.allclose(eager, jitted, atol=1e-5)

    def test_vmap_geocentric(self):
        """vmap over batch of geocentric coordinates."""
        coords = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 100e3],
            [-0.5, 0.3, 200e3],
        ])
        ecef = jax.vmap(position_geocentric_to_ecef)(coords)
        assert ecef.shape == (3, 3)

        back = jax.vmap(position_ecef_to_geocentric)(ecef)
        assert back.shape == (3, 3)
        assert jnp.allclose(back, coords, atol=_GEOC_ROUNDTRIP_TOL)

    def test_vmap_geodetic(self):
        """vmap over batch of geodetic coordinates."""
        coords = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 100e3],
            [-0.5, 0.3, 200e3],
        ])
        ecef = jax.vmap(position_geodetic_to_ecef)(coords)
        assert ecef.shape == (3, 3)

        back = jax.vmap(position_ecef_to_geodetic)(ecef)
        assert back.shape == (3, 3)
        assert jnp.allclose(back[:, :2], coords[:, :2], atol=_ANGLE_TOL)
        assert jnp.allclose(back[:, 2], coords[:, 2], atol=_GEOD_ROUNDTRIP_TOL)

    def test_vmap_koe_to_eci(self):
        """vmap over batch of orbital elements."""
        oes = jnp.array([
            [7000e3, 0.001, 0.5, 0.3, 0.2, 0.8],
            [8000e3, 0.01, 1.0, 0.1, 0.5, 1.5],
            [42164e3, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        states = jax.vmap(state_koe_to_eci)(oes)
        assert states.shape == (3, 6)

    def test_vmap_eci_to_koe(self):
        """vmap over batch of ECI states."""
        oes = jnp.array([
            [7000e3, 0.001, 0.5, 0.3, 0.2, 0.8],
            [8000e3, 0.01, 1.0, 0.1, 0.5, 1.5],
        ])
        states = jax.vmap(state_koe_to_eci)(oes)
        oes_back = jax.vmap(state_eci_to_koe)(states)
        assert oes_back.shape == (2, 6)
        assert jnp.allclose(oes_back[:, 0], oes[:, 0], atol=_SMA_TOL)
