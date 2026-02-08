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
    rotation_ellipsoid_to_enz,
    rotation_enz_to_ellipsoid,
    relative_position_ecef_to_enz,
    relative_position_enz_to_ecef,
    position_enz_to_azel,
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


# ──────────────────────────────────────────────
# ENZ Topocentric transformations
# ──────────────────────────────────────────────

_ROT_TOL = 1e-5  # rotation matrix element tolerance (float32)
_ENZ_POS_TOL = 1.0  # metres
_AZEL_DEG_TOL = 1e-3  # degrees


class TestRotationEllipsoidToENZ:
    def test_origin_equator(self):
        """At (lon=0, lat=0): ECEF X→Z, Y→E, Z→N."""
        x = jnp.array([0.0, 0.0, 0.0])
        rot = rotation_ellipsoid_to_enz(x)

        # ECEF X [1,0,0] → ENZ Zenith [0,0,1]
        enz = rot @ jnp.array([1.0, 0.0, 0.0])
        assert jnp.abs(enz[0]) < _ROT_TOL
        assert jnp.abs(enz[1]) < _ROT_TOL
        assert jnp.abs(enz[2] - 1.0) < _ROT_TOL

        # ECEF Y [0,1,0] → ENZ East [1,0,0]
        enz = rot @ jnp.array([0.0, 1.0, 0.0])
        assert jnp.abs(enz[0] - 1.0) < _ROT_TOL
        assert jnp.abs(enz[1]) < _ROT_TOL
        assert jnp.abs(enz[2]) < _ROT_TOL

        # ECEF Z [0,0,1] → ENZ North [0,1,0]
        enz = rot @ jnp.array([0.0, 0.0, 1.0])
        assert jnp.abs(enz[0]) < _ROT_TOL
        assert jnp.abs(enz[1] - 1.0) < _ROT_TOL
        assert jnp.abs(enz[2]) < _ROT_TOL

    def test_lon_90(self):
        """At (lon=90, lat=0): ECEF X→-E, Y→Z, Z→N."""
        x = jnp.array([90.0, 0.0, 0.0])
        rot = rotation_ellipsoid_to_enz(x, use_degrees=True)

        # ECEF X → -East
        enz = rot @ jnp.array([1.0, 0.0, 0.0])
        assert jnp.abs(enz[0] - (-1.0)) < _ROT_TOL
        assert jnp.abs(enz[1]) < _ROT_TOL
        assert jnp.abs(enz[2]) < _ROT_TOL

        # ECEF Y → Zenith
        enz = rot @ jnp.array([0.0, 1.0, 0.0])
        assert jnp.abs(enz[0]) < _ROT_TOL
        assert jnp.abs(enz[1]) < _ROT_TOL
        assert jnp.abs(enz[2] - 1.0) < _ROT_TOL

    def test_lat_90(self):
        """At (lon=0, lat=90): ECEF X→-N, Y→E, Z→Z."""
        x = jnp.array([0.0, 90.0, 0.0])
        rot = rotation_ellipsoid_to_enz(x, use_degrees=True)

        # ECEF X → -North
        enz = rot @ jnp.array([1.0, 0.0, 0.0])
        assert jnp.abs(enz[0]) < _ROT_TOL
        assert jnp.abs(enz[1] - (-1.0)) < _ROT_TOL
        assert jnp.abs(enz[2]) < _ROT_TOL

        # ECEF Z → Zenith
        enz = rot @ jnp.array([0.0, 0.0, 1.0])
        assert jnp.abs(enz[0]) < _ROT_TOL
        assert jnp.abs(enz[1]) < _ROT_TOL
        assert jnp.abs(enz[2] - 1.0) < _ROT_TOL

    def test_determinant_is_one(self):
        """Rotation matrix has determinant 1.0."""
        x = jnp.array([42.1, 53.9, 100.0])
        rot = rotation_ellipsoid_to_enz(x, use_degrees=True)
        assert jnp.abs(jnp.linalg.det(rot) - 1.0) < _ROT_TOL

    def test_degrees_radians_consistent(self):
        """Degrees and radians inputs give the same rotation matrix."""
        lon_deg, lat_deg = 30.0, 60.0
        rot_deg = rotation_ellipsoid_to_enz(
            jnp.array([lon_deg, lat_deg, 0.0]), use_degrees=True
        )
        rot_rad = rotation_ellipsoid_to_enz(
            jnp.array([lon_deg * DEG2RAD, lat_deg * DEG2RAD, 0.0]),
            use_degrees=False,
        )
        assert jnp.allclose(rot_deg, rot_rad, atol=_ROT_TOL)


class TestRotationENZToEllipsoid:
    def test_inverse_is_transpose(self):
        """R * R^T = I for an arbitrary location."""
        x = jnp.array([42.1, 53.9, 100.0])
        rot = rotation_ellipsoid_to_enz(x, use_degrees=True)
        rot_t = rotation_enz_to_ellipsoid(x, use_degrees=True)

        identity = rot @ rot_t
        assert jnp.allclose(identity, jnp.eye(3), atol=_ROT_TOL)

    def test_inverse_at_origin(self):
        """R * R^T = I at (0,0,0)."""
        x = jnp.array([0.0, 0.0, 0.0])
        identity = rotation_ellipsoid_to_enz(x) @ rotation_enz_to_ellipsoid(x)
        assert jnp.allclose(identity, jnp.eye(3), atol=_ROT_TOL)


class TestRelativePositionECEFToENZ:
    def test_overhead_geocentric(self):
        """100m overhead at equator → [0, 0, 100] in ENZ (geocentric)."""
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        r_ecef = jnp.array([R_EARTH + 100.0, 0.0, 0.0])
        r_enz = relative_position_ecef_to_enz(x_sta, r_ecef, use_geodetic=False)

        assert jnp.abs(r_enz[0]) < _ENZ_POS_TOL
        assert jnp.abs(r_enz[1]) < _ENZ_POS_TOL
        assert jnp.abs(r_enz[2] - 100.0) < _ENZ_POS_TOL

    def test_north_geocentric(self):
        """100m north at equator → [0, 100, 0] in ENZ (geocentric)."""
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        r_ecef = jnp.array([R_EARTH, 0.0, 100.0])
        r_enz = relative_position_ecef_to_enz(x_sta, r_ecef, use_geodetic=False)

        assert jnp.abs(r_enz[0]) < _ENZ_POS_TOL
        assert jnp.abs(r_enz[1] - 100.0) < _ENZ_POS_TOL
        assert jnp.abs(r_enz[2]) < _ENZ_POS_TOL

    def test_east_geocentric(self):
        """100m east at equator → [100, 0, 0] in ENZ (geocentric)."""
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        r_ecef = jnp.array([R_EARTH, 100.0, 0.0])
        r_enz = relative_position_ecef_to_enz(x_sta, r_ecef, use_geodetic=False)

        assert jnp.abs(r_enz[0] - 100.0) < _ENZ_POS_TOL
        assert jnp.abs(r_enz[1]) < _ENZ_POS_TOL
        assert jnp.abs(r_enz[2]) < _ENZ_POS_TOL

    def test_geodetic_differs_from_geocentric(self):
        """Geodetic and geocentric give different results at mid-latitude."""
        # Station at ~45 deg latitude where geodetic/geocentric differ most
        x_sta = position_geocentric_to_ecef(
            jnp.array([0.0, 45.0, 0.0]), use_degrees=True
        )
        r_ecef = x_sta + jnp.array([1000.0, 500.0, 500.0])

        r_geod = relative_position_ecef_to_enz(x_sta, r_ecef, use_geodetic=True)
        r_geoc = relative_position_ecef_to_enz(x_sta, r_ecef, use_geodetic=False)

        # Geodetic and geocentric latitudes differ at mid-latitudes
        assert not jnp.allclose(r_geod, r_geoc, atol=0.01)

    def test_overhead_geodetic(self):
        """100m overhead at equator → [0, 0, 100] in ENZ (geodetic)."""
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        r_ecef = jnp.array([R_EARTH + 100.0, 0.0, 0.0])
        r_enz = relative_position_ecef_to_enz(x_sta, r_ecef, use_geodetic=True)

        assert jnp.abs(r_enz[0]) < _ENZ_POS_TOL
        assert jnp.abs(r_enz[1]) < _ENZ_POS_TOL
        assert jnp.abs(r_enz[2] - 100.0) < _ENZ_POS_TOL


class TestRelativePositionENZToECEF:
    def test_inverse_overhead_geodetic(self):
        """[0, 0, 100] ENZ → R_EARTH+100 along ECEF X (geodetic at equator)."""
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        r_enz = jnp.array([0.0, 0.0, 100.0])
        r_ecef = relative_position_enz_to_ecef(x_sta, r_enz, use_geodetic=True)

        assert jnp.abs(r_ecef[0] - (R_EARTH + 100.0)) < _ENZ_POS_TOL
        assert jnp.abs(r_ecef[1]) < _ENZ_POS_TOL
        assert jnp.abs(r_ecef[2]) < _ENZ_POS_TOL

    def test_inverse_overhead_geocentric(self):
        """[0, 0, 100] ENZ → R_EARTH+100 along ECEF X (geocentric at equator)."""
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        r_enz = jnp.array([0.0, 0.0, 100.0])
        r_ecef = relative_position_enz_to_ecef(x_sta, r_enz, use_geodetic=False)

        assert jnp.abs(r_ecef[0] - (R_EARTH + 100.0)) < _ENZ_POS_TOL
        assert jnp.abs(r_ecef[1]) < _ENZ_POS_TOL
        assert jnp.abs(r_ecef[2]) < _ENZ_POS_TOL


class TestPositionENZToAzEl:
    def test_directly_above(self):
        """[0, 0, 100] → az=0, el=90, range=100."""
        azel = position_enz_to_azel(jnp.array([0.0, 0.0, 100.0]), use_degrees=True)

        assert jnp.abs(azel[0]) < _AZEL_DEG_TOL  # az=0
        assert jnp.abs(azel[1] - 90.0) < _AZEL_DEG_TOL  # el=90
        assert jnp.abs(azel[2] - 100.0) < _ENZ_POS_TOL  # range=100

    def test_due_north(self):
        """[0, 100, 0] → az=0, el=0, range=100."""
        azel = position_enz_to_azel(jnp.array([0.0, 100.0, 0.0]), use_degrees=True)

        assert jnp.abs(azel[0]) < _AZEL_DEG_TOL
        assert jnp.abs(azel[1]) < _AZEL_DEG_TOL
        assert jnp.abs(azel[2] - 100.0) < _ENZ_POS_TOL

    def test_due_east(self):
        """[100, 0, 0] → az=90, el=0, range=100."""
        azel = position_enz_to_azel(jnp.array([100.0, 0.0, 0.0]), use_degrees=True)

        assert jnp.abs(azel[0] - 90.0) < _AZEL_DEG_TOL
        assert jnp.abs(azel[1]) < _AZEL_DEG_TOL
        assert jnp.abs(azel[2] - 100.0) < _ENZ_POS_TOL

    def test_northwest(self):
        """[-100, 100, 0] → az=315, el=0, range=100*sqrt(2)."""
        azel = position_enz_to_azel(
            jnp.array([-100.0, 100.0, 0.0]), use_degrees=True
        )
        expected_range = 100.0 * jnp.sqrt(2.0)

        assert jnp.abs(azel[0] - 315.0) < _AZEL_DEG_TOL
        assert jnp.abs(azel[1]) < _AZEL_DEG_TOL
        assert jnp.abs(azel[2] - expected_range) < _ENZ_POS_TOL

    def test_degrees_radians_consistent(self):
        """Degrees and radians outputs are consistent."""
        x_enz = jnp.array([100.0, 50.0, 30.0])
        azel_deg = position_enz_to_azel(x_enz, use_degrees=True)
        azel_rad = position_enz_to_azel(x_enz, use_degrees=False)

        assert jnp.allclose(
            azel_deg[:2], jnp.rad2deg(azel_rad[:2]), atol=_AZEL_DEG_TOL
        )
        assert jnp.abs(azel_deg[2] - azel_rad[2]) < _ENZ_POS_TOL


class TestENZRoundTrip:
    def test_ecef_enz_ecef_geocentric(self):
        """ECEF → ENZ → ECEF round-trip with geocentric."""
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        r_ecef_orig = jnp.array([R_EARTH + 500.0, 300.0, 200.0])

        r_enz = relative_position_ecef_to_enz(
            x_sta, r_ecef_orig, use_geodetic=False
        )
        r_ecef_back = relative_position_enz_to_ecef(
            x_sta, r_enz, use_geodetic=False
        )
        assert jnp.allclose(r_ecef_back, r_ecef_orig, atol=_ENZ_POS_TOL)

    def test_ecef_enz_ecef_geodetic(self):
        """ECEF → ENZ → ECEF round-trip with geodetic."""
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        r_ecef_orig = jnp.array([R_EARTH + 500.0, 300.0, 200.0])

        r_enz = relative_position_ecef_to_enz(
            x_sta, r_ecef_orig, use_geodetic=True
        )
        r_ecef_back = relative_position_enz_to_ecef(
            x_sta, r_enz, use_geodetic=True
        )
        assert jnp.allclose(r_ecef_back, r_ecef_orig, atol=_ENZ_POS_TOL)

    def test_roundtrip_off_equator(self):
        """Round-trip at a non-trivial station location."""
        # Station at approximately Boulder, CO in geocentric ECEF
        x_sta = position_geocentric_to_ecef(
            jnp.array([-105.0, 40.0, 1655.0]), use_degrees=True
        )
        r_ecef_orig = x_sta + jnp.array([1000.0, 2000.0, 3000.0])

        for use_geodetic in [True, False]:
            r_enz = relative_position_ecef_to_enz(
                x_sta, r_ecef_orig, use_geodetic=use_geodetic
            )
            r_ecef_back = relative_position_enz_to_ecef(
                x_sta, r_enz, use_geodetic=use_geodetic
            )
            assert jnp.allclose(r_ecef_back, r_ecef_orig, atol=_ENZ_POS_TOL), (
                f"Round-trip failed for use_geodetic={use_geodetic}"
            )


class TestENZJAXCompatibility:
    def test_jit_rotation_ellipsoid_to_enz(self):
        """rotation_ellipsoid_to_enz is JIT-compilable."""
        x = jnp.array([0.5, 0.3, 0.0])
        eager = rotation_ellipsoid_to_enz(x)
        jitted = jax.jit(rotation_ellipsoid_to_enz)(x)
        assert jnp.allclose(eager, jitted, atol=1e-6)

    def test_jit_rotation_enz_to_ellipsoid(self):
        """rotation_enz_to_ellipsoid is JIT-compilable."""
        x = jnp.array([0.5, 0.3, 0.0])
        eager = rotation_enz_to_ellipsoid(x)
        jitted = jax.jit(rotation_enz_to_ellipsoid)(x)
        assert jnp.allclose(eager, jitted, atol=1e-6)

    def test_jit_relative_position_ecef_to_enz(self):
        """relative_position_ecef_to_enz is JIT-compilable."""
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        r_ecef = jnp.array([R_EARTH + 100.0, 50.0, 50.0])
        eager = relative_position_ecef_to_enz(x_sta, r_ecef, use_geodetic=False)
        jitted = jax.jit(
            lambda s, r: relative_position_ecef_to_enz(s, r, use_geodetic=False)
        )(x_sta, r_ecef)
        assert jnp.allclose(eager, jitted, atol=1e-5)

    def test_jit_relative_position_enz_to_ecef(self):
        """relative_position_enz_to_ecef is JIT-compilable."""
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        r_enz = jnp.array([100.0, 50.0, 200.0])
        eager = relative_position_enz_to_ecef(x_sta, r_enz, use_geodetic=False)
        jitted = jax.jit(
            lambda s, r: relative_position_enz_to_ecef(s, r, use_geodetic=False)
        )(x_sta, r_enz)
        assert jnp.allclose(eager, jitted, atol=1e-5)

    def test_jit_position_enz_to_azel(self):
        """position_enz_to_azel is JIT-compilable."""
        x = jnp.array([100.0, 50.0, 30.0])
        eager = position_enz_to_azel(x)
        jitted = jax.jit(position_enz_to_azel)(x)
        assert jnp.allclose(eager, jitted, atol=1e-6)

    def test_vmap_rotation_ellipsoid_to_enz(self):
        """vmap over batch of ellipsoidal coordinates."""
        coords = jnp.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.3, 0.0],
            [-1.0, 0.8, 100e3],
        ])
        rots = jax.vmap(rotation_ellipsoid_to_enz)(coords)
        assert rots.shape == (3, 3, 3)

    def test_vmap_position_enz_to_azel(self):
        """vmap over batch of ENZ positions."""
        enz_batch = jnp.array([
            [0.0, 0.0, 100.0],
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
        ])
        azels = jax.vmap(position_enz_to_azel)(enz_batch)
        assert azels.shape == (3, 3)

    def test_vmap_relative_position_ecef_to_enz(self):
        """vmap over batch of target positions with same station."""
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        targets = jnp.array([
            [R_EARTH + 100.0, 0.0, 0.0],
            [R_EARTH, 100.0, 0.0],
            [R_EARTH, 0.0, 100.0],
        ])
        # vmap over second argument only
        enz_batch = jax.vmap(
            lambda r: relative_position_ecef_to_enz(x_sta, r, use_geodetic=False)
        )(targets)
        assert enz_batch.shape == (3, 3)
