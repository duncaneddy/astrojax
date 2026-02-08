"""Cross-validation tests comparing astrojax.coordinates output against brahe 1.0+.

Brahe uses float64 internally while astrojax uses float32. Tolerances
are set to accommodate the precision difference (~7 significant digits
for float32 vs ~15 for float64).
"""

import brahe as bh
import jax.numpy as jnp
import numpy as np
import pytest

from astrojax.coordinates import (
    position_ecef_to_geocentric,
    position_ecef_to_geodetic,
    position_enz_to_azel,
    position_geocentric_to_ecef,
    position_geodetic_to_ecef,
    relative_position_ecef_to_enz,
    relative_position_enz_to_ecef,
    rotation_ellipsoid_to_enz,
    rotation_enz_to_ellipsoid,
    state_eci_to_koe,
    state_koe_to_eci,
)

DEGREES = bh.AngleFormat.DEGREES
RADIANS = bh.AngleFormat.RADIANS

# Float32 vs float64 tolerances
_REL_TOL = 5e-6  # relative tolerance (~float32 precision)
_POS_ATOL = 1.0  # metres
_ALT_ATOL = 1.0  # metres
_ANGLE_DEG_ATOL = 0.01  # degrees
_ANGLE_RAD_ATOL = 2e-4  # radians
_SMA_ATOL = 100.0  # metres (vis-viva float32 roundoff on ~7e6 m)
_ECC_ATOL = 5e-4  # dimensionless (float32 precision)
_VEL_ATOL = 0.1  # m/s


# ──────────────────────────────────────────────
# Geocentric coordinate transformations
# ──────────────────────────────────────────────


class TestGeocentricToECEFVsBrahe:
    @pytest.mark.parametrize(
        "lon_deg, lat_deg, alt",
        [
            (0.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            (0.0, 90.0, 0.0),
            (-45.0, 30.0, 0.0),
            (77.875, 20.9752, 0.0),
            (0.0, 0.0, 500e3),
            (120.0, -35.0, 200e3),
        ],
    )
    def test_geocentric_to_ecef_degrees(self, lon_deg, lat_deg, alt):
        """position_geocentric_to_ecef matches brahe in degrees."""
        x_geoc = np.array([lon_deg, lat_deg, alt])

        expected = bh.position_geocentric_to_ecef(x_geoc, DEGREES)
        actual = np.array(position_geocentric_to_ecef(
            jnp.array(x_geoc), use_degrees=True
        ))

        np.testing.assert_allclose(
            actual, expected,
            atol=_POS_ATOL, rtol=_REL_TOL,
            err_msg=f"geocentric_to_ecef mismatch for ({lon_deg}, {lat_deg}, {alt})",
        )

    @pytest.mark.parametrize(
        "lon_rad, lat_rad, alt",
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.5, 0.0),
            (-0.5, 0.3, 100e3),
            (np.pi / 2, 0.0, 0.0),
        ],
    )
    def test_geocentric_to_ecef_radians(self, lon_rad, lat_rad, alt):
        """position_geocentric_to_ecef matches brahe in radians."""
        x_geoc = np.array([lon_rad, lat_rad, alt])

        expected = bh.position_geocentric_to_ecef(x_geoc, RADIANS)
        actual = np.array(position_geocentric_to_ecef(
            jnp.array(x_geoc), use_degrees=False
        ))

        np.testing.assert_allclose(
            actual, expected,
            atol=_POS_ATOL, rtol=_REL_TOL,
            err_msg=f"geocentric_to_ecef mismatch for ({lon_rad}, {lat_rad}, {alt})",
        )


class TestECEFToGeocentricVsBrahe:
    @pytest.mark.parametrize(
        "x, y, z",
        [
            (bh.WGS84_A, 0.0, 0.0),
            (0.0, bh.WGS84_A, 0.0),
            (0.0, 0.0, bh.WGS84_A),
            (4e6, 3e6, 5e6),
            (-2e6, 5e6, -3e6),
            (bh.WGS84_A + 500e3, 0.0, 0.0),
        ],
    )
    def test_ecef_to_geocentric_degrees(self, x, y, z):
        """position_ecef_to_geocentric matches brahe in degrees."""
        x_ecef = np.array([x, y, z])

        expected = bh.position_ecef_to_geocentric(x_ecef, DEGREES)
        actual = np.array(position_ecef_to_geocentric(
            jnp.array(x_ecef), use_degrees=True
        ))

        # Compare angles
        np.testing.assert_allclose(
            actual[:2], expected[:2],
            atol=_ANGLE_DEG_ATOL, rtol=_REL_TOL,
            err_msg=f"ecef_to_geocentric angle mismatch for ({x}, {y}, {z})",
        )
        # Compare altitude
        np.testing.assert_allclose(
            actual[2], expected[2],
            atol=_ALT_ATOL, rtol=_REL_TOL,
            err_msg=f"ecef_to_geocentric alt mismatch for ({x}, {y}, {z})",
        )

    @pytest.mark.parametrize(
        "x, y, z",
        [
            (bh.WGS84_A, 0.0, 0.0),
            (4e6, 3e6, 5e6),
        ],
    )
    def test_ecef_to_geocentric_radians(self, x, y, z):
        """position_ecef_to_geocentric matches brahe in radians."""
        x_ecef = np.array([x, y, z])

        expected = bh.position_ecef_to_geocentric(x_ecef, RADIANS)
        actual = np.array(position_ecef_to_geocentric(
            jnp.array(x_ecef), use_degrees=False
        ))

        np.testing.assert_allclose(
            actual[:2], expected[:2],
            atol=_ANGLE_RAD_ATOL, rtol=_REL_TOL,
            err_msg=f"ecef_to_geocentric angle mismatch for ({x}, {y}, {z})",
        )
        np.testing.assert_allclose(
            actual[2], expected[2],
            atol=_ALT_ATOL, rtol=_REL_TOL,
        )


class TestGeocentricRoundTripVsBrahe:
    @pytest.mark.parametrize(
        "lon_deg, lat_deg, alt",
        [
            (0.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            (77.875, 20.9752, 0.0),
            (-120.0, 45.0, 300e3),
        ],
    )
    def test_roundtrip_via_brahe_ecef(self, lon_deg, lat_deg, alt):
        """astrojax forward → brahe inverse ≈ identity."""
        x_geoc = jnp.array([lon_deg, lat_deg, alt])

        # astrojax forward
        ecef = np.array(position_geocentric_to_ecef(x_geoc, use_degrees=True))
        # brahe inverse
        back = bh.position_ecef_to_geocentric(ecef, DEGREES)

        np.testing.assert_allclose(
            back[:2], np.array([lon_deg, lat_deg]),
            atol=_ANGLE_DEG_ATOL, rtol=_REL_TOL,
        )
        np.testing.assert_allclose(back[2], alt, atol=_ALT_ATOL, rtol=_REL_TOL)


# ──────────────────────────────────────────────
# Geodetic coordinate transformations
# ──────────────────────────────────────────────


class TestGeodeticToECEFVsBrahe:
    @pytest.mark.parametrize(
        "lon_deg, lat_deg, alt",
        [
            (0.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            (0.0, 90.0, 0.0),
            (-105.0, 40.0, 1655.0),  # Boulder, CO
            (77.875, 20.9752, 0.0),
            (0.0, 0.0, 500e3),
            (120.0, -35.0, 200e3),
            (-73.9857, 40.7484, 443.0),  # Empire State Building
        ],
    )
    def test_geodetic_to_ecef_degrees(self, lon_deg, lat_deg, alt):
        """position_geodetic_to_ecef matches brahe in degrees."""
        x_geod = np.array([lon_deg, lat_deg, alt])

        expected = bh.position_geodetic_to_ecef(x_geod, DEGREES)
        actual = np.array(position_geodetic_to_ecef(
            jnp.array(x_geod), use_degrees=True
        ))

        np.testing.assert_allclose(
            actual, expected,
            atol=_POS_ATOL, rtol=_REL_TOL,
            err_msg=f"geodetic_to_ecef mismatch for ({lon_deg}, {lat_deg}, {alt})",
        )

    @pytest.mark.parametrize(
        "lon_rad, lat_rad, alt",
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.5, 0.0),
            (-0.5, 0.3, 100e3),
        ],
    )
    def test_geodetic_to_ecef_radians(self, lon_rad, lat_rad, alt):
        """position_geodetic_to_ecef matches brahe in radians."""
        x_geod = np.array([lon_rad, lat_rad, alt])

        expected = bh.position_geodetic_to_ecef(x_geod, RADIANS)
        actual = np.array(position_geodetic_to_ecef(
            jnp.array(x_geod), use_degrees=False
        ))

        np.testing.assert_allclose(
            actual, expected,
            atol=_POS_ATOL, rtol=_REL_TOL,
            err_msg=f"geodetic_to_ecef mismatch for ({lon_rad}, {lat_rad}, {alt})",
        )


class TestECEFToGeodeticVsBrahe:
    @pytest.mark.parametrize(
        "x, y, z",
        [
            (bh.WGS84_A, 0.0, 0.0),
            (0.0, bh.WGS84_A, 0.0),
            (0.0, 0.0, bh.WGS84_A * (1.0 - bh.WGS84_F)),  # pole at semi-minor
            (4e6, 3e6, 5e6),
            (-2e6, 5e6, -3e6),
            (-1275936.0, -4797210.0, 4020109.0),  # from brahe docs example
        ],
    )
    def test_ecef_to_geodetic_degrees(self, x, y, z):
        """position_ecef_to_geodetic matches brahe in degrees."""
        x_ecef = np.array([x, y, z])

        expected = bh.position_ecef_to_geodetic(x_ecef, DEGREES)
        actual = np.array(position_ecef_to_geodetic(
            jnp.array(x_ecef), use_degrees=True
        ))

        # Compare angles
        np.testing.assert_allclose(
            actual[:2], expected[:2],
            atol=_ANGLE_DEG_ATOL, rtol=_REL_TOL,
            err_msg=f"ecef_to_geodetic angle mismatch for ({x}, {y}, {z})",
        )
        # Compare altitude
        np.testing.assert_allclose(
            actual[2], expected[2],
            atol=_ALT_ATOL, rtol=_REL_TOL,
            err_msg=f"ecef_to_geodetic alt mismatch for ({x}, {y}, {z})",
        )

    @pytest.mark.parametrize(
        "x, y, z",
        [
            (bh.WGS84_A, 0.0, 0.0),
            (4e6, 3e6, 5e6),
        ],
    )
    def test_ecef_to_geodetic_radians(self, x, y, z):
        """position_ecef_to_geodetic matches brahe in radians."""
        x_ecef = np.array([x, y, z])

        expected = bh.position_ecef_to_geodetic(x_ecef, RADIANS)
        actual = np.array(position_ecef_to_geodetic(
            jnp.array(x_ecef), use_degrees=False
        ))

        np.testing.assert_allclose(
            actual[:2], expected[:2],
            atol=_ANGLE_RAD_ATOL, rtol=_REL_TOL,
        )
        np.testing.assert_allclose(
            actual[2], expected[2],
            atol=_ALT_ATOL, rtol=_REL_TOL,
        )


class TestGeodeticRoundTripVsBrahe:
    @pytest.mark.parametrize(
        "lon_deg, lat_deg, alt",
        [
            (0.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            (77.875, 20.9752, 0.0),
            (-105.0, 40.0, 1655.0),
        ],
    )
    def test_roundtrip_via_brahe_ecef(self, lon_deg, lat_deg, alt):
        """astrojax forward → brahe inverse ≈ identity."""
        x_geod = jnp.array([lon_deg, lat_deg, alt])

        # astrojax forward
        ecef = np.array(position_geodetic_to_ecef(x_geod, use_degrees=True))
        # brahe inverse
        back = bh.position_ecef_to_geodetic(ecef, DEGREES)

        np.testing.assert_allclose(
            back[:2], np.array([lon_deg, lat_deg]),
            atol=_ANGLE_DEG_ATOL, rtol=_REL_TOL,
        )
        np.testing.assert_allclose(back[2], alt, atol=_ALT_ATOL, rtol=_REL_TOL)

    @pytest.mark.parametrize(
        "lon_deg, lat_deg, alt",
        [
            (0.0, 0.0, 0.0),
            (-105.0, 40.0, 1655.0),
            (120.0, -35.0, 200e3),
        ],
    )
    def test_roundtrip_brahe_forward_astrojax_inverse(self, lon_deg, lat_deg, alt):
        """brahe forward → astrojax inverse ≈ identity."""
        x_geod = np.array([lon_deg, lat_deg, alt])

        # brahe forward
        ecef = bh.position_geodetic_to_ecef(x_geod, DEGREES)
        # astrojax inverse
        back = np.array(position_ecef_to_geodetic(
            jnp.array(ecef), use_degrees=True
        ))

        np.testing.assert_allclose(
            back[:2], np.array([lon_deg, lat_deg]),
            atol=_ANGLE_DEG_ATOL, rtol=_REL_TOL,
        )
        np.testing.assert_allclose(back[2], alt, atol=_ALT_ATOL, rtol=_REL_TOL)


# ──────────────────────────────────────────────
# Keplerian orbital element conversions
# ──────────────────────────────────────────────


class TestKOEToECIVsBrahe:
    @pytest.mark.parametrize(
        "a, e, i, raan, omega, M",
        [
            (bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0),
            (bh.R_EARTH + 500e3, 0.0, 90.0, 0.0, 0.0, 0.0),
            (7000e3, 0.001, 98.0, 15.0, 30.0, 45.0),
            (26560e3, 0.74, 63.4, 250.0, 90.0, 180.0),
            (42164e3, 0.0, 0.0, 0.0, 0.0, 0.0),
            (7000e3, 0.5, 45.0, 120.0, 270.0, 300.0),
        ],
    )
    def test_koe_to_eci_degrees(self, a, e, i, raan, omega, M):
        """state_koe_to_eci matches brahe in degrees."""
        x_oe = np.array([a, e, i, raan, omega, M])

        expected = bh.state_koe_to_eci(x_oe, DEGREES)
        actual = np.array(state_koe_to_eci(
            jnp.array(x_oe), use_degrees=True
        ))

        # Position tolerance
        np.testing.assert_allclose(
            actual[:3], expected[:3],
            atol=_POS_ATOL, rtol=_REL_TOL,
            err_msg=f"koe_to_eci position mismatch for a={a}, e={e}",
        )
        # Velocity tolerance
        np.testing.assert_allclose(
            actual[3:], expected[3:],
            atol=_VEL_ATOL, rtol=_REL_TOL,
            err_msg=f"koe_to_eci velocity mismatch for a={a}, e={e}",
        )

    @pytest.mark.parametrize(
        "a, e, i, raan, omega, M",
        [
            (7000e3, 0.001, 1.7104, 0.2618, 0.5236, 0.7854),
            (26560e3, 0.74, 1.1066, 4.3633, 1.5708, np.pi),
        ],
    )
    def test_koe_to_eci_radians(self, a, e, i, raan, omega, M):
        """state_koe_to_eci matches brahe in radians."""
        x_oe = np.array([a, e, i, raan, omega, M])

        expected = bh.state_koe_to_eci(x_oe, RADIANS)
        actual = np.array(state_koe_to_eci(
            jnp.array(x_oe), use_degrees=False
        ))

        np.testing.assert_allclose(
            actual[:3], expected[:3],
            atol=_POS_ATOL, rtol=_REL_TOL,
        )
        np.testing.assert_allclose(
            actual[3:], expected[3:],
            atol=_VEL_ATOL, rtol=_REL_TOL,
        )


class TestECIToKOEVsBrahe:
    @pytest.mark.parametrize(
        "a, e, i, raan, omega, M",
        [
            (bh.R_EARTH + 500e3, 0.001, 45.0, 30.0, 60.0, 90.0),
            (7000e3, 0.01, 98.0, 15.0, 30.0, 45.0),
            (26560e3, 0.74, 63.4, 250.0, 90.0, 180.0),
            (7000e3, 0.5, 45.0, 120.0, 270.0, 300.0),
        ],
    )
    def test_eci_to_koe_degrees(self, a, e, i, raan, omega, M):
        """state_eci_to_koe matches brahe for non-degenerate orbits in degrees."""
        # Generate a Cartesian state from brahe (float64 truth)
        x_oe = np.array([a, e, i, raan, omega, M])
        x_cart = bh.state_koe_to_eci(x_oe, DEGREES)

        expected = bh.state_eci_to_koe(x_cart, DEGREES)
        actual = np.array(state_eci_to_koe(
            jnp.array(x_cart), use_degrees=True
        ))

        # Semi-major axis
        np.testing.assert_allclose(
            actual[0], expected[0],
            atol=_SMA_ATOL, rtol=_REL_TOL,
            err_msg=f"eci_to_koe SMA mismatch for a={a}",
        )
        # Eccentricity
        np.testing.assert_allclose(
            actual[1], expected[1],
            atol=_ECC_ATOL, rtol=_REL_TOL,
            err_msg=f"eci_to_koe ecc mismatch for e={e}",
        )
        # Angular elements
        for idx, name in [(2, "i"), (3, "RAAN"), (4, "omega"), (5, "M")]:
            np.testing.assert_allclose(
                actual[idx], expected[idx],
                atol=_ANGLE_DEG_ATOL, rtol=_REL_TOL,
                err_msg=f"eci_to_koe {name} mismatch for a={a}, e={e}",
            )

    @pytest.mark.parametrize(
        "a, e, i, raan, omega, M",
        [
            (7000e3, 0.01, 1.7104, 0.2618, 0.5236, 0.7854),
            (26560e3, 0.74, 1.1066, 4.3633, 1.5708, np.pi),
        ],
    )
    def test_eci_to_koe_radians(self, a, e, i, raan, omega, M):
        """state_eci_to_koe matches brahe for non-degenerate orbits in radians."""
        x_oe = np.array([a, e, i, raan, omega, M])
        x_cart = bh.state_koe_to_eci(x_oe, RADIANS)

        expected = bh.state_eci_to_koe(x_cart, RADIANS)
        actual = np.array(state_eci_to_koe(
            jnp.array(x_cart), use_degrees=False
        ))

        np.testing.assert_allclose(
            actual[0], expected[0],
            atol=_SMA_ATOL, rtol=_REL_TOL,
        )
        np.testing.assert_allclose(
            actual[1], expected[1],
            atol=_ECC_ATOL, rtol=_REL_TOL,
        )
        for idx in range(2, 6):
            np.testing.assert_allclose(
                actual[idx], expected[idx],
                atol=_ANGLE_RAD_ATOL, rtol=_REL_TOL,
            )


class TestKeplerianRoundTripVsBrahe:
    @pytest.mark.parametrize(
        "a, e, i, raan, omega, M",
        [
            (7000e3, 0.001, 98.0, 15.0, 30.0, 45.0),
            (26560e3, 0.74, 63.4, 250.0, 90.0, 180.0),
            (42164e3, 0.001, 0.1, 0.0, 0.0, 0.0),
            (7000e3, 0.5, 45.0, 120.0, 270.0, 300.0),
        ],
    )
    def test_astrojax_forward_brahe_inverse(self, a, e, i, raan, omega, M):
        """astrojax KOE→ECI, brahe ECI→KOE recovers original elements."""
        x_oe = jnp.array([a, e, i, raan, omega, M])

        # astrojax forward
        x_cart = np.array(state_koe_to_eci(x_oe, use_degrees=True))
        # brahe inverse
        oe_back = bh.state_eci_to_koe(x_cart, DEGREES)

        np.testing.assert_allclose(oe_back[0], a, atol=_SMA_ATOL, rtol=_REL_TOL)
        np.testing.assert_allclose(oe_back[1], e, atol=_ECC_ATOL, rtol=_REL_TOL)
        for idx, val in [(2, i), (3, raan), (4, omega), (5, M)]:
            np.testing.assert_allclose(
                oe_back[idx], val,
                atol=_ANGLE_DEG_ATOL, rtol=_REL_TOL,
            )

    @pytest.mark.parametrize(
        "a, e, i, raan, omega, M",
        [
            (7000e3, 0.001, 98.0, 15.0, 30.0, 45.0),
            (26560e3, 0.74, 63.4, 250.0, 90.0, 180.0),
            (7000e3, 0.5, 45.0, 120.0, 270.0, 300.0),
        ],
    )
    def test_brahe_forward_astrojax_inverse(self, a, e, i, raan, omega, M):
        """brahe KOE→ECI, astrojax ECI→KOE recovers original elements."""
        x_oe = np.array([a, e, i, raan, omega, M])

        # brahe forward
        x_cart = bh.state_koe_to_eci(x_oe, DEGREES)
        # astrojax inverse
        oe_back = np.array(state_eci_to_koe(
            jnp.array(x_cart), use_degrees=True
        ))

        np.testing.assert_allclose(oe_back[0], a, atol=_SMA_ATOL, rtol=_REL_TOL)
        np.testing.assert_allclose(oe_back[1], e, atol=_ECC_ATOL, rtol=_REL_TOL)
        for idx, val in [(2, i), (3, raan), (4, omega), (5, M)]:
            np.testing.assert_allclose(
                oe_back[idx], val,
                atol=_ANGLE_DEG_ATOL, rtol=_REL_TOL,
            )


# ──────────────────────────────────────────────
# ENZ Topocentric coordinate transformations
# ──────────────────────────────────────────────

GEODETIC = bh.EllipsoidalConversionType.GEODETIC
GEOCENTRIC = bh.EllipsoidalConversionType.GEOCENTRIC

# Tolerances for rotation matrix and position comparisons
_ROT_ATOL = 1e-5  # rotation matrix element (float32 vs float64)
_ENZ_POS_ATOL = 1.0  # metres
_AZEL_ATOL = 0.01  # degrees


class TestRotationEllipsoidToENZVsBrahe:
    @pytest.mark.parametrize(
        "lon_deg, lat_deg, alt",
        [
            (0.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            (0.0, 90.0, 0.0),
            (30.0, 60.0, 0.0),
            (-105.0, 40.0, 1655.0),
            (42.1, 53.9, 100.0),
        ],
    )
    def test_rotation_degrees(self, lon_deg, lat_deg, alt):
        """rotation_ellipsoid_to_enz matches brahe in degrees."""
        x = np.array([lon_deg, lat_deg, alt])

        expected = bh.rotation_ellipsoid_to_enz(x, DEGREES)
        actual = np.array(rotation_ellipsoid_to_enz(
            jnp.array(x), use_degrees=True
        ))

        np.testing.assert_allclose(
            actual, expected,
            atol=_ROT_ATOL, rtol=_REL_TOL,
            err_msg=f"rotation mismatch for ({lon_deg}, {lat_deg}, {alt})",
        )

    @pytest.mark.parametrize(
        "lon_rad, lat_rad, alt",
        [
            (0.0, 0.0, 0.0),
            (0.5, 1.0, 0.0),
            (-1.8, 0.7, 100e3),
        ],
    )
    def test_rotation_radians(self, lon_rad, lat_rad, alt):
        """rotation_ellipsoid_to_enz matches brahe in radians."""
        x = np.array([lon_rad, lat_rad, alt])

        expected = bh.rotation_ellipsoid_to_enz(x, RADIANS)
        actual = np.array(rotation_ellipsoid_to_enz(
            jnp.array(x), use_degrees=False
        ))

        np.testing.assert_allclose(
            actual, expected,
            atol=_ROT_ATOL, rtol=_REL_TOL,
        )


class TestRotationENZToEllipsoidVsBrahe:
    @pytest.mark.parametrize(
        "lon_deg, lat_deg, alt",
        [
            (0.0, 0.0, 0.0),
            (42.1, 53.9, 100.0),
            (-73.9, 40.7, 443.0),
        ],
    )
    def test_inverse_rotation_degrees(self, lon_deg, lat_deg, alt):
        """rotation_enz_to_ellipsoid matches brahe in degrees."""
        x = np.array([lon_deg, lat_deg, alt])

        expected = bh.rotation_enz_to_ellipsoid(x, DEGREES)
        actual = np.array(rotation_enz_to_ellipsoid(
            jnp.array(x), use_degrees=True
        ))

        np.testing.assert_allclose(
            actual, expected,
            atol=_ROT_ATOL, rtol=_REL_TOL,
        )


class TestRelativePositionECEFToENZVsBrahe:
    @pytest.mark.parametrize(
        "station, target, conv_type, use_geodetic",
        [
            # 100m overhead (geocentric)
            (
                [bh.R_EARTH, 0.0, 0.0],
                [bh.R_EARTH + 100.0, 0.0, 0.0],
                GEOCENTRIC, False,
            ),
            # 100m north (geocentric)
            (
                [bh.R_EARTH, 0.0, 0.0],
                [bh.R_EARTH, 0.0, 100.0],
                GEOCENTRIC, False,
            ),
            # 100m east (geocentric)
            (
                [bh.R_EARTH, 0.0, 0.0],
                [bh.R_EARTH, 100.0, 0.0],
                GEOCENTRIC, False,
            ),
            # 100m overhead (geodetic)
            (
                [bh.R_EARTH, 0.0, 0.0],
                [bh.R_EARTH + 100.0, 0.0, 0.0],
                GEODETIC, True,
            ),
            # Non-trivial station at ~45° lat (geocentric)
            (
                [4.5e6, 0.0, 4.5e6],
                [4.5e6 + 500.0, 300.0, 4.5e6 + 200.0],
                GEOCENTRIC, False,
            ),
            # Non-trivial station at ~45° lat (geodetic)
            (
                [4.5e6, 0.0, 4.5e6],
                [4.5e6 + 500.0, 300.0, 4.5e6 + 200.0],
                GEODETIC, True,
            ),
        ],
    )
    def test_ecef_to_enz(self, station, target, conv_type, use_geodetic):
        """relative_position_ecef_to_enz matches brahe."""
        sta = np.array(station)
        tgt = np.array(target)

        expected = bh.relative_position_ecef_to_enz(sta, tgt, conv_type)
        actual = np.array(relative_position_ecef_to_enz(
            jnp.array(sta), jnp.array(tgt), use_geodetic=use_geodetic
        ))

        np.testing.assert_allclose(
            actual, expected,
            atol=_ENZ_POS_ATOL, rtol=_REL_TOL,
            err_msg=f"ecef_to_enz mismatch for station={station}, target={target}",
        )


class TestRelativePositionENZToECEFVsBrahe:
    @pytest.mark.parametrize(
        "station, enz, conv_type, use_geodetic",
        [
            # Zenith at equator (geodetic)
            (
                [bh.R_EARTH, 0.0, 0.0],
                [0.0, 0.0, 100.0],
                GEODETIC, True,
            ),
            # Zenith at equator (geocentric)
            (
                [bh.R_EARTH, 0.0, 0.0],
                [0.0, 0.0, 100.0],
                GEOCENTRIC, False,
            ),
            # Off-axis at mid-latitude (geodetic)
            (
                [4.5e6, 0.0, 4.5e6],
                [100.0, 200.0, 300.0],
                GEODETIC, True,
            ),
            # Off-axis at mid-latitude (geocentric)
            (
                [4.5e6, 0.0, 4.5e6],
                [100.0, 200.0, 300.0],
                GEOCENTRIC, False,
            ),
        ],
    )
    def test_enz_to_ecef(self, station, enz, conv_type, use_geodetic):
        """relative_position_enz_to_ecef matches brahe."""
        sta = np.array(station)
        enz_arr = np.array(enz)

        expected = bh.relative_position_enz_to_ecef(sta, enz_arr, conv_type)
        actual = np.array(relative_position_enz_to_ecef(
            jnp.array(sta), jnp.array(enz_arr), use_geodetic=use_geodetic
        ))

        np.testing.assert_allclose(
            actual, expected,
            atol=_ENZ_POS_ATOL, rtol=_REL_TOL,
            err_msg=f"enz_to_ecef mismatch for station={station}, enz={enz}",
        )


class TestPositionENZToAzElVsBrahe:
    @pytest.mark.parametrize(
        "e, n, z",
        [
            (0.0, 0.0, 100.0),      # directly above
            (0.0, 100.0, 0.0),      # due north
            (100.0, 0.0, 0.0),      # due east
            (-100.0, 100.0, 0.0),   # northwest
            (50.0, 50.0, 100.0),    # general off-axis
            (0.0, -100.0, 0.0),     # due south
            (-100.0, 0.0, 0.0),     # due west
        ],
    )
    def test_enz_to_azel_degrees(self, e, n, z):
        """position_enz_to_azel matches brahe in degrees."""
        x_enz = np.array([e, n, z])

        expected = bh.position_enz_to_azel(x_enz, DEGREES)
        actual = np.array(position_enz_to_azel(
            jnp.array(x_enz), use_degrees=True
        ))

        np.testing.assert_allclose(
            actual, expected,
            atol=_AZEL_ATOL, rtol=_REL_TOL,
            err_msg=f"enz_to_azel mismatch for ({e}, {n}, {z})",
        )

    @pytest.mark.parametrize(
        "e, n, z",
        [
            (0.0, 0.0, 100.0),
            (100.0, 0.0, 0.0),
            (50.0, 50.0, 50.0),
        ],
    )
    def test_enz_to_azel_radians(self, e, n, z):
        """position_enz_to_azel matches brahe in radians."""
        x_enz = np.array([e, n, z])

        expected = bh.position_enz_to_azel(x_enz, RADIANS)
        actual = np.array(position_enz_to_azel(
            jnp.array(x_enz), use_degrees=False
        ))

        np.testing.assert_allclose(
            actual, expected,
            atol=_ANGLE_RAD_ATOL, rtol=_REL_TOL,
        )


class TestENZRoundTripVsBrahe:
    @pytest.mark.parametrize(
        "station, target, conv_type, use_geodetic",
        [
            (
                [bh.R_EARTH, 0.0, 0.0],
                [bh.R_EARTH + 500.0, 300.0, 200.0],
                GEOCENTRIC, False,
            ),
            (
                [bh.R_EARTH, 0.0, 0.0],
                [bh.R_EARTH + 500.0, 300.0, 200.0],
                GEODETIC, True,
            ),
            (
                [4.5e6, 0.0, 4.5e6],
                [4.5e6 + 1000.0, 500.0, 4.5e6 + 800.0],
                GEOCENTRIC, False,
            ),
        ],
    )
    def test_astrojax_forward_brahe_inverse(
        self, station, target, conv_type, use_geodetic
    ):
        """astrojax ECEF→ENZ, brahe ENZ→ECEF recovers target."""
        sta = np.array(station)
        tgt = np.array(target)

        # astrojax forward
        r_enz = np.array(relative_position_ecef_to_enz(
            jnp.array(sta), jnp.array(tgt), use_geodetic=use_geodetic
        ))
        # brahe inverse
        r_ecef_back = bh.relative_position_enz_to_ecef(sta, r_enz, conv_type)

        np.testing.assert_allclose(
            r_ecef_back, tgt,
            atol=_ENZ_POS_ATOL, rtol=_REL_TOL,
        )

    @pytest.mark.parametrize(
        "station, target, conv_type, use_geodetic",
        [
            (
                [bh.R_EARTH, 0.0, 0.0],
                [bh.R_EARTH + 500.0, 300.0, 200.0],
                GEOCENTRIC, False,
            ),
            (
                [bh.R_EARTH, 0.0, 0.0],
                [bh.R_EARTH + 500.0, 300.0, 200.0],
                GEODETIC, True,
            ),
        ],
    )
    def test_brahe_forward_astrojax_inverse(
        self, station, target, conv_type, use_geodetic
    ):
        """brahe ECEF→ENZ, astrojax ENZ→ECEF recovers target."""
        sta = np.array(station)
        tgt = np.array(target)

        # brahe forward
        r_enz = bh.relative_position_ecef_to_enz(sta, tgt, conv_type)
        # astrojax inverse
        r_ecef_back = np.array(relative_position_enz_to_ecef(
            jnp.array(sta), jnp.array(r_enz), use_geodetic=use_geodetic
        ))

        np.testing.assert_allclose(
            r_ecef_back, tgt,
            atol=_ENZ_POS_ATOL, rtol=_REL_TOL,
        )
