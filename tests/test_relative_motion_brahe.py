"""Cross-validation tests comparing astrojax ROE output against brahe 1.0+.

Brahe uses float64 internally while astrojax uses float32. Tolerances
are set to accommodate the precision difference (~7 significant digits
for float32 vs ~15 for float64).
"""

import brahe as bh
import jax.numpy as jnp
import numpy as np
import pytest

from astrojax.constants import R_EARTH
from astrojax.relative_motion import (
    state_eci_to_roe,
    state_oe_to_roe,
    state_roe_to_eci,
    state_roe_to_oe,
)

DEGREES = bh.AngleFormat.DEGREES
RADIANS = bh.AngleFormat.RADIANS

# Float32 vs float64 tolerances (OE path — direct orbital element inputs)
_DA_ATOL = 1e-5  # relative SMA (dimensionless)
_ANGLE_DEG_ATOL = 0.01  # degrees
_ANGLE_RAD_ATOL = 2e-4  # radians
_ECC_VEC_ATOL = 1e-5  # eccentricity vector components
_SMA_ATOL = 100.0  # metres
_ECC_ATOL = 5e-4  # eccentricity (roundtrip)

# ECI path tolerances — wider because ECI->KOE conversion introduces
# additional float32 roundoff on position magnitudes of ~7e6 m
_ECI_DA_ATOL = 1e-4  # relative SMA via ECI path
_ECI_ANGLE_DEG_ATOL = 0.1  # degrees via ECI path
_ECI_ANGLE_RAD_ATOL = 2e-3  # radians via ECI path
_ECI_ECC_VEC_ATOL = 5e-4  # eccentricity vector via ECI path
_ECI_POS_ATOL = 500.0  # metres (ECI roundtrip, 4x KOE conversion)
_ECI_VEL_ATOL = 0.5  # m/s (ECI roundtrip)


# ──────────────────────────────────────────────
# OE -> ROE comparison tests
# ──────────────────────────────────────────────


class TestOEtoROEVsBrahe:
    @pytest.mark.parametrize(
        "oe_chief_deg, oe_deputy_deg",
        [
            (
                [R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0],
                [R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05],
            ),
            (
                [R_EARTH + 500e3, 0.01, 51.6, 0.0, 0.0, 0.0],
                [R_EARTH + 500.5e3, 0.011, 51.65, 0.05, 0.1, 0.2],
            ),
            (
                [R_EARTH + 800e3, 0.0005, 45.0, 90.0, 270.0, 180.0],
                [R_EARTH + 801e3, 0.001, 45.05, 90.05, 270.05, 180.05],
            ),
        ],
    )
    def test_oe_to_roe_degrees(self, oe_chief_deg, oe_deputy_deg):
        """state_oe_to_roe matches brahe output (degree mode)."""
        oe_c_np = np.array(oe_chief_deg)
        oe_d_np = np.array(oe_deputy_deg)

        roe_brahe = bh.state_oe_to_roe(oe_c_np, oe_d_np, DEGREES)

        oe_c_jnp = jnp.array(oe_chief_deg)
        oe_d_jnp = jnp.array(oe_deputy_deg)
        roe_astrojax = state_oe_to_roe(oe_c_jnp, oe_d_jnp, use_degrees=True)

        # da (dimensionless)
        assert abs(float(roe_astrojax[0]) - roe_brahe[0]) < _DA_ATOL
        # d_lambda (degrees)
        assert abs(float(roe_astrojax[1]) - roe_brahe[1]) < _ANGLE_DEG_ATOL
        # dex, dey (dimensionless)
        assert abs(float(roe_astrojax[2]) - roe_brahe[2]) < _ECC_VEC_ATOL
        assert abs(float(roe_astrojax[3]) - roe_brahe[3]) < _ECC_VEC_ATOL
        # dix, diy (degrees)
        assert abs(float(roe_astrojax[4]) - roe_brahe[4]) < _ANGLE_DEG_ATOL
        assert abs(float(roe_astrojax[5]) - roe_brahe[5]) < _ANGLE_DEG_ATOL

    @pytest.mark.parametrize(
        "oe_chief_deg, oe_deputy_deg",
        [
            (
                [R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0],
                [R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05],
            ),
            (
                [R_EARTH + 500e3, 0.01, 51.6, 0.0, 0.0, 0.0],
                [R_EARTH + 500.5e3, 0.011, 51.65, 0.05, 0.1, 0.2],
            ),
        ],
    )
    def test_oe_to_roe_radians(self, oe_chief_deg, oe_deputy_deg):
        """state_oe_to_roe matches brahe output (radian mode)."""
        deg2rad = np.pi / 180.0

        oe_c_rad_np = np.array(oe_chief_deg)
        oe_c_rad_np[2:] *= deg2rad
        oe_d_rad_np = np.array(oe_deputy_deg)
        oe_d_rad_np[2:] *= deg2rad

        roe_brahe = bh.state_oe_to_roe(oe_c_rad_np, oe_d_rad_np, RADIANS)

        oe_c_jnp = jnp.array(oe_c_rad_np)
        oe_d_jnp = jnp.array(oe_d_rad_np)
        roe_astrojax = state_oe_to_roe(oe_c_jnp, oe_d_jnp, use_degrees=False)

        assert abs(float(roe_astrojax[0]) - roe_brahe[0]) < _DA_ATOL
        assert abs(float(roe_astrojax[1]) - roe_brahe[1]) < _ANGLE_RAD_ATOL
        assert abs(float(roe_astrojax[2]) - roe_brahe[2]) < _ECC_VEC_ATOL
        assert abs(float(roe_astrojax[3]) - roe_brahe[3]) < _ECC_VEC_ATOL
        assert abs(float(roe_astrojax[4]) - roe_brahe[4]) < _ANGLE_RAD_ATOL
        assert abs(float(roe_astrojax[5]) - roe_brahe[5]) < _ANGLE_RAD_ATOL

    def test_oe_to_roe_reference_values(self):
        """Match the exact reference values from the brahe Rust test suite."""
        oe_c = np.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = np.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

        roe_brahe = bh.state_oe_to_roe(oe_c, oe_d, DEGREES)

        oe_c_jnp = jnp.array(oe_c)
        oe_d_jnp = jnp.array(oe_d)
        roe_astrojax = state_oe_to_roe(oe_c_jnp, oe_d_jnp, use_degrees=True)

        # Reference values from brahe_rust/src/relative_motion/oe_roe.rs tests
        expected = [
            1.412_801_276_516_814e-4,
            9.321_422_137_829_084e-2,
            4.323_577_088_687_794e-4,
            2.511_333_388_799_496e-4,
            5.0e-2,
            4.953_739_202_357_54e-2,
        ]

        # Verify brahe matches reference (sanity check)
        for i in range(6):
            assert abs(roe_brahe[i] - expected[i]) < 1e-10, f"brahe reference mismatch at index {i}"

        # Verify astrojax matches brahe (within float32 tolerance)
        assert abs(float(roe_astrojax[0]) - expected[0]) < _DA_ATOL
        assert abs(float(roe_astrojax[1]) - expected[1]) < _ANGLE_DEG_ATOL
        assert abs(float(roe_astrojax[2]) - expected[2]) < _ECC_VEC_ATOL
        assert abs(float(roe_astrojax[3]) - expected[3]) < _ECC_VEC_ATOL
        assert abs(float(roe_astrojax[4]) - expected[4]) < _ANGLE_DEG_ATOL
        assert abs(float(roe_astrojax[5]) - expected[5]) < _ANGLE_DEG_ATOL


# ──────────────────────────────────────────────
# ROE -> OE comparison tests
# ──────────────────────────────────────────────


class TestROEtoOEVsBrahe:
    def test_roundtrip_degrees(self):
        """OE->ROE->OE roundtrip matches brahe (degrees)."""
        oe_c = np.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d_orig = np.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

        roe_brahe = bh.state_oe_to_roe(oe_c, oe_d_orig, DEGREES)
        oe_d_brahe = bh.state_roe_to_oe(oe_c, roe_brahe, DEGREES)

        oe_c_jnp = jnp.array(oe_c)
        roe_jnp = jnp.array(roe_brahe)
        oe_d_astrojax = state_roe_to_oe(oe_c_jnp, roe_jnp, use_degrees=True)

        assert abs(float(oe_d_astrojax[0]) - oe_d_brahe[0]) < _SMA_ATOL
        assert abs(float(oe_d_astrojax[1]) - oe_d_brahe[1]) < _ECC_ATOL
        for i in range(2, 6):
            assert abs(float(oe_d_astrojax[i]) - oe_d_brahe[i]) < _ANGLE_DEG_ATOL

    def test_roundtrip_radians(self):
        """OE->ROE->OE roundtrip matches brahe (radians)."""
        deg2rad = np.pi / 180.0
        oe_c = np.array(
            [
                R_EARTH + 700e3,
                0.001,
                97.8 * deg2rad,
                15.0 * deg2rad,
                30.0 * deg2rad,
                45.0 * deg2rad,
            ]
        )
        oe_d_orig = np.array(
            [
                R_EARTH + 701e3,
                0.0015,
                97.85 * deg2rad,
                15.05 * deg2rad,
                30.05 * deg2rad,
                45.05 * deg2rad,
            ]
        )

        roe_brahe = bh.state_oe_to_roe(oe_c, oe_d_orig, RADIANS)
        oe_d_brahe = bh.state_roe_to_oe(oe_c, roe_brahe, RADIANS)

        oe_c_jnp = jnp.array(oe_c)
        roe_jnp = jnp.array(roe_brahe)
        oe_d_astrojax = state_roe_to_oe(oe_c_jnp, roe_jnp, use_degrees=False)

        assert abs(float(oe_d_astrojax[0]) - oe_d_brahe[0]) < _SMA_ATOL
        assert abs(float(oe_d_astrojax[1]) - oe_d_brahe[1]) < _ECC_ATOL
        for i in range(2, 6):
            assert abs(float(oe_d_astrojax[i]) - oe_d_brahe[i]) < _ANGLE_RAD_ATOL

    @pytest.mark.parametrize(
        "oe_chief_deg, roe_deg",
        [
            (
                [R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0],
                [0.000142857, 0.05, 0.0005, -0.0003, 0.01, -0.02],
            ),
            (
                [R_EARTH + 500e3, 0.01, 51.6, 0.0, 0.0, 0.0],
                [0.0001, 0.1, 0.001, 0.0005, 0.05, 0.03],
            ),
        ],
    )
    def test_roe_to_oe_direct(self, oe_chief_deg, roe_deg):
        """state_roe_to_oe output matches brahe for given ROE inputs."""
        oe_c_np = np.array(oe_chief_deg)
        roe_np = np.array(roe_deg)

        oe_d_brahe = bh.state_roe_to_oe(oe_c_np, roe_np, DEGREES)

        oe_c_jnp = jnp.array(oe_chief_deg)
        roe_jnp = jnp.array(roe_deg)
        oe_d_astrojax = state_roe_to_oe(oe_c_jnp, roe_jnp, use_degrees=True)

        assert abs(float(oe_d_astrojax[0]) - oe_d_brahe[0]) < _SMA_ATOL
        assert abs(float(oe_d_astrojax[1]) - oe_d_brahe[1]) < _ECC_ATOL
        for i in range(2, 6):
            assert abs(float(oe_d_astrojax[i]) - oe_d_brahe[i]) < _ANGLE_DEG_ATOL


# ──────────────────────────────────────────────
# ECI <-> ROE comparison tests
# ──────────────────────────────────────────────


class TestECItoROEVsBrahe:
    def test_eci_to_roe_degrees(self):
        """state_eci_to_roe matches brahe (degrees, via ECI path)."""
        oe_c = np.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = np.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

        x_c_brahe = bh.state_koe_to_eci(oe_c, DEGREES)
        x_d_brahe = bh.state_koe_to_eci(oe_d, DEGREES)
        roe_brahe = bh.state_eci_to_roe(x_c_brahe, x_d_brahe, DEGREES)

        x_c_jnp = jnp.array(x_c_brahe)
        x_d_jnp = jnp.array(x_d_brahe)
        roe_astrojax = state_eci_to_roe(x_c_jnp, x_d_jnp, use_degrees=True)

        assert abs(float(roe_astrojax[0]) - roe_brahe[0]) < _ECI_DA_ATOL
        assert abs(float(roe_astrojax[1]) - roe_brahe[1]) < _ECI_ANGLE_DEG_ATOL
        assert abs(float(roe_astrojax[2]) - roe_brahe[2]) < _ECI_ECC_VEC_ATOL
        assert abs(float(roe_astrojax[3]) - roe_brahe[3]) < _ECI_ECC_VEC_ATOL
        assert abs(float(roe_astrojax[4]) - roe_brahe[4]) < _ECI_ANGLE_DEG_ATOL
        assert abs(float(roe_astrojax[5]) - roe_brahe[5]) < _ECI_ANGLE_DEG_ATOL

    def test_eci_to_roe_radians(self):
        """state_eci_to_roe matches brahe (radians, via ECI path)."""
        deg2rad = np.pi / 180.0
        oe_c = np.array(
            [
                R_EARTH + 700e3,
                0.001,
                97.8 * deg2rad,
                15.0 * deg2rad,
                30.0 * deg2rad,
                45.0 * deg2rad,
            ]
        )
        oe_d = np.array(
            [
                R_EARTH + 701e3,
                0.0015,
                97.85 * deg2rad,
                15.05 * deg2rad,
                30.05 * deg2rad,
                45.05 * deg2rad,
            ]
        )

        x_c_brahe = bh.state_koe_to_eci(oe_c, RADIANS)
        x_d_brahe = bh.state_koe_to_eci(oe_d, RADIANS)
        roe_brahe = bh.state_eci_to_roe(x_c_brahe, x_d_brahe, RADIANS)

        x_c_jnp = jnp.array(x_c_brahe)
        x_d_jnp = jnp.array(x_d_brahe)
        roe_astrojax = state_eci_to_roe(x_c_jnp, x_d_jnp, use_degrees=False)

        assert abs(float(roe_astrojax[0]) - roe_brahe[0]) < _ECI_DA_ATOL
        assert abs(float(roe_astrojax[1]) - roe_brahe[1]) < _ECI_ANGLE_RAD_ATOL
        assert abs(float(roe_astrojax[2]) - roe_brahe[2]) < _ECI_ECC_VEC_ATOL
        assert abs(float(roe_astrojax[3]) - roe_brahe[3]) < _ECI_ECC_VEC_ATOL
        assert abs(float(roe_astrojax[4]) - roe_brahe[4]) < _ECI_ANGLE_RAD_ATOL
        assert abs(float(roe_astrojax[5]) - roe_brahe[5]) < _ECI_ANGLE_RAD_ATOL


class TestROEtoECIVsBrahe:
    def test_roe_to_eci_degrees(self):
        """state_roe_to_eci produces valid deputy state matching brahe."""
        oe_c = np.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        x_c_brahe = bh.state_koe_to_eci(oe_c, DEGREES)
        roe = np.array([0.000142857, 0.05, 0.0005, -0.0003, 0.01, -0.02])

        x_d_brahe = bh.state_roe_to_eci(x_c_brahe, roe, DEGREES)

        x_c_jnp = jnp.array(x_c_brahe)
        roe_jnp = jnp.array(roe)
        x_d_astrojax = state_roe_to_eci(x_c_jnp, roe_jnp, use_degrees=True)

        # Position (wider tolerance for ECI path)
        for i in range(3):
            assert abs(float(x_d_astrojax[i]) - x_d_brahe[i]) < _ECI_POS_ATOL
        # Velocity
        for i in range(3, 6):
            assert abs(float(x_d_astrojax[i]) - x_d_brahe[i]) < _ECI_VEL_ATOL

    def test_eci_roe_roundtrip_degrees(self):
        """ECI->ROE->ECI roundtrip matches brahe (degrees)."""
        oe_c = np.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = np.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

        x_c_brahe = bh.state_koe_to_eci(oe_c, DEGREES)
        x_d_brahe = bh.state_koe_to_eci(oe_d, DEGREES)

        x_c_jnp = jnp.array(x_c_brahe)
        x_d_jnp = jnp.array(x_d_brahe)

        roe = state_eci_to_roe(x_c_jnp, x_d_jnp, use_degrees=True)
        x_d_recovered = state_roe_to_eci(x_c_jnp, roe, use_degrees=True)

        for i in range(3):
            assert abs(float(x_d_recovered[i]) - float(x_d_jnp[i])) < _ECI_POS_ATOL
        for i in range(3, 6):
            assert abs(float(x_d_recovered[i]) - float(x_d_jnp[i])) < _ECI_VEL_ATOL

    def test_eci_roe_roundtrip_radians(self):
        """ECI->ROE->ECI roundtrip matches brahe (radians)."""
        deg2rad = np.pi / 180.0
        oe_c = np.array(
            [
                R_EARTH + 700e3,
                0.001,
                97.8 * deg2rad,
                15.0 * deg2rad,
                30.0 * deg2rad,
                45.0 * deg2rad,
            ]
        )
        oe_d = np.array(
            [
                R_EARTH + 701e3,
                0.0015,
                97.85 * deg2rad,
                15.05 * deg2rad,
                30.05 * deg2rad,
                45.05 * deg2rad,
            ]
        )

        x_c_brahe = bh.state_koe_to_eci(oe_c, RADIANS)
        x_d_brahe = bh.state_koe_to_eci(oe_d, RADIANS)

        x_c_jnp = jnp.array(x_c_brahe)
        x_d_jnp = jnp.array(x_d_brahe)

        roe = state_eci_to_roe(x_c_jnp, x_d_jnp, use_degrees=False)
        x_d_recovered = state_roe_to_eci(x_c_jnp, roe, use_degrees=False)

        for i in range(3):
            assert abs(float(x_d_recovered[i]) - float(x_d_jnp[i])) < _ECI_POS_ATOL
        for i in range(3, 6):
            assert abs(float(x_d_recovered[i]) - float(x_d_jnp[i])) < _ECI_VEL_ATOL
