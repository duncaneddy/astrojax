"""Tests for GCRF-ITRF frame transformations (IAU 2006/2000A).

Reference values from IAU SOFA Tools for Earth Attitude, Example 5.5
(Software Version 18, 2021-04-18).

Test date: 2007 April 5, 12:00:00 UTC
"""

import jax
import jax.numpy as jnp

from astrojax.config import set_dtype
from astrojax.constants import AS2RAD, GM_EARTH, R_EARTH
from astrojax.eop import static_eop, zero_eop
from astrojax.epoch import Epoch
from astrojax.frames import (
    bias_precession_nutation,
    earth_rotation,
    earth_rotation_angle,
    rotation_ecef_to_eci,
    rotation_eci_to_ecef,
    rotation_gcrf_to_itrf,
    rotation_itrf_to_gcrf,
    state_ecef_to_eci,
    state_eci_to_ecef,
    state_gcrf_to_itrf,
    state_itrf_to_gcrf,
)

# Use float64 for precision matching against SOFA reference values
set_dtype(jnp.float64)

# SOFA Example 5.5 test fixture
# EOP values from the example
_EOP = static_eop(
    pm_x=0.0349282 * AS2RAD,
    pm_y=0.4833163 * AS2RAD,
    ut1_utc=-0.072073685,
    dX=0.0001750 * AS2RAD * 1e-3,  # milliarcseconds -> radians
    dY=-0.0002259 * AS2RAD * 1e-3,  # milliarcseconds -> radians
)
_EPC = Epoch(2007, 4, 5, 12, 0, 0.0)

# Tolerance for 1e-8 precision matching
_TOL = 1e-8

# Position and velocity tolerances for state roundtrips
_POS_ROUNDTRIP_TOL = 1e-6  # metres
_VEL_ROUNDTRIP_TOL = 1e-6  # m/s


# ──────────────────────────────────────────────
# SOFA Example 5.5 reference values
# ──────────────────────────────────────────────

# BPN matrix (GCRF -> CIRS), SOFA Example 5.5 Step 4
_BPN_REF = jnp.array(
    [
        [+0.999999746339445, -0.000000005138822, -0.000712264729525],
        [-0.000000026475227, +0.999999999014975, -0.000044385242827],
        [+0.000712264729599, +0.000044385250426, +0.999999745354420],
    ]
)

# ER @ BPN product (intermediate step), SOFA Example 5.5 Step 5
_ER_BPN_REF = jnp.array(
    [
        [+0.973104317573127, +0.230363826247709, -0.000703332818845],
        [-0.230363798804182, +0.973104570735574, +0.000120888549586],
        [+0.000712264729599, +0.000044385250426, +0.999999745354420],
    ]
)

# Full rotation GCRF->ITRF (PM @ ER @ BPN), SOFA Example 5.5 Step 7
_GCRF_TO_ITRF_REF = jnp.array(
    [
        [+0.973104317697535, +0.230363826239128, -0.000703163482198],
        [-0.230363800456037, +0.973104570632801, +0.000118545366625],
        [+0.000711560162668, +0.000046626403995, +0.999999745754024],
    ]
)


class TestBiasPrecessionNutation:
    """Test BPN matrix against SOFA Example 5.5."""

    def test_bpn_matrix(self):
        """BPN matrix matches SOFA Example 5.5 to 1e-8."""
        bpn = bias_precession_nutation(_EOP, _EPC)
        assert jnp.allclose(bpn, _BPN_REF, atol=_TOL), (
            f"Max error: {float(jnp.max(jnp.abs(bpn - _BPN_REF)))}"
        )

    def test_bpn_orthogonal(self):
        """BPN matrix is orthogonal."""
        bpn = bias_precession_nutation(_EOP, _EPC)
        eye = bpn.T @ bpn
        assert jnp.allclose(eye, jnp.eye(3), atol=1e-14)

    def test_bpn_determinant(self):
        """BPN matrix has determinant +1."""
        bpn = bias_precession_nutation(_EOP, _EPC)
        det = jnp.linalg.det(bpn)
        assert jnp.abs(det - 1.0) < 1e-14


class TestEarthRotationTimesBPN:
    """Test ER @ BPN product against SOFA Example 5.5."""

    def test_er_bpn_product(self):
        """ER @ BPN matches SOFA Example 5.5 to 1e-8."""
        bpn = bias_precession_nutation(_EOP, _EPC)
        er = earth_rotation_angle(_EOP, _EPC)
        product = er @ bpn
        assert jnp.allclose(product, _ER_BPN_REF, atol=_TOL), (
            f"Max error: {float(jnp.max(jnp.abs(product - _ER_BPN_REF)))}"
        )


class TestRotationGCRFtoITRF:
    """Test full GCRF -> ITRF rotation against SOFA Example 5.5."""

    def test_rotation_gcrf_to_itrf(self):
        """Full GCRF->ITRF rotation matches SOFA Example 5.5 to 1e-8."""
        R = rotation_gcrf_to_itrf(_EOP, _EPC)
        assert jnp.allclose(R, _GCRF_TO_ITRF_REF, atol=_TOL), (
            f"Max error: {float(jnp.max(jnp.abs(R - _GCRF_TO_ITRF_REF)))}"
        )

    def test_rotation_shape(self):
        """Rotation matrix has shape (3, 3)."""
        R = rotation_gcrf_to_itrf(_EOP, _EPC)
        assert R.shape == (3, 3)

    def test_rotation_orthogonal(self):
        """Full rotation matrix is orthogonal."""
        R = rotation_gcrf_to_itrf(_EOP, _EPC)
        eye = R.T @ R
        assert jnp.allclose(eye, jnp.eye(3), atol=1e-14)

    def test_rotation_determinant(self):
        """Full rotation matrix has determinant +1."""
        R = rotation_gcrf_to_itrf(_EOP, _EPC)
        det = jnp.linalg.det(R)
        assert jnp.abs(det - 1.0) < 1e-14


class TestRotationITRFtoGCRF:
    """Test ITRF -> GCRF rotation (transpose)."""

    def test_rotation_itrf_to_gcrf(self):
        """ITRF->GCRF is transpose of GCRF->ITRF."""
        R_fwd = rotation_gcrf_to_itrf(_EOP, _EPC)
        R_inv = rotation_itrf_to_gcrf(_EOP, _EPC)
        assert jnp.allclose(R_inv, R_fwd.T, atol=1e-14)


class TestStateRoundtrip:
    """Test state transformation roundtrip GCRF -> ITRF -> GCRF."""

    def test_roundtrip_equatorial(self):
        """GCRF -> ITRF -> GCRF roundtrip preserves equatorial LEO state."""
        sma = R_EARTH + 500e3
        v_circ = jnp.sqrt(GM_EARTH / sma)
        x_gcrf = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])

        x_itrf = state_gcrf_to_itrf(_EOP, _EPC, x_gcrf)
        x_back = state_itrf_to_gcrf(_EOP, _EPC, x_itrf)

        assert jnp.allclose(x_back[:3], x_gcrf[:3], atol=_POS_ROUNDTRIP_TOL)
        assert jnp.allclose(x_back[3:6], x_gcrf[3:6], atol=_VEL_ROUNDTRIP_TOL)

    def test_roundtrip_inclined(self):
        """GCRF -> ITRF -> GCRF roundtrip preserves inclined orbit state."""
        sma = R_EARTH + 500e3
        v_circ = float(jnp.sqrt(GM_EARTH / sma))
        inc = jnp.deg2rad(52.0)
        x_gcrf = jnp.array([sma, 0.0, 0.0, 0.0, v_circ * jnp.cos(inc), v_circ * jnp.sin(inc)])

        x_itrf = state_gcrf_to_itrf(_EOP, _EPC, x_gcrf)
        x_back = state_itrf_to_gcrf(_EOP, _EPC, x_itrf)

        assert jnp.allclose(x_back[:3], x_gcrf[:3], atol=_POS_ROUNDTRIP_TOL)
        assert jnp.allclose(x_back[3:6], x_gcrf[3:6], atol=_VEL_ROUNDTRIP_TOL)

    def test_position_magnitude_preserved(self):
        """Position magnitude is preserved through rotation."""
        sma = R_EARTH + 500e3
        v_circ = jnp.sqrt(GM_EARTH / sma)
        x_gcrf = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])

        x_itrf = state_gcrf_to_itrf(_EOP, _EPC, x_gcrf)

        r_gcrf = jnp.linalg.norm(x_gcrf[:3])
        r_itrf = jnp.linalg.norm(x_itrf[:3])
        assert jnp.abs(r_gcrf - r_itrf) < _POS_ROUNDTRIP_TOL


class TestECIECEFAliases:
    """Test ECI/ECEF aliases match GCRF/ITRF functions."""

    def test_rotation_eci_ecef_equals_gcrf_itrf(self):
        """rotation_eci_to_ecef is the same as rotation_gcrf_to_itrf."""
        R1 = rotation_eci_to_ecef(_EOP, _EPC)
        R2 = rotation_gcrf_to_itrf(_EOP, _EPC)
        assert jnp.allclose(R1, R2, atol=1e-14)

    def test_rotation_ecef_eci_equals_itrf_gcrf(self):
        """rotation_ecef_to_eci is the same as rotation_itrf_to_gcrf."""
        R1 = rotation_ecef_to_eci(_EOP, _EPC)
        R2 = rotation_itrf_to_gcrf(_EOP, _EPC)
        assert jnp.allclose(R1, R2, atol=1e-14)

    def test_state_eci_ecef_equals_gcrf_itrf(self):
        """state_eci_to_ecef is the same as state_gcrf_to_itrf."""
        x = jnp.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
        x1 = state_eci_to_ecef(_EOP, _EPC, x)
        x2 = state_gcrf_to_itrf(_EOP, _EPC, x)
        assert jnp.allclose(x1, x2, atol=1e-14)

    def test_state_ecef_eci_equals_itrf_gcrf(self):
        """state_ecef_to_eci is the same as state_itrf_to_gcrf."""
        x = jnp.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0])
        x1 = state_ecef_to_eci(_EOP, _EPC, x)
        x2 = state_itrf_to_gcrf(_EOP, _EPC, x)
        assert jnp.allclose(x1, x2, atol=1e-14)

    def test_earth_rotation_equals_gcrf_itrf(self):
        """earth_rotation is the same as rotation_gcrf_to_itrf."""
        R1 = earth_rotation(_EOP, _EPC)
        R2 = rotation_gcrf_to_itrf(_EOP, _EPC)
        assert jnp.allclose(R1, R2, atol=1e-14)


class TestJAXCompatibility:
    """Test JAX JIT and vmap compatibility."""

    def test_jit_rotation_gcrf_to_itrf(self):
        """rotation_gcrf_to_itrf is JIT-compilable."""
        eop = zero_eop()
        epc = Epoch(2024, 1, 1)
        R_eager = rotation_gcrf_to_itrf(eop, epc)
        R_jit = jax.jit(rotation_gcrf_to_itrf)(eop, epc)
        assert jnp.allclose(R_eager, R_jit, atol=1e-12)

    def test_jit_state_gcrf_to_itrf(self):
        """state_gcrf_to_itrf is JIT-compilable."""
        eop = zero_eop()
        epc = Epoch(2024, 1, 1)
        x = jnp.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
        y_eager = state_gcrf_to_itrf(eop, epc, x)
        y_jit = jax.jit(state_gcrf_to_itrf)(eop, epc, x)
        assert jnp.allclose(y_eager, y_jit, atol=1e-12)

    def test_jit_state_itrf_to_gcrf(self):
        """state_itrf_to_gcrf is JIT-compilable."""
        eop = zero_eop()
        epc = Epoch(2024, 1, 1)
        x = jnp.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0])
        y_eager = state_itrf_to_gcrf(eop, epc, x)
        y_jit = jax.jit(state_itrf_to_gcrf)(eop, epc, x)
        assert jnp.allclose(y_eager, y_jit, atol=1e-12)

    def test_vmap_state_gcrf_to_itrf(self):
        """state_gcrf_to_itrf works with vmap over batched states."""
        eop = zero_eop()
        epc = Epoch(2024, 1, 1)
        states = jnp.stack(
            [
                jnp.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 7612.0, 0.0]),
                jnp.array([R_EARTH + 700e3, 0.0, 0.0, 0.0, 7400.0, 0.0]),
            ]
        )
        batched = jax.vmap(state_gcrf_to_itrf, in_axes=(None, None, 0))(eop, epc, states)
        assert batched.shape == (2, 6)
