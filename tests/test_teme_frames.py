"""Tests for TEME frame transformations."""

import jax
import jax.numpy as jnp

from astrojax.config import set_dtype
from astrojax.constants import OMEGA_EARTH
from astrojax.eop import zero_eop
from astrojax.epoch import Epoch
from astrojax.frames import (
    rotation_itrf_to_teme,
    rotation_pef_to_teme,
    rotation_teme_to_itrf,
    rotation_teme_to_pef,
    state_gcrf_to_teme,
    state_itrf_to_teme,
    state_pef_to_teme,
    state_teme_to_gcrf,
    state_teme_to_itrf,
    state_teme_to_pef,
)

jax.config.update("jax_enable_x64", True)
set_dtype(jnp.float64)


# Test epoch: 2008-09-20T12:25:40Z (ISS TLE epoch)
TEST_EPOCH = Epoch(2008, 9, 20, 12, 25, 40.0)

# LEO-like state in TEME (m, m/s)
LEO_STATE_TEME = jnp.array(
    [
        6.525e6,
        1.710e6,
        1.416e5,  # position [m]
        -1.713e3,
        6.874e3,
        2.396e3,  # velocity [m/s]
    ]
)


class TestRotationTEMEToPEF:
    """Test TEME -> PEF rotation matrices."""

    def test_is_orthogonal(self) -> None:
        """Rotation matrix should be orthogonal (R @ R.T = I)."""
        R = rotation_teme_to_pef(TEST_EPOCH)
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-14)
        assert jnp.allclose(R.T @ R, jnp.eye(3), atol=1e-14)

    def test_determinant_is_one(self) -> None:
        """Rotation matrix should have determinant +1."""
        R = rotation_teme_to_pef(TEST_EPOCH)
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-14)

    def test_inverse_is_transpose(self) -> None:
        """PEF -> TEME should be the transpose of TEME -> PEF."""
        R_fwd = rotation_teme_to_pef(TEST_EPOCH)
        R_inv = rotation_pef_to_teme(TEST_EPOCH)
        assert jnp.allclose(R_fwd.T, R_inv, atol=1e-14)

    def test_changes_at_different_epochs(self) -> None:
        """Rotation should change with epoch (Earth rotates)."""
        R1 = rotation_teme_to_pef(Epoch(2024, 1, 1, 0, 0, 0.0))
        R2 = rotation_teme_to_pef(Epoch(2024, 1, 1, 6, 0, 0.0))
        assert not jnp.allclose(R1, R2, atol=1e-6)


class TestRotationTEMEToITRF:
    """Test TEME -> ITRF rotation matrices."""

    def test_is_orthogonal(self) -> None:
        eop = zero_eop()
        R = rotation_teme_to_itrf(eop, TEST_EPOCH)
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-14)

    def test_determinant_is_one(self) -> None:
        eop = zero_eop()
        R = rotation_teme_to_itrf(eop, TEST_EPOCH)
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-14)

    def test_inverse_is_transpose(self) -> None:
        eop = zero_eop()
        R_fwd = rotation_teme_to_itrf(eop, TEST_EPOCH)
        R_inv = rotation_itrf_to_teme(eop, TEST_EPOCH)
        assert jnp.allclose(R_fwd.T, R_inv, atol=1e-14)

    def test_zero_eop_equals_pef(self) -> None:
        """With zero EOP, polar motion is identity, so ITRF == PEF."""
        eop = zero_eop()
        R_itrf = rotation_teme_to_itrf(eop, TEST_EPOCH)
        R_pef = rotation_teme_to_pef(TEST_EPOCH)
        assert jnp.allclose(R_itrf, R_pef, atol=1e-10)


class TestStateTEMEToPEF:
    """Test TEME -> PEF state vector transformation."""

    def test_position_magnitude_preserved(self) -> None:
        """Position magnitude should be unchanged by rotation."""
        x_pef = state_teme_to_pef(TEST_EPOCH, LEO_STATE_TEME)
        r_teme = jnp.linalg.norm(LEO_STATE_TEME[:3])
        r_pef = jnp.linalg.norm(x_pef[:3])
        assert jnp.allclose(r_teme, r_pef, rtol=1e-14)

    def test_velocity_includes_omega_correction(self) -> None:
        """PEF velocity should differ from simple rotation by omega x r."""
        R = rotation_teme_to_pef(TEST_EPOCH)
        omega = jnp.array([0.0, 0.0, OMEGA_EARTH])

        r_pef = R @ LEO_STATE_TEME[:3]
        v_simple = R @ LEO_STATE_TEME[3:6]
        v_corrected = v_simple - jnp.cross(omega, r_pef)

        x_pef = state_teme_to_pef(TEST_EPOCH, LEO_STATE_TEME)
        assert jnp.allclose(x_pef[3:6], v_corrected, atol=1e-10)

    def test_round_trip_teme_pef_teme(self) -> None:
        """TEME -> PEF -> TEME should recover the original state."""
        x_pef = state_teme_to_pef(TEST_EPOCH, LEO_STATE_TEME)
        x_back = state_pef_to_teme(TEST_EPOCH, x_pef)
        assert jnp.allclose(x_back, LEO_STATE_TEME, atol=1e-6)


class TestStateTEMEToITRF:
    """Test TEME -> ITRF state vector transformation."""

    def test_position_magnitude_preserved(self) -> None:
        eop = zero_eop()
        x_itrf = state_teme_to_itrf(eop, TEST_EPOCH, LEO_STATE_TEME)
        r_teme = jnp.linalg.norm(LEO_STATE_TEME[:3])
        r_itrf = jnp.linalg.norm(x_itrf[:3])
        assert jnp.allclose(r_teme, r_itrf, rtol=1e-14)

    def test_round_trip_teme_itrf_teme(self) -> None:
        eop = zero_eop()
        x_itrf = state_teme_to_itrf(eop, TEST_EPOCH, LEO_STATE_TEME)
        x_back = state_itrf_to_teme(eop, TEST_EPOCH, x_itrf)
        assert jnp.allclose(x_back, LEO_STATE_TEME, atol=1e-6)

    def test_zero_eop_matches_pef(self) -> None:
        """With zero EOP (no polar motion), ITRF result == PEF result."""
        eop = zero_eop()
        x_itrf = state_teme_to_itrf(eop, TEST_EPOCH, LEO_STATE_TEME)
        x_pef = state_teme_to_pef(TEST_EPOCH, LEO_STATE_TEME)
        assert jnp.allclose(x_itrf, x_pef, atol=1e-6)


class TestStateTEMEToGCRF:
    """Test TEME -> GCRF state vector transformation."""

    def test_position_magnitude_preserved(self) -> None:
        eop = zero_eop()
        x_gcrf = state_teme_to_gcrf(eop, TEST_EPOCH, LEO_STATE_TEME)
        r_teme = jnp.linalg.norm(LEO_STATE_TEME[:3])
        r_gcrf = jnp.linalg.norm(x_gcrf[:3])
        # Multi-step chain (TEME->PEF->ITRF->GCRF) loses some precision
        assert jnp.allclose(r_teme, r_gcrf, rtol=1e-6)

    def test_round_trip_teme_gcrf_teme(self) -> None:
        eop = zero_eop()
        x_gcrf = state_teme_to_gcrf(eop, TEST_EPOCH, LEO_STATE_TEME)
        x_back = state_gcrf_to_teme(eop, TEST_EPOCH, x_gcrf)
        assert jnp.allclose(x_back, LEO_STATE_TEME, atol=1e-4)

    def test_gcrf_differs_from_teme(self) -> None:
        """GCRF state should differ from TEME (frames are not identical)."""
        eop = zero_eop()
        x_gcrf = state_teme_to_gcrf(eop, TEST_EPOCH, LEO_STATE_TEME)
        assert not jnp.allclose(x_gcrf[:3], LEO_STATE_TEME[:3], atol=1e3)

    def test_output_shape(self) -> None:
        eop = zero_eop()
        x_gcrf = state_teme_to_gcrf(eop, TEST_EPOCH, LEO_STATE_TEME)
        assert x_gcrf.shape == (6,)


class TestTEMEFramesJIT:
    """Test JIT compatibility of TEME transformations."""

    def test_rotation_teme_to_pef_jit(self) -> None:
        @jax.jit
        def compute(epc):
            return rotation_teme_to_pef(epc)

        R = compute(TEST_EPOCH)
        assert R.shape == (3, 3)
        assert jnp.all(jnp.isfinite(R))

    def test_state_teme_to_itrf_jit(self) -> None:
        eop = zero_eop()

        @jax.jit
        def compute(epc, x):
            return state_teme_to_itrf(eop, epc, x)

        x_itrf = compute(TEST_EPOCH, LEO_STATE_TEME)
        assert x_itrf.shape == (6,)
        assert jnp.all(jnp.isfinite(x_itrf))

    def test_state_teme_to_gcrf_jit(self) -> None:
        eop = zero_eop()

        @jax.jit
        def compute(epc, x):
            return state_teme_to_gcrf(eop, epc, x)

        x_gcrf = compute(TEST_EPOCH, LEO_STATE_TEME)
        assert x_gcrf.shape == (6,)
        assert jnp.all(jnp.isfinite(x_gcrf))
