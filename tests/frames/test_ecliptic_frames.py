"""Tests for ecliptic-ICRF frame transformations."""

import jax
import jax.numpy as jnp

from astrojax.constants import AS2RAD, OBLIQUITY_J2000
from astrojax.frames import (
    rotation_ecliptic_to_icrf,
    rotation_icrf_to_ecliptic,
    state_ecliptic_to_icrf,
    state_icrf_to_ecliptic,
)

# Reference obliquity in radians
_EPS = OBLIQUITY_J2000 * AS2RAD


# ---------------------------------------------------------------------------
# Rotation matrix properties
# ---------------------------------------------------------------------------


class TestRotationMatrixProperties:
    """Test that rotation matrices satisfy basic matrix properties."""

    def test_ecliptic_to_icrf_shape(self):
        R = rotation_ecliptic_to_icrf()
        assert R.shape == (3, 3)

    def test_icrf_to_ecliptic_shape(self):
        R = rotation_icrf_to_ecliptic()
        assert R.shape == (3, 3)

    def test_ecliptic_to_icrf_orthogonality(self):
        R = rotation_ecliptic_to_icrf()
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-7)

    def test_icrf_to_ecliptic_orthogonality(self):
        R = rotation_icrf_to_ecliptic()
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-7)

    def test_ecliptic_to_icrf_determinant(self):
        R = rotation_ecliptic_to_icrf()
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-7)

    def test_icrf_to_ecliptic_determinant(self):
        R = rotation_icrf_to_ecliptic()
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-7)

    def test_matrices_are_transposes(self):
        R_ecl2icrf = rotation_ecliptic_to_icrf()
        R_icrf2ecl = rotation_icrf_to_ecliptic()
        assert jnp.allclose(R_ecl2icrf, R_icrf2ecl.T, atol=1e-7)

    def test_inverse_composition_is_identity(self):
        R_ecl2icrf = rotation_ecliptic_to_icrf()
        R_icrf2ecl = rotation_icrf_to_ecliptic()
        assert jnp.allclose(R_ecl2icrf @ R_icrf2ecl, jnp.eye(3), atol=1e-7)


# ---------------------------------------------------------------------------
# Rotation matrix element-by-element verification
# ---------------------------------------------------------------------------


class TestRotationMatrixElements:
    """Verify rotation matrix elements against the explicit Rx formula."""

    def test_ecliptic_to_icrf_elements(self):
        """Rx(-ε) should have specific element values."""
        R = rotation_ecliptic_to_icrf()
        c = jnp.cos(_EPS)
        s = jnp.sin(_EPS)

        # Rx(-ε) with SOFA Rx convention: [[1,0,0],[0,cos(-ε),sin(-ε)],[0,-sin(-ε),cos(-ε)]]
        # = [[1,0,0],[0,cos(ε),-sin(ε)],[0,sin(ε),cos(ε)]]
        expected = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ]
        )
        assert jnp.allclose(R, expected, atol=1e-7)

    def test_icrf_to_ecliptic_elements(self):
        """Rx(ε) should have specific element values."""
        R = rotation_icrf_to_ecliptic()
        c = jnp.cos(_EPS)
        s = jnp.sin(_EPS)

        # Rx(ε) with SOFA Rx convention: [[1,0,0],[0,cos(ε),sin(ε)],[0,-sin(ε),cos(ε)]]
        expected = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, s],
                [0.0, -s, c],
            ]
        )
        assert jnp.allclose(R, expected, atol=1e-7)

    def test_x_axis_invariant(self):
        """Rotation about x-axis should leave x-axis vector unchanged."""
        R = rotation_ecliptic_to_icrf()
        x = jnp.array([1.0, 0.0, 0.0])
        assert jnp.allclose(R @ x, x, atol=1e-7)


# ---------------------------------------------------------------------------
# Known-value tests
# ---------------------------------------------------------------------------


class TestKnownValues:
    """Test transformations of specific vectors with known results."""

    def test_ecliptic_z_to_icrf(self):
        """Pure ecliptic-z [0,0,1] should map to [0, -sin(ε), cos(ε)] in ICRF."""
        R = rotation_ecliptic_to_icrf()
        z_ecl = jnp.array([0.0, 0.0, 1.0])
        result = R @ z_ecl
        expected = jnp.array([0.0, -jnp.sin(_EPS), jnp.cos(_EPS)])
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_icrf_z_to_ecliptic(self):
        """Pure ICRF-z [0,0,1] should map to [0, sin(ε), cos(ε)] in ecliptic."""
        R = rotation_icrf_to_ecliptic()
        z_icrf = jnp.array([0.0, 0.0, 1.0])
        result = R @ z_icrf
        expected = jnp.array([0.0, jnp.sin(_EPS), jnp.cos(_EPS)])
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_ecliptic_y_to_icrf(self):
        """Pure ecliptic-y [0,1,0] should map to [0, cos(ε), sin(ε)] in ICRF."""
        R = rotation_ecliptic_to_icrf()
        y_ecl = jnp.array([0.0, 1.0, 0.0])
        result = R @ y_ecl
        expected = jnp.array([0.0, jnp.cos(_EPS), jnp.sin(_EPS)])
        assert jnp.allclose(result, expected, atol=1e-7)

    def test_norm_preserved(self):
        """Rotation should preserve vector norm."""
        R = rotation_ecliptic_to_icrf()
        v = jnp.array([3.0, 4.0, 5.0])
        assert jnp.allclose(jnp.linalg.norm(R @ v), jnp.linalg.norm(v), atol=1e-6)


# ---------------------------------------------------------------------------
# State vector transformations
# ---------------------------------------------------------------------------


class TestStateTransformations:
    """Test 6-element state vector transformations."""

    def test_roundtrip_ecl_icrf_ecl(self):
        """ecl -> icrf -> ecl should recover the original state."""
        x_ecl = jnp.array([7000e3, 1000e3, 500e3, 1.0e3, 7.5e3, 0.5e3])
        x_icrf = state_ecliptic_to_icrf(x_ecl)
        x_back = state_icrf_to_ecliptic(x_icrf)
        assert jnp.allclose(x_back, x_ecl, atol=1e-3)

    def test_roundtrip_icrf_ecl_icrf(self):
        """icrf -> ecl -> icrf should recover the original state."""
        x_icrf = jnp.array([7000e3, 1000e3, 500e3, 1.0e3, 7.5e3, 0.5e3])
        x_ecl = state_icrf_to_ecliptic(x_icrf)
        x_back = state_ecliptic_to_icrf(x_ecl)
        assert jnp.allclose(x_back, x_icrf, atol=1e-3)

    def test_velocity_rotated_same_as_position(self):
        """Velocity should be rotated identically to position (no Coriolis)."""
        R = rotation_ecliptic_to_icrf()
        x_ecl = jnp.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        x_icrf = state_ecliptic_to_icrf(x_ecl)

        # Position and velocity transformed with same rotation
        r_expected = R @ x_ecl[:3]
        v_expected = R @ x_ecl[3:6]

        assert jnp.allclose(x_icrf[:3], r_expected, atol=1e-3)
        assert jnp.allclose(x_icrf[3:6], v_expected, atol=1e-3)

    def test_state_output_shape(self):
        x_ecl = jnp.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        x_icrf = state_ecliptic_to_icrf(x_ecl)
        assert x_icrf.shape == (6,)

    def test_state_norm_preserved(self):
        """Position and velocity norms should be preserved."""
        x_ecl = jnp.array([7000e3, 1000e3, 500e3, 1.0e3, 7.5e3, 0.5e3])
        x_icrf = state_ecliptic_to_icrf(x_ecl)
        assert jnp.allclose(
            jnp.linalg.norm(x_ecl[:3]),
            jnp.linalg.norm(x_icrf[:3]),
            atol=1e-3,
        )
        assert jnp.allclose(
            jnp.linalg.norm(x_ecl[3:6]),
            jnp.linalg.norm(x_icrf[3:6]),
            atol=1e-3,
        )


# ---------------------------------------------------------------------------
# JAX compatibility
# ---------------------------------------------------------------------------


class TestJAXCompatibility:
    """Test JIT compilation and JAX tracing."""

    def test_rotation_ecliptic_to_icrf_jit(self):
        R = jax.jit(rotation_ecliptic_to_icrf)()
        assert R.shape == (3, 3)

    def test_rotation_icrf_to_ecliptic_jit(self):
        R = jax.jit(rotation_icrf_to_ecliptic)()
        assert R.shape == (3, 3)

    def test_state_ecliptic_to_icrf_jit(self):
        jit_fn = jax.jit(state_ecliptic_to_icrf)
        x_ecl = jnp.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        x_icrf = jit_fn(x_ecl)
        assert x_icrf.shape == (6,)

    def test_state_icrf_to_ecliptic_jit(self):
        jit_fn = jax.jit(state_icrf_to_ecliptic)
        x_icrf = jnp.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
        x_ecl = jit_fn(x_icrf)
        assert x_ecl.shape == (6,)
