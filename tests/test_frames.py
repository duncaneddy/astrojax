"""Tests for ECI-ECEF frame transformations (basic properties).

These tests verify rotation matrix properties and state transformation
correctness using the full IAU 2006/2000A model with zero EOP data.
"""

import jax
import jax.numpy as jnp

from astrojax.constants import GM_EARTH, OMEGA_EARTH, R_EARTH
from astrojax.eop import zero_eop
from astrojax.epoch import Epoch
from astrojax.frames import (
    earth_rotation,
    rotation_ecef_to_eci,
    rotation_eci_to_ecef,
    state_ecef_to_eci,
    state_eci_to_ecef,
)

# Tolerances for float64 arithmetic
_SINGLE_TOL = 1e-10
_POS_ROUNDTRIP_TOL = 1e-6  # metres
_VEL_ROUNDTRIP_TOL = 1e-6  # m/s

# Use zero EOP for these basic property tests
_EOP = zero_eop()


def _leo_eci_state(sma=R_EARTH + 500e3):
    """Create a circular equatorial LEO orbit state in ECI."""
    v_circ = jnp.sqrt(GM_EARTH / sma)
    return jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])


def _inclined_eci_state(sma=R_EARTH + 500e3, inc_deg=45.0):
    """Create a circular inclined orbit state in ECI."""
    v_circ = float(jnp.sqrt(GM_EARTH / sma))
    inc = jnp.deg2rad(inc_deg)
    vy = v_circ * jnp.cos(inc)
    vz = v_circ * jnp.sin(inc)
    return jnp.array([sma, 0.0, 0.0, 0.0, vy, vz])


# ──────────────────────────────────────────────
# Rotation matrix properties
# ──────────────────────────────────────────────


class TestRotationMatrix:
    def test_shape(self):
        """Earth rotation matrix has shape (3, 3)."""
        epc = Epoch(2024, 1, 1)
        R = earth_rotation(_EOP, epc)
        assert R.shape == (3, 3)

    def test_orthonormality(self):
        """R^T R should be the identity matrix."""
        epc = Epoch(2024, 6, 15, 12, 30, 0)
        R = earth_rotation(_EOP, epc)
        eye = R.T @ R
        assert jnp.allclose(eye, jnp.eye(3), atol=_SINGLE_TOL)

    def test_determinant_positive_one(self):
        """Rotation matrix determinant should be +1 (proper rotation)."""
        epc = Epoch(2024, 3, 20, 6, 0, 0)
        R = earth_rotation(_EOP, epc)
        assert jnp.abs(jnp.linalg.det(R) - 1.0) < _SINGLE_TOL

    def test_rotation_eci_to_ecef_equals_earth_rotation(self):
        """rotation_eci_to_ecef returns the same matrix as earth_rotation."""
        epc = Epoch(2024, 1, 1)
        R1 = earth_rotation(_EOP, epc)
        R2 = rotation_eci_to_ecef(_EOP, epc)
        assert jnp.allclose(R1, R2, atol=_SINGLE_TOL)

    def test_rotation_ecef_to_eci_is_transpose(self):
        """rotation_ecef_to_eci should be the transpose of rotation_eci_to_ecef."""
        epc = Epoch(2024, 9, 21, 3, 45, 0)
        R_fwd = rotation_eci_to_ecef(_EOP, epc)
        R_inv = rotation_ecef_to_eci(_EOP, epc)
        assert jnp.allclose(R_inv, R_fwd.T, atol=_SINGLE_TOL)


# ──────────────────────────────────────────────
# State transformation correctness
# ──────────────────────────────────────────────


class TestStateTransformation:
    def test_roundtrip_equatorial(self):
        """ECI -> ECEF -> ECI roundtrip preserves state (equatorial LEO)."""
        epc = Epoch(2024, 1, 1, 12, 0, 0)
        x_eci = _leo_eci_state()

        x_ecef = state_eci_to_ecef(_EOP, epc, x_eci)
        x_back = state_ecef_to_eci(_EOP, epc, x_ecef)

        assert jnp.allclose(x_back[:3], x_eci[:3], atol=_POS_ROUNDTRIP_TOL)
        assert jnp.allclose(x_back[3:6], x_eci[3:6], atol=_VEL_ROUNDTRIP_TOL)

    def test_roundtrip_inclined(self):
        """ECI -> ECEF -> ECI roundtrip preserves state (inclined orbit)."""
        epc = Epoch(2024, 6, 15, 8, 30, 0)
        x_eci = _inclined_eci_state(inc_deg=52.0)

        x_ecef = state_eci_to_ecef(_EOP, epc, x_eci)
        x_back = state_ecef_to_eci(_EOP, epc, x_ecef)

        assert jnp.allclose(x_back[:3], x_eci[:3], atol=_POS_ROUNDTRIP_TOL)
        assert jnp.allclose(x_back[3:6], x_eci[3:6], atol=_VEL_ROUNDTRIP_TOL)

    def test_stationary_ecef_has_eci_velocity(self):
        """A point stationary in ECEF should have ECI velocity ~ omega x r."""
        epc = Epoch(2024, 1, 1)
        x_ecef = jnp.array([R_EARTH, 0.0, 0.0, 0.0, 0.0, 0.0])

        x_eci = state_ecef_to_eci(_EOP, epc, x_ecef)

        # Expected ECI velocity magnitude ~ omega * R_EARTH ~ 465 m/s
        omega = jnp.array([0.0, 0.0, OMEGA_EARTH])
        r_ecef = jnp.array([R_EARTH, 0.0, 0.0])
        v_expected = jnp.linalg.norm(jnp.cross(omega, r_ecef))
        v_eci_speed = jnp.linalg.norm(x_eci[3:6])
        assert jnp.abs(v_eci_speed - v_expected) < 1.0  # 1 m/s tolerance

    def test_position_magnitude_preserved(self):
        """Rotation preserves the position magnitude."""
        epc = Epoch(2024, 3, 15, 6, 0, 0)
        x_eci = _leo_eci_state()

        x_ecef = state_eci_to_ecef(_EOP, epc, x_eci)

        r_eci_mag = jnp.linalg.norm(x_eci[:3])
        r_ecef_mag = jnp.linalg.norm(x_ecef[:3])
        assert jnp.abs(r_eci_mag - r_ecef_mag) < _POS_ROUNDTRIP_TOL


# ──────────────────────────────────────────────
# JAX compatibility tests
# ──────────────────────────────────────────────


class TestJAXCompatibility:
    def test_jit_earth_rotation(self):
        """earth_rotation is JIT-compilable."""
        epc = Epoch(2024, 1, 1)
        R_eager = earth_rotation(_EOP, epc)
        R_jit = jax.jit(earth_rotation)(_EOP, epc)
        assert jnp.allclose(R_eager, R_jit, atol=_SINGLE_TOL)

    def test_jit_rotation_eci_to_ecef(self):
        """rotation_eci_to_ecef is JIT-compilable."""
        epc = Epoch(2024, 1, 1)
        R_eager = rotation_eci_to_ecef(_EOP, epc)
        R_jit = jax.jit(rotation_eci_to_ecef)(_EOP, epc)
        assert jnp.allclose(R_eager, R_jit, atol=_SINGLE_TOL)

    def test_jit_rotation_ecef_to_eci(self):
        """rotation_ecef_to_eci is JIT-compilable."""
        epc = Epoch(2024, 1, 1)
        R_eager = rotation_ecef_to_eci(_EOP, epc)
        R_jit = jax.jit(rotation_ecef_to_eci)(_EOP, epc)
        assert jnp.allclose(R_eager, R_jit, atol=_SINGLE_TOL)

    def test_jit_state_eci_to_ecef(self):
        """state_eci_to_ecef is JIT-compilable."""
        epc = Epoch(2024, 1, 1)
        x = _leo_eci_state()
        y_eager = state_eci_to_ecef(_EOP, epc, x)
        y_jit = jax.jit(state_eci_to_ecef)(_EOP, epc, x)
        assert jnp.allclose(y_eager, y_jit, atol=_SINGLE_TOL)

    def test_jit_state_ecef_to_eci(self):
        """state_ecef_to_eci is JIT-compilable."""
        epc = Epoch(2024, 1, 1)
        x_ecef = jnp.array([R_EARTH, 0.0, 0.0, 0.0, 0.0, 0.0])
        y_eager = state_ecef_to_eci(_EOP, epc, x_ecef)
        y_jit = jax.jit(state_ecef_to_eci)(_EOP, epc, x_ecef)
        assert jnp.allclose(y_eager, y_jit, atol=_SINGLE_TOL)

    def test_vmap_state_eci_to_ecef(self):
        """state_eci_to_ecef works with vmap over batched states."""
        epc = Epoch(2024, 1, 1)
        states = jnp.stack(
            [
                _leo_eci_state(R_EARTH + 500e3),
                _leo_eci_state(R_EARTH + 700e3),
                _inclined_eci_state(),
            ]
        )
        batched = jax.vmap(state_eci_to_ecef, in_axes=(None, None, 0))(_EOP, epc, states)
        assert batched.shape == (3, 6)

    def test_vmap_state_ecef_to_eci(self):
        """state_ecef_to_eci works with vmap over batched states."""
        epc = Epoch(2024, 1, 1)
        eci_states = jnp.stack(
            [
                _leo_eci_state(R_EARTH + 500e3),
                _leo_eci_state(R_EARTH + 700e3),
            ]
        )
        ecef_states = jax.vmap(state_eci_to_ecef, in_axes=(None, None, 0))(_EOP, epc, eci_states)
        eci_back = jax.vmap(state_ecef_to_eci, in_axes=(None, None, 0))(_EOP, epc, ecef_states)
        assert eci_back.shape == (2, 6)
        assert jnp.allclose(eci_back[:, :3], eci_states[:, :3], atol=_POS_ROUNDTRIP_TOL)
