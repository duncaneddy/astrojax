import jax
import jax.numpy as jnp

from astrojax.constants import GM_EARTH, OMEGA_EARTH, R_EARTH
from astrojax.epoch import Epoch
from astrojax.frames import (
    earth_rotation,
    rotation_ecef_to_eci,
    rotation_eci_to_ecef,
    state_ecef_to_eci,
    state_eci_to_ecef,
)

# Tolerances for float32 arithmetic
_SINGLE_TOL = 1e-5
_POS_ROUNDTRIP_TOL = 1.0  # metres (float32 on ~7e6 m magnitudes)
_VEL_ROUNDTRIP_TOL = 1e-3  # m/s


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
        R = earth_rotation(epc)
        assert R.shape == (3, 3)

    def test_orthonormality(self):
        """R^T R should be the identity matrix."""
        epc = Epoch(2024, 6, 15, 12, 30, 0)
        R = earth_rotation(epc)
        eye = R.T @ R
        assert jnp.allclose(eye, jnp.eye(3), atol=_SINGLE_TOL)

    def test_determinant_positive_one(self):
        """Rotation matrix determinant should be +1 (proper rotation)."""
        epc = Epoch(2024, 3, 20, 6, 0, 0)
        R = earth_rotation(epc)
        assert jnp.abs(jnp.linalg.det(R) - 1.0) < _SINGLE_TOL

    def test_z_axis_invariant(self):
        """Rz leaves the z-axis unchanged: R @ [0,0,1] = [0,0,1]."""
        epc = Epoch(2024, 7, 4, 18, 0, 0)
        R = earth_rotation(epc)
        z_hat = jnp.array([0.0, 0.0, 1.0])
        result = R @ z_hat
        assert jnp.allclose(result, z_hat, atol=_SINGLE_TOL)

    def test_rotation_eci_to_ecef_equals_earth_rotation(self):
        """rotation_eci_to_ecef returns the same matrix as earth_rotation."""
        epc = Epoch(2024, 1, 1)
        R1 = earth_rotation(epc)
        R2 = rotation_eci_to_ecef(epc)
        assert jnp.allclose(R1, R2, atol=_SINGLE_TOL)

    def test_rotation_ecef_to_eci_is_transpose(self):
        """rotation_ecef_to_eci should be the transpose of rotation_eci_to_ecef."""
        epc = Epoch(2024, 9, 21, 3, 45, 0)
        R_fwd = rotation_eci_to_ecef(epc)
        R_inv = rotation_ecef_to_eci(epc)
        assert jnp.allclose(R_inv, R_fwd.T, atol=_SINGLE_TOL)


# ──────────────────────────────────────────────
# State transformation correctness
# ──────────────────────────────────────────────


class TestStateTransformation:
    def test_roundtrip_equatorial(self):
        """ECI -> ECEF -> ECI roundtrip preserves state (equatorial LEO)."""
        epc = Epoch(2024, 1, 1, 12, 0, 0)
        x_eci = _leo_eci_state()

        x_ecef = state_eci_to_ecef(epc, x_eci)
        x_back = state_ecef_to_eci(epc, x_ecef)

        assert jnp.allclose(x_back[:3], x_eci[:3], atol=_POS_ROUNDTRIP_TOL)
        assert jnp.allclose(x_back[3:6], x_eci[3:6], atol=_VEL_ROUNDTRIP_TOL)

    def test_roundtrip_inclined(self):
        """ECI -> ECEF -> ECI roundtrip preserves state (inclined orbit)."""
        epc = Epoch(2024, 6, 15, 8, 30, 0)
        x_eci = _inclined_eci_state(inc_deg=52.0)

        x_ecef = state_eci_to_ecef(epc, x_eci)
        x_back = state_ecef_to_eci(epc, x_ecef)

        assert jnp.allclose(x_back[:3], x_eci[:3], atol=_POS_ROUNDTRIP_TOL)
        assert jnp.allclose(x_back[3:6], x_eci[3:6], atol=_VEL_ROUNDTRIP_TOL)

    def test_stationary_ecef_has_eci_velocity(self):
        """A point stationary in ECEF should have ECI velocity ≈ ω × r."""
        epc = Epoch(2024, 1, 1)
        # Stationary in ECEF at the equator along the x-axis
        r_ecef = jnp.array([R_EARTH, 0.0, 0.0])
        x_ecef = jnp.array([R_EARTH, 0.0, 0.0, 0.0, 0.0, 0.0])

        x_eci = state_ecef_to_eci(epc, x_ecef)

        # Expected ECI velocity = R.T @ (ω × r_ecef)
        omega = jnp.array([0.0, 0.0, OMEGA_EARTH])
        v_expected_ecef = jnp.cross(omega, r_ecef)  # [0, ω*R_EARTH, 0]
        v_speed = jnp.linalg.norm(v_expected_ecef)

        # The ECI velocity magnitude should match ω * R_EARTH ≈ 465 m/s
        v_eci_speed = jnp.linalg.norm(x_eci[3:6])
        assert jnp.abs(v_eci_speed - v_speed) < 1.0  # 1 m/s tolerance

    def test_north_pole_invariant(self):
        """North pole position and velocity are unchanged by the transformation.

        The rotation axis is z, so [0,0,R_EARTH] is invariant under Rz.
        ω × [0,0,z] = 0, so velocity is also invariant.
        """
        epc = Epoch(2024, 1, 1)
        x_eci = jnp.array([0.0, 0.0, R_EARTH, 0.0, 0.0, 100.0])

        x_ecef = state_eci_to_ecef(epc, x_eci)

        # Position: z-axis unchanged by Rz, x/y are zero
        assert jnp.allclose(x_ecef[:3], x_eci[:3], atol=_POS_ROUNDTRIP_TOL)
        # Velocity: ω × [0,0,R_EARTH] = 0, so only Rz acts on v=[0,0,100]
        # Rz leaves z unchanged, x/y are zero -> v_ecef = v_eci
        assert jnp.allclose(x_ecef[3:6], x_eci[3:6], atol=_VEL_ROUNDTRIP_TOL)

    def test_position_magnitude_preserved(self):
        """Rotation preserves the position magnitude."""
        epc = Epoch(2024, 3, 15, 6, 0, 0)
        x_eci = _leo_eci_state()

        x_ecef = state_eci_to_ecef(epc, x_eci)

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
        R_eager = earth_rotation(epc)
        R_jit = jax.jit(earth_rotation)(epc)
        assert jnp.allclose(R_eager, R_jit, atol=_SINGLE_TOL)

    def test_jit_rotation_eci_to_ecef(self):
        """rotation_eci_to_ecef is JIT-compilable."""
        epc = Epoch(2024, 1, 1)
        R_eager = rotation_eci_to_ecef(epc)
        R_jit = jax.jit(rotation_eci_to_ecef)(epc)
        assert jnp.allclose(R_eager, R_jit, atol=_SINGLE_TOL)

    def test_jit_rotation_ecef_to_eci(self):
        """rotation_ecef_to_eci is JIT-compilable."""
        epc = Epoch(2024, 1, 1)
        R_eager = rotation_ecef_to_eci(epc)
        R_jit = jax.jit(rotation_ecef_to_eci)(epc)
        assert jnp.allclose(R_eager, R_jit, atol=_SINGLE_TOL)

    def test_jit_state_eci_to_ecef(self):
        """state_eci_to_ecef is JIT-compilable."""
        epc = Epoch(2024, 1, 1)
        x = _leo_eci_state()
        y_eager = state_eci_to_ecef(epc, x)
        y_jit = jax.jit(state_eci_to_ecef)(epc, x)
        assert jnp.allclose(y_eager, y_jit, atol=_SINGLE_TOL)

    def test_jit_state_ecef_to_eci(self):
        """state_ecef_to_eci is JIT-compilable."""
        epc = Epoch(2024, 1, 1)
        x_ecef = jnp.array([R_EARTH, 0.0, 0.0, 0.0, 0.0, 0.0])
        y_eager = state_ecef_to_eci(epc, x_ecef)
        y_jit = jax.jit(state_ecef_to_eci)(epc, x_ecef)
        assert jnp.allclose(y_eager, y_jit, atol=_SINGLE_TOL)

    def test_vmap_state_eci_to_ecef(self):
        """state_eci_to_ecef works with vmap over batched states."""
        epc = Epoch(2024, 1, 1)
        states = jnp.stack([
            _leo_eci_state(R_EARTH + 500e3),
            _leo_eci_state(R_EARTH + 700e3),
            _inclined_eci_state(),
        ])
        batched = jax.vmap(state_eci_to_ecef, in_axes=(None, 0))(epc, states)
        assert batched.shape == (3, 6)

    def test_vmap_state_ecef_to_eci(self):
        """state_ecef_to_eci works with vmap over batched states."""
        epc = Epoch(2024, 1, 1)
        # Create ECEF states by transforming ECI states
        eci_states = jnp.stack([
            _leo_eci_state(R_EARTH + 500e3),
            _leo_eci_state(R_EARTH + 700e3),
        ])
        ecef_states = jax.vmap(state_eci_to_ecef, in_axes=(None, 0))(
            epc, eci_states
        )
        eci_back = jax.vmap(state_ecef_to_eci, in_axes=(None, 0))(
            epc, ecef_states
        )
        assert eci_back.shape == (2, 6)
        assert jnp.allclose(eci_back[:, :3], eci_states[:, :3],
                            atol=_POS_ROUNDTRIP_TOL)
