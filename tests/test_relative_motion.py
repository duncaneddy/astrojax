import jax
import jax.numpy as jnp
import pytest

from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.relative_motion import (
    hcw_derivative,
    rotation_eci_to_rtn,
    rotation_rtn_to_eci,
    state_eci_to_rtn,
    state_rtn_to_eci,
)

# Tolerances for float32 arithmetic
_SINGLE_TOL = 1e-5
_ROUNDTRIP_TOL = 1e-4


def _circular_orbit_state(sma):
    """Create a circular equatorial orbit state [x, y, z, vx, vy, vz]."""
    v_circ = jnp.sqrt(GM_EARTH / sma)
    return jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])


def _inclined_orbit_state(sma, inc_deg):
    """Create a circular orbit state inclined in the XZ plane.

    Rotates the velocity vector by *inc_deg* about the X-axis so the
    orbit plane is tilted, producing a non-axis-aligned test case.
    """
    v_circ = float(jnp.sqrt(GM_EARTH / sma))
    inc = jnp.deg2rad(inc_deg)
    vy = v_circ * jnp.cos(inc)
    vz = v_circ * jnp.sin(inc)
    return jnp.array([sma, 0.0, 0.0, 0.0, vy, vz])


# ──────────────────────────────────────────────
# Rotation matrix properties
# ──────────────────────────────────────────────

class TestRotationMatrix:
    def test_orthonormality(self):
        """R^T R should be the identity matrix."""
        x = _circular_orbit_state(R_EARTH + 500e3)
        R = rotation_rtn_to_eci(x)
        eye = R.T @ R
        assert jnp.allclose(eye, jnp.eye(3), atol=_SINGLE_TOL)

    def test_determinant_positive_one(self):
        """Rotation matrix determinant should be +1 (proper rotation)."""
        x = _circular_orbit_state(R_EARTH + 500e3)
        R = rotation_rtn_to_eci(x)
        assert jnp.abs(jnp.linalg.det(R) - 1.0) < _SINGLE_TOL

    def test_transpose_relationship(self):
        """rotation_eci_to_rtn should be the transpose of rotation_rtn_to_eci."""
        x = _circular_orbit_state(R_EARTH + 700e3)
        R_rtn2eci = rotation_rtn_to_eci(x)
        R_eci2rtn = rotation_eci_to_rtn(x)
        assert jnp.allclose(R_eci2rtn, R_rtn2eci.T, atol=_SINGLE_TOL)

    def test_radial_alignment(self):
        """R_eci2rtn @ r should yield [|r|, 0, 0]."""
        x = _circular_orbit_state(R_EARTH + 500e3)
        r = x[:3]
        R_eci2rtn = rotation_eci_to_rtn(x)
        r_rtn = R_eci2rtn @ r
        r_mag = jnp.linalg.norm(r)
        expected = jnp.array([r_mag, 0.0, 0.0])
        assert jnp.allclose(r_rtn, expected, atol=1.0)  # float32 on ~7e6 m

    def test_inclined_orbit_orthonormality(self):
        """Rotation matrix properties hold for inclined orbits."""
        x = _inclined_orbit_state(R_EARTH + 600e3, 45.0)
        R = rotation_rtn_to_eci(x)
        eye = R.T @ R
        assert jnp.allclose(eye, jnp.eye(3), atol=_SINGLE_TOL)


# ──────────────────────────────────────────────
# State transformation tests
# ──────────────────────────────────────────────

class TestStateTransformation:
    def test_radial_offset(self):
        """A +100m radial offset in ECI maps to ~[100, 0, 0, ...] in RTN."""
        sma = R_EARTH + 500e3
        chief = _circular_orbit_state(sma)
        deputy = chief + jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        rel = state_eci_to_rtn(chief, deputy)
        assert jnp.abs(rel[0] - 100.0) < 1.0
        assert jnp.abs(rel[1]) < 1.0
        assert jnp.abs(rel[2]) < 1.0

    def test_roundtrip_equatorial(self):
        """eci->rtn->eci roundtrip preserves the deputy state (equatorial)."""
        sma = R_EARTH + 500e3
        chief = _circular_orbit_state(sma)
        deputy = chief + jnp.array([100.0, 200.0, 300.0, 0.1, 0.2, 0.3])
        rel_rtn = state_eci_to_rtn(chief, deputy)
        deputy_back = state_rtn_to_eci(chief, rel_rtn)
        assert jnp.allclose(deputy_back, deputy, atol=_ROUNDTRIP_TOL)

    def test_roundtrip_inclined(self):
        """eci->rtn->eci roundtrip preserves the deputy state (inclined orbit)."""
        sma = R_EARTH + 600e3
        chief = _inclined_orbit_state(sma, 52.0)
        deputy = chief + jnp.array([50.0, -80.0, 120.0, 0.05, -0.1, 0.07])
        rel_rtn = state_eci_to_rtn(chief, deputy)
        deputy_back = state_rtn_to_eci(chief, rel_rtn)
        assert jnp.allclose(deputy_back, deputy, atol=_ROUNDTRIP_TOL)

    def test_zero_separation(self):
        """When deputy == chief the relative state should be near-zero."""
        chief = _circular_orbit_state(R_EARTH + 500e3)
        rel = state_eci_to_rtn(chief, chief)
        assert jnp.allclose(rel, jnp.zeros(6), atol=_SINGLE_TOL)

    def test_along_track_offset(self):
        """A +200m along-track (Y) ECI offset maps mostly to RTN T-axis."""
        sma = R_EARTH + 500e3
        chief = _circular_orbit_state(sma)
        deputy = chief + jnp.array([0.0, 200.0, 0.0, 0.0, 0.0, 0.0])
        rel = state_eci_to_rtn(chief, deputy)
        assert jnp.abs(rel[1] - 200.0) < 1.0  # along-track component


# ──────────────────────────────────────────────
# HCW dynamics tests
# ──────────────────────────────────────────────

class TestHCWDynamics:
    @pytest.fixture()
    def n(self):
        """Mean motion for a 500km LEO orbit."""
        sma = R_EARTH + 500e3
        return jnp.sqrt(GM_EARTH / sma**3)

    def test_origin_equilibrium(self, n):
        """Zero state yields zero derivative (origin is an equilibrium)."""
        state = jnp.zeros(6)
        deriv = hcw_derivative(state, n)
        assert jnp.allclose(deriv, jnp.zeros(6), atol=_SINGLE_TOL)

    def test_radial_displacement(self, n):
        """[x,0,0,0,0,0] -> x_ddot = 3 n^2 x (radial restoring)."""
        x_val = 100.0
        state = jnp.array([x_val, 0.0, 0.0, 0.0, 0.0, 0.0])
        deriv = hcw_derivative(state, n)
        expected_xddot = 3.0 * n**2 * x_val
        assert jnp.abs(deriv[3] - expected_xddot) < _SINGLE_TOL

    def test_cross_track_oscillation(self, n):
        """[0,0,z,0,0,0] -> z_ddot = -n^2 z (simple harmonic)."""
        z_val = 50.0
        state = jnp.array([0.0, 0.0, z_val, 0.0, 0.0, 0.0])
        deriv = hcw_derivative(state, n)
        expected_zddot = -(n**2) * z_val
        assert jnp.abs(deriv[5] - expected_zddot) < _SINGLE_TOL

    def test_coriolis_coupling(self, n):
        """[0,0,0,0,ydot,0] -> x_ddot = 2 n ydot (Coriolis)."""
        ydot_val = 1.0
        state = jnp.array([0.0, 0.0, 0.0, 0.0, ydot_val, 0.0])
        deriv = hcw_derivative(state, n)
        expected_xddot = 2.0 * n * ydot_val
        assert jnp.abs(deriv[3] - expected_xddot) < _SINGLE_TOL

    def test_velocity_passthrough(self, n):
        """First three elements of derivative equal input velocities."""
        state = jnp.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
        deriv = hcw_derivative(state, n)
        assert jnp.allclose(deriv[:3], state[3:6], atol=_SINGLE_TOL)


# ──────────────────────────────────────────────
# JAX compatibility tests
# ──────────────────────────────────────────────

class TestJAXCompatibility:
    def test_jit_rotation_rtn_to_eci(self):
        """rotation_rtn_to_eci is JIT-compilable."""
        x = _circular_orbit_state(R_EARTH + 500e3)
        R_eager = rotation_rtn_to_eci(x)
        R_jit = jax.jit(rotation_rtn_to_eci)(x)
        assert jnp.allclose(R_eager, R_jit, atol=_SINGLE_TOL)

    def test_jit_state_eci_to_rtn(self):
        """state_eci_to_rtn is JIT-compilable."""
        chief = _circular_orbit_state(R_EARTH + 500e3)
        deputy = chief + jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        rel_eager = state_eci_to_rtn(chief, deputy)
        rel_jit = jax.jit(state_eci_to_rtn)(chief, deputy)
        assert jnp.allclose(rel_eager, rel_jit, atol=_SINGLE_TOL)

    def test_jit_hcw_derivative(self):
        """hcw_derivative is JIT-compilable."""
        n = jnp.sqrt(GM_EARTH / (R_EARTH + 500e3) ** 3)
        state = jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        d_eager = hcw_derivative(state, n)
        d_jit = jax.jit(hcw_derivative)(state, n)
        assert jnp.allclose(d_eager, d_jit, atol=_SINGLE_TOL)

    def test_vmap_hcw_derivative(self):
        """hcw_derivative works with vmap over a batch of states."""
        n = jnp.sqrt(GM_EARTH / (R_EARTH + 500e3) ** 3)
        states = jnp.array([
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 200.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 50.0, 0.0, 0.0, 0.0],
        ])
        batched = jax.vmap(hcw_derivative, in_axes=(0, None))(states, n)
        assert batched.shape == (3, 6)
        # Spot-check: first row x_ddot = 3 n^2 * 100
        expected_xddot = 3.0 * n**2 * 100.0
        assert jnp.abs(batched[0, 3] - expected_xddot) < _SINGLE_TOL

    def test_vmap_rotation_rtn_to_eci(self):
        """rotation_rtn_to_eci works with vmap over a batch of states."""
        sma1 = R_EARTH + 500e3
        sma2 = R_EARTH + 700e3
        states = jnp.stack([
            _circular_orbit_state(sma1),
            _circular_orbit_state(sma2),
        ])
        Rs = jax.vmap(rotation_rtn_to_eci)(states)
        assert Rs.shape == (2, 3, 3)

    def test_grad_hcw_derivative(self):
        """hcw_derivative supports gradient computation."""
        n = jnp.sqrt(GM_EARTH / (R_EARTH + 500e3) ** 3)

        def scalar_fn(state):
            return jnp.sum(hcw_derivative(state, n) ** 2)

        state = jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        grad = jax.grad(scalar_fn)(state)
        assert grad.shape == (6,)
        # Gradient should be nonzero for the x component
        assert jnp.abs(grad[0]) > 0.0
