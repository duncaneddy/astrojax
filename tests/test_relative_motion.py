import jax
import jax.numpy as jnp
import pytest

from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.coordinates import state_koe_to_eci
from astrojax.integrators import rk4_step
from astrojax.relative_motion import (
    hcw_derivative,
    hcw_stm,
    rotation_eci_to_rtn,
    rotation_rtn_to_eci,
    state_eci_to_roe,
    state_eci_to_rtn,
    state_oe_to_roe,
    state_roe_to_eci,
    state_roe_to_oe,
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
# HCW STM tests
# ──────────────────────────────────────────────


class TestHCWSTM:
    @pytest.fixture()
    def n(self):
        """Mean motion for a 500 km LEO orbit."""
        sma = R_EARTH + 500e3
        return jnp.sqrt(GM_EARTH / sma**3)

    def test_identity_at_t_zero(self, n):
        """STM at t=0 should be the 6x6 identity matrix."""
        phi = hcw_stm(0.0, n)
        assert jnp.allclose(phi, jnp.eye(6), atol=_SINGLE_TOL)

    def test_output_shape(self, n):
        """STM should be a (6, 6) matrix."""
        phi = hcw_stm(60.0, n)
        assert phi.shape == (6, 6)

    def test_radial_displacement_propagation(self, n):
        """Propagating [x0,0,0,0,0,0] should give x(t) = (4 - 3cos(nt)) x0."""
        x0 = 100.0
        t = 300.0
        state0 = jnp.array([x0, 0.0, 0.0, 0.0, 0.0, 0.0])
        state_t = hcw_stm(t, n) @ state0
        expected_x = (4.0 - 3.0 * jnp.cos(n * t)) * x0
        assert jnp.abs(state_t[0] - expected_x) < _SINGLE_TOL

    def test_cross_track_decoupled(self, n):
        """z-only initial state stays in z: z(t) = z0 cos(nt)."""
        z0 = 50.0
        t = 200.0
        state0 = jnp.array([0.0, 0.0, z0, 0.0, 0.0, 0.0])
        state_t = hcw_stm(t, n) @ state0
        expected_z = z0 * jnp.cos(n * t)
        assert jnp.abs(state_t[2] - expected_z) < _SINGLE_TOL
        # In-plane components should be zero
        assert jnp.abs(state_t[0]) < _SINGLE_TOL
        assert jnp.abs(state_t[1]) < _SINGLE_TOL

    def test_consistency_with_derivative(self, n):
        """STM propagation should match RK4 integration of hcw_derivative."""
        state0 = jnp.array([100.0, 50.0, 30.0, 0.1, -0.2, 0.05])
        dt_total = 60.0
        n_steps = 200
        dt = dt_total / n_steps

        # Wrap hcw_derivative to match rk4_step's dynamics(t, state) signature
        def dynamics(t, s):
            return hcw_derivative(s, n)

        # Integrate with many small RK4 steps
        state_rk4 = state0
        t_current = jnp.array(0.0)
        for _ in range(n_steps):
            result = rk4_step(dynamics, t_current, state_rk4, dt)
            state_rk4 = result.state
            t_current = t_current + dt

        # Propagate with STM
        state_stm = hcw_stm(dt_total, n) @ state0

        assert jnp.allclose(state_stm, state_rk4, atol=_SINGLE_TOL)

    def test_stm_composition(self, n):
        """Semigroup property: Phi(t1+t2) == Phi(t2) @ Phi(t1)."""
        t1 = 120.0
        t2 = 180.0
        phi_total = hcw_stm(t1 + t2, n)
        phi_composed = hcw_stm(t2, n) @ hcw_stm(t1, n)
        assert jnp.allclose(phi_total, phi_composed, atol=_SINGLE_TOL)

    def test_determinant_is_one(self, n):
        """det(Phi(t)) should be 1.0 (volume-preserving)."""
        phi = hcw_stm(300.0, n)
        det = jnp.linalg.det(phi)
        assert jnp.abs(det - 1.0) < _SINGLE_TOL

    def test_use_degrees(self, n):
        """use_degrees=True should give the same result as radians."""
        t = 200.0
        n_deg = jnp.rad2deg(n)
        phi_rad = hcw_stm(t, n)
        phi_deg = hcw_stm(t, n_deg, use_degrees=True)
        assert jnp.allclose(phi_rad, phi_deg, atol=_SINGLE_TOL)


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
        states = jnp.array(
            [
                [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 200.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 50.0, 0.0, 0.0, 0.0],
            ]
        )
        batched = jax.vmap(hcw_derivative, in_axes=(0, None))(states, n)
        assert batched.shape == (3, 6)
        # Spot-check: first row x_ddot = 3 n^2 * 100
        expected_xddot = 3.0 * n**2 * 100.0
        assert jnp.abs(batched[0, 3] - expected_xddot) < _SINGLE_TOL

    def test_vmap_rotation_rtn_to_eci(self):
        """rotation_rtn_to_eci works with vmap over a batch of states."""
        sma1 = R_EARTH + 500e3
        sma2 = R_EARTH + 700e3
        states = jnp.stack(
            [
                _circular_orbit_state(sma1),
                _circular_orbit_state(sma2),
            ]
        )
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

    def test_jit_hcw_stm(self):
        """hcw_stm is JIT-compilable."""
        n = jnp.sqrt(GM_EARTH / (R_EARTH + 500e3) ** 3)
        t = jnp.array(300.0)
        phi_eager = hcw_stm(t, n)
        phi_jit = jax.jit(hcw_stm)(t, n)
        assert jnp.allclose(phi_eager, phi_jit, atol=_SINGLE_TOL)

    def test_vmap_hcw_stm(self):
        """hcw_stm works with vmap over a batch of time values."""
        n = jnp.sqrt(GM_EARTH / (R_EARTH + 500e3) ** 3)
        times = jnp.array([0.0, 60.0, 120.0, 300.0])
        batched = jax.vmap(hcw_stm, in_axes=(0, None))(times, n)
        assert batched.shape == (4, 6, 6)
        # First entry (t=0) should be identity
        assert jnp.allclose(batched[0], jnp.eye(6), atol=_SINGLE_TOL)

    def test_grad_hcw_stm(self):
        """hcw_stm supports gradient computation w.r.t. time."""
        n = jnp.sqrt(GM_EARTH / (R_EARTH + 500e3) ** 3)

        def scalar_fn(t):
            return jnp.sum(hcw_stm(t, n) ** 2)

        t = jnp.array(300.0)
        grad = jax.grad(scalar_fn)(t)
        assert jnp.isfinite(grad)
        assert jnp.abs(grad) > 0.0


# ──────────────────────────────────────────────
# OE <-> ROE transformation tests
# ──────────────────────────────────────────────

# ROE tolerances — float32 introduces rounding on the large SMA values
_ROE_DA_TOL = 1e-5  # relative SMA (dimensionless)
_ROE_ANGLE_TOL = 1e-3  # degrees (angular ROE components)
_ROE_ECC_TOL = 1e-5  # eccentricity vector components (dimensionless)
_OE_SMA_TOL = 10.0  # metres (roundtrip SMA)
_OE_ECC_RTOL = 1e-3  # eccentricity roundtrip (relative)
_OE_ANGLE_DEG_TOL = 1e-2  # degrees (roundtrip angles)


class TestOEtoROE:
    def test_output_shape(self):
        """state_oe_to_roe returns a 6-element array."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])
        roe = state_oe_to_roe(oe_c, oe_d, use_degrees=True)
        assert roe.shape == (6,)

    def test_identical_satellites(self):
        """ROE of identical satellites should be near-zero."""
        oe = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        roe = state_oe_to_roe(oe, oe, use_degrees=True)
        assert jnp.abs(float(roe[0])) < _ROE_DA_TOL  # da
        assert jnp.abs(float(roe[2])) < _ROE_ECC_TOL  # dex
        assert jnp.abs(float(roe[3])) < _ROE_ECC_TOL  # dey

    def test_da_positive_when_deputy_higher(self):
        """da should be positive when deputy has larger SMA."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = jnp.array([R_EARTH + 701e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        roe = state_oe_to_roe(oe_c, oe_d, use_degrees=True)
        assert float(roe[0]) > 0.0

    def test_da_value(self):
        """da = (ad - ac) / ac should be computed correctly."""
        ac = R_EARTH + 700e3
        ad = R_EARTH + 701e3
        oe_c = jnp.array([ac, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = jnp.array([ad, 0.001, 97.8, 15.0, 30.0, 45.0])
        roe = state_oe_to_roe(oe_c, oe_d, use_degrees=True)
        expected_da = (ad - ac) / ac
        assert jnp.abs(float(roe[0]) - float(expected_da)) < _ROE_DA_TOL

    def test_radians_vs_degrees(self):
        """Radians and degrees modes should give consistent results."""
        oe_c_deg = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d_deg = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

        deg2rad = jnp.pi / 180.0
        oe_c_rad = jnp.array(
            [
                R_EARTH + 700e3,
                0.001,
                97.8 * deg2rad,
                15.0 * deg2rad,
                30.0 * deg2rad,
                45.0 * deg2rad,
            ]
        )
        oe_d_rad = jnp.array(
            [
                R_EARTH + 701e3,
                0.0015,
                97.85 * deg2rad,
                15.05 * deg2rad,
                30.05 * deg2rad,
                45.05 * deg2rad,
            ]
        )

        roe_deg = state_oe_to_roe(oe_c_deg, oe_d_deg, use_degrees=True)
        roe_rad = state_oe_to_roe(oe_c_rad, oe_d_rad, use_degrees=False)

        # da, dex, dey should match exactly (no angle conversion)
        assert jnp.allclose(roe_deg[0], roe_rad[0], atol=_ROE_DA_TOL)
        assert jnp.allclose(roe_deg[2], roe_rad[2], atol=_ROE_ECC_TOL)
        assert jnp.allclose(roe_deg[3], roe_rad[3], atol=_ROE_ECC_TOL)

        # Angular components: convert deg result to radians and compare
        assert jnp.allclose(
            jnp.deg2rad(roe_deg[1]),
            roe_rad[1],
            atol=1e-4,
        )


class TestROEtoOE:
    def test_output_shape(self):
        """state_roe_to_oe returns a 6-element array."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        roe = jnp.array([1.41e-4, 0.093, 4.32e-4, 2.51e-4, 0.05, 0.0495])
        oe_d = state_roe_to_oe(oe_c, roe, use_degrees=True)
        assert oe_d.shape == (6,)

    def test_roundtrip_degrees(self):
        """OE -> ROE -> OE roundtrip recovers the original deputy OE (degrees)."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d_orig = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

        roe = state_oe_to_roe(oe_c, oe_d_orig, use_degrees=True)
        oe_d_recovered = state_roe_to_oe(oe_c, roe, use_degrees=True)

        assert jnp.abs(float(oe_d_recovered[0]) - float(oe_d_orig[0])) < _OE_SMA_TOL
        assert jnp.abs(float(oe_d_recovered[1]) - float(oe_d_orig[1])) < _ROE_ECC_TOL
        assert jnp.abs(float(oe_d_recovered[2]) - float(oe_d_orig[2])) < _OE_ANGLE_DEG_TOL
        assert jnp.abs(float(oe_d_recovered[3]) - float(oe_d_orig[3])) < _OE_ANGLE_DEG_TOL
        assert jnp.abs(float(oe_d_recovered[4]) - float(oe_d_orig[4])) < _OE_ANGLE_DEG_TOL
        assert jnp.abs(float(oe_d_recovered[5]) - float(oe_d_orig[5])) < _OE_ANGLE_DEG_TOL

    def test_roundtrip_radians(self):
        """OE -> ROE -> OE roundtrip recovers the original deputy OE (radians)."""
        deg2rad = jnp.pi / 180.0
        oe_c = jnp.array(
            [
                R_EARTH + 700e3,
                0.001,
                97.8 * deg2rad,
                15.0 * deg2rad,
                30.0 * deg2rad,
                45.0 * deg2rad,
            ]
        )
        oe_d_orig = jnp.array(
            [
                R_EARTH + 701e3,
                0.0015,
                97.85 * deg2rad,
                15.05 * deg2rad,
                30.05 * deg2rad,
                45.05 * deg2rad,
            ]
        )

        roe = state_oe_to_roe(oe_c, oe_d_orig, use_degrees=False)
        oe_d_recovered = state_roe_to_oe(oe_c, roe, use_degrees=False)

        assert jnp.abs(float(oe_d_recovered[0]) - float(oe_d_orig[0])) < _OE_SMA_TOL
        assert jnp.abs(float(oe_d_recovered[1]) - float(oe_d_orig[1])) < _ROE_ECC_TOL
        angle_rad_tol = _OE_ANGLE_DEG_TOL * deg2rad
        for idx in range(2, 6):
            assert jnp.abs(float(oe_d_recovered[idx]) - float(oe_d_orig[idx])) < angle_rad_tol


# ──────────────────────────────────────────────
# ECI <-> ROE transformation tests
# ──────────────────────────────────────────────


class TestECItoROE:
    def test_output_shape(self):
        """state_eci_to_roe returns a 6-element array."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])
        x_c = state_koe_to_eci(oe_c, use_degrees=True)
        x_d = state_koe_to_eci(oe_d, use_degrees=True)
        roe = state_eci_to_roe(x_c, x_d, use_degrees=True)
        assert roe.shape == (6,)

    def test_matches_oe_path(self):
        """ECI->ROE should approximately match OE->ROE (float32 KOE conversion adds error)."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

        roe_from_oe = state_oe_to_roe(oe_c, oe_d, use_degrees=True)

        x_c = state_koe_to_eci(oe_c, use_degrees=True)
        x_d = state_koe_to_eci(oe_d, use_degrees=True)
        roe_from_eci = state_eci_to_roe(x_c, x_d, use_degrees=True)

        # Wider tolerance for ECI path due to float32 KOE conversion roundoff
        assert jnp.allclose(roe_from_eci[0], roe_from_oe[0], atol=1e-4)
        assert jnp.allclose(roe_from_eci[2], roe_from_oe[2], atol=5e-4)
        assert jnp.allclose(roe_from_eci[3], roe_from_oe[3], atol=5e-4)


class TestROEtoECI:
    def test_output_shape(self):
        """state_roe_to_eci returns a 6-element array."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        x_c = state_koe_to_eci(oe_c, use_degrees=True)
        roe = jnp.array([0.000142857, 0.05, 0.0005, -0.0003, 0.01, -0.02])
        x_d = state_roe_to_eci(x_c, roe, use_degrees=True)
        assert x_d.shape == (6,)

    def test_valid_orbit(self):
        """Deputy state from ROE should represent a valid orbit."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        x_c = state_koe_to_eci(oe_c, use_degrees=True)
        roe = jnp.array([0.000142857, 0.05, 0.0005, -0.0003, 0.01, -0.02])
        x_d = state_roe_to_eci(x_c, roe, use_degrees=True)

        pos_mag = float(jnp.linalg.norm(x_d[:3]))
        vel_mag = float(jnp.linalg.norm(x_d[3:6]))
        assert pos_mag > R_EARTH
        assert pos_mag < R_EARTH + 2000e3
        assert vel_mag > 6000.0
        assert vel_mag < 9000.0

    def test_eci_roe_roundtrip(self):
        """ECI -> ROE -> ECI roundtrip recovers deputy state."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

        x_c = state_koe_to_eci(oe_c, use_degrees=True)
        x_d_orig = state_koe_to_eci(oe_d, use_degrees=True)

        roe = state_eci_to_roe(x_c, x_d_orig, use_degrees=True)
        x_d_recovered = state_roe_to_eci(x_c, roe, use_degrees=True)

        # ECI path has two KOE conversions (ECI→KOE→ROE→KOE→ECI), so
        # float32 roundoff on ~7e6 m positions accumulates to ~500 m
        assert jnp.allclose(x_d_recovered[:3], x_d_orig[:3], atol=500.0)
        assert jnp.allclose(x_d_recovered[3:], x_d_orig[3:], atol=0.5)


# ──────────────────────────────────────────────
# JAX compatibility for ROE functions
# ──────────────────────────────────────────────


class TestROEJAXCompatibility:
    def test_jit_state_oe_to_roe(self):
        """state_oe_to_roe is JIT-compilable."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

        roe_eager = state_oe_to_roe(oe_c, oe_d, use_degrees=True)

        @jax.jit
        def f(c, d):
            return state_oe_to_roe(c, d, use_degrees=True)

        roe_jit = f(oe_c, oe_d)
        assert jnp.allclose(roe_eager, roe_jit, atol=_SINGLE_TOL)

    def test_jit_state_roe_to_oe(self):
        """state_roe_to_oe is JIT-compilable."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        roe = jnp.array([1.41e-4, 0.093, 4.32e-4, 2.51e-4, 0.05, 0.0495])

        oe_eager = state_roe_to_oe(oe_c, roe, use_degrees=True)

        @jax.jit
        def f(c, r):
            return state_roe_to_oe(c, r, use_degrees=True)

        oe_jit = f(oe_c, roe)
        assert jnp.allclose(oe_eager, oe_jit, atol=_SINGLE_TOL)

    def test_jit_state_eci_to_roe(self):
        """state_eci_to_roe is JIT-compilable."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_d = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])
        x_c = state_koe_to_eci(oe_c, use_degrees=True)
        x_d = state_koe_to_eci(oe_d, use_degrees=True)

        @jax.jit
        def f(c, d):
            return state_eci_to_roe(c, d, use_degrees=True)

        roe_jit = f(x_c, x_d)
        # Verify JIT compilation succeeds and output is valid
        assert roe_jit.shape == (6,)
        assert jnp.all(jnp.isfinite(roe_jit))

    def test_jit_state_roe_to_eci(self):
        """state_roe_to_eci is JIT-compilable."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        x_c = state_koe_to_eci(oe_c, use_degrees=True)
        roe = jnp.array([0.000142857, 0.05, 0.0005, -0.0003, 0.01, -0.02])

        @jax.jit
        def f(c, r):
            return state_roe_to_eci(c, r, use_degrees=True)

        x_jit = f(x_c, roe)
        # Verify JIT compilation succeeds and output is valid
        assert x_jit.shape == (6,)
        assert jnp.all(jnp.isfinite(x_jit))
        pos_mag = float(jnp.linalg.norm(x_jit[:3]))
        assert pos_mag > R_EARTH

    def test_vmap_state_oe_to_roe(self):
        """state_oe_to_roe works with vmap over batched deputies."""
        oe_c = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_deputies = jnp.array(
            [
                [R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05],
                [R_EARTH + 702e3, 0.002, 97.9, 15.1, 30.1, 45.1],
            ]
        )

        def f(oe_d):
            return state_oe_to_roe(oe_c, oe_d, use_degrees=True)

        roes = jax.vmap(f)(oe_deputies)
        assert roes.shape == (2, 6)
