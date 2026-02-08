"""Tests for the attitude_dynamics module.

Tests cover:
- Configuration dataclasses and presets
- Quaternion kinematics (derivative)
- Euler rotational equation
- Gravity gradient torque (direction, scaling, alignment)
- Torque-free propagation (angular momentum conservation)
- Gravity gradient propagation (oscillation about nadir)
- JIT compatibility
- Integrator compatibility (rk4_step, dp54_step)
- State normalization utility
"""

import jax
import jax.numpy as jnp
import pytest

from astrojax.config import get_dtype
from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.integrators import AdaptiveConfig, dp54_step, rk4_step

from astrojax.attitude_dynamics.config import (
    AttitudeDynamicsConfig,
    SpacecraftInertia,
)
from astrojax.attitude_dynamics.euler_dynamics import (
    euler_equation,
    quaternion_derivative,
)
from astrojax.attitude_dynamics.factory import create_attitude_dynamics
from astrojax.attitude_dynamics.gravity_gradient import torque_gravity_gradient
from astrojax.attitude_dynamics.utils import normalize_attitude_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_q() -> jnp.ndarray:
    """Identity quaternion [w, x, y, z]."""
    return jnp.array([1.0, 0.0, 0.0, 0.0], dtype=get_dtype())


def _asymmetric_inertia() -> SpacecraftInertia:
    """Asymmetric spacecraft inertia (all axes different)."""
    return SpacecraftInertia.from_principal(10.0, 20.0, 30.0)


def _leo_position_fn():
    """Constant position function for a circular LEO orbit."""
    r = jnp.array([R_EARTH + 500e3, 0.0, 0.0], dtype=get_dtype())
    return lambda t: r


# ===========================================================================
# Configuration dataclass tests
# ===========================================================================


class TestSpacecraftInertia:
    """Tests for SpacecraftInertia."""

    def test_default(self):
        """Default inertia is 3x3 identity."""
        inertia = SpacecraftInertia()
        assert inertia.I.shape == (3, 3)
        assert jnp.allclose(inertia.I, jnp.eye(3))

    def test_from_principal(self):
        """from_principal creates diagonal tensor."""
        inertia = SpacecraftInertia.from_principal(10.0, 20.0, 30.0)
        expected = jnp.diag(jnp.array([10.0, 20.0, 30.0]))
        assert jnp.allclose(inertia.I, expected)

    def test_from_principal_values(self):
        """from_principal stores correct diagonal values."""
        inertia = SpacecraftInertia.from_principal(1.5, 2.5, 3.5)
        assert float(inertia.I[0, 0]) == pytest.approx(1.5)
        assert float(inertia.I[1, 1]) == pytest.approx(2.5)
        assert float(inertia.I[2, 2]) == pytest.approx(3.5)

    def test_frozen(self):
        """SpacecraftInertia is immutable."""
        inertia = SpacecraftInertia()
        with pytest.raises(AttributeError):
            inertia.I = jnp.eye(3) * 2.0


class TestAttitudeDynamicsConfig:
    """Tests for AttitudeDynamicsConfig."""

    def test_defaults(self):
        """Default config has no torques enabled."""
        cfg = AttitudeDynamicsConfig()
        assert cfg.gravity_gradient is False
        assert cfg.mu == pytest.approx(GM_EARTH)

    def test_frozen(self):
        """AttitudeDynamicsConfig is immutable."""
        cfg = AttitudeDynamicsConfig()
        with pytest.raises(AttributeError):
            cfg.gravity_gradient = True

    def test_torque_free_preset(self):
        """torque_free() preset has no torques."""
        inertia = _asymmetric_inertia()
        cfg = AttitudeDynamicsConfig.torque_free(inertia)
        assert cfg.gravity_gradient is False
        assert jnp.allclose(cfg.inertia.I, inertia.I)

    def test_with_gravity_gradient_preset(self):
        """with_gravity_gradient() preset enables gravity gradient."""
        inertia = _asymmetric_inertia()
        cfg = AttitudeDynamicsConfig.with_gravity_gradient(inertia)
        assert cfg.gravity_gradient is True
        assert cfg.mu == pytest.approx(GM_EARTH)

    def test_with_gravity_gradient_custom_mu(self):
        """with_gravity_gradient() accepts custom mu."""
        inertia = _asymmetric_inertia()
        mu_custom = 1.0e10
        cfg = AttitudeDynamicsConfig.with_gravity_gradient(inertia, mu=mu_custom)
        assert cfg.mu == pytest.approx(mu_custom)


# ===========================================================================
# Quaternion kinematics tests
# ===========================================================================


class TestQuaternionDerivative:
    """Tests for quaternion_derivative."""

    def test_output_shape(self):
        """Output is shape (4,)."""
        q = _identity_q()
        omega = jnp.array([0.1, 0.0, 0.0], dtype=get_dtype())
        q_dot = quaternion_derivative(q, omega)
        assert q_dot.shape == (4,)

    def test_identity_q_with_zero_omega(self):
        """Zero angular velocity gives zero quaternion derivative."""
        q = _identity_q()
        omega = jnp.zeros(3, dtype=get_dtype())
        q_dot = quaternion_derivative(q, omega)
        assert jnp.allclose(q_dot, jnp.zeros(4), atol=1e-10)

    def test_identity_q_with_x_rotation(self):
        """Rotation about x-axis from identity quaternion.

        q_dot = 0.5 * Omega(omega) @ q with omega = [wx, 0, 0]:
        q_dot = 0.5 * [0, wx, 0, 0] (for identity q)
        """
        q = _identity_q()
        wx = 0.1
        omega = jnp.array([wx, 0.0, 0.0], dtype=get_dtype())
        q_dot = quaternion_derivative(q, omega)

        expected = jnp.array([0.0, 0.5 * wx, 0.0, 0.0], dtype=get_dtype())
        assert jnp.allclose(q_dot, expected, atol=1e-6)

    def test_identity_q_with_z_rotation(self):
        """Rotation about z-axis from identity quaternion.

        q_dot = 0.5 * [0, 0, 0, wz] for identity q.
        """
        q = _identity_q()
        wz = 0.2
        omega = jnp.array([0.0, 0.0, wz], dtype=get_dtype())
        q_dot = quaternion_derivative(q, omega)

        expected = jnp.array([0.0, 0.0, 0.0, 0.5 * wz], dtype=get_dtype())
        assert jnp.allclose(q_dot, expected, atol=1e-6)

    def test_preserves_unit_norm_infinitesimally(self):
        """q_dot is perpendicular to q (preserves unit norm to first order).

        For a unit quaternion, d/dt(|q|^2) = 2 * q . q_dot = 0.
        """
        q = jnp.array([0.5, 0.5, 0.5, 0.5], dtype=get_dtype())
        omega = jnp.array([0.1, -0.2, 0.3], dtype=get_dtype())
        q_dot = quaternion_derivative(q, omega)

        # q . q_dot should be zero (orthogonality)
        dot = float(jnp.dot(q, q_dot))
        assert abs(dot) < 1e-6


# ===========================================================================
# Euler equation tests
# ===========================================================================


class TestEulerEquation:
    """Tests for euler_equation."""

    def test_output_shape(self):
        """Output is shape (3,)."""
        omega = jnp.array([0.1, 0.0, 0.0], dtype=get_dtype())
        J = jnp.diag(jnp.array([10.0, 20.0, 30.0], dtype=get_dtype()))
        tau = jnp.zeros(3, dtype=get_dtype())
        omega_dot = euler_equation(omega, J, tau)
        assert omega_dot.shape == (3,)

    def test_principal_axis_spin_torque_free(self):
        """Spin about a principal axis with no torque gives zero acceleration."""
        J = jnp.diag(jnp.array([10.0, 20.0, 30.0], dtype=get_dtype()))
        tau = jnp.zeros(3, dtype=get_dtype())

        for axis in [
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([0.0, 1.0, 0.0]),
            jnp.array([0.0, 0.0, 1.0]),
        ]:
            omega = 0.5 * axis
            omega_dot = euler_equation(omega, J, tau)
            assert jnp.allclose(omega_dot, jnp.zeros(3), atol=1e-6), (
                f"Principal axis spin {axis} should give zero omega_dot"
            )

    def test_off_axis_spin_nonzero(self):
        """Off-axis spin with asymmetric inertia produces nonzero acceleration."""
        J = jnp.diag(jnp.array([10.0, 20.0, 30.0], dtype=get_dtype()))
        omega = jnp.array([0.1, 0.2, 0.3], dtype=get_dtype())
        tau = jnp.zeros(3, dtype=get_dtype())
        omega_dot = euler_equation(omega, J, tau)
        assert float(jnp.linalg.norm(omega_dot)) > 1e-6

    def test_known_euler_result(self):
        """Manual computation of Euler's equation.

        I = diag(10, 20, 30), omega = [1, 0, 0], tau = 0
        I @ omega = [10, 0, 0]
        omega x (I @ omega) = [1,0,0] x [10,0,0] = [0, 0, 0]
        omega_dot = I^{-1} @ [0, 0, 0] = [0, 0, 0]
        """
        J = jnp.diag(jnp.array([10.0, 20.0, 30.0], dtype=get_dtype()))
        omega = jnp.array([1.0, 0.0, 0.0], dtype=get_dtype())
        tau = jnp.zeros(3, dtype=get_dtype())
        omega_dot = euler_equation(omega, J, tau)
        assert jnp.allclose(omega_dot, jnp.zeros(3), atol=1e-6)

    def test_known_euler_result_off_axis(self):
        """Manual computation with off-axis spin.

        I = diag(10, 20, 30), omega = [1, 1, 0], tau = 0
        I @ omega = [10, 20, 0]
        omega x (I @ omega) = [1,1,0] x [10,20,0] = [0*0-0*20, 0*10-1*0, 1*20-1*10]
                             = [0, 0, 10]
        rhs = -[0, 0, 10] = [0, 0, -10]
        omega_dot = I^{-1} @ [0, 0, -10] = [0, 0, -10/30] = [0, 0, -1/3]
        """
        J = jnp.diag(jnp.array([10.0, 20.0, 30.0], dtype=get_dtype()))
        omega = jnp.array([1.0, 1.0, 0.0], dtype=get_dtype())
        tau = jnp.zeros(3, dtype=get_dtype())
        omega_dot = euler_equation(omega, J, tau)
        expected = jnp.array([0.0, 0.0, -1.0 / 3.0], dtype=get_dtype())
        assert jnp.allclose(omega_dot, expected, atol=1e-5)

    def test_with_external_torque(self):
        """External torque produces expected angular acceleration.

        I = diag(10, 20, 30), omega = [0, 0, 0], tau = [10, 0, 0]
        omega x (I @ omega) = 0
        rhs = [10, 0, 0]
        omega_dot = I^{-1} @ [10, 0, 0] = [1, 0, 0]
        """
        J = jnp.diag(jnp.array([10.0, 20.0, 30.0], dtype=get_dtype()))
        omega = jnp.zeros(3, dtype=get_dtype())
        tau = jnp.array([10.0, 0.0, 0.0], dtype=get_dtype())
        omega_dot = euler_equation(omega, J, tau)
        expected = jnp.array([1.0, 0.0, 0.0], dtype=get_dtype())
        assert jnp.allclose(omega_dot, expected, atol=1e-6)


# ===========================================================================
# Gravity gradient torque tests
# ===========================================================================


class TestGravityGradient:
    """Tests for torque_gravity_gradient."""

    def test_output_shape(self):
        """Output is shape (3,)."""
        q = _identity_q()
        r_eci = jnp.array([7000e3, 0.0, 0.0], dtype=get_dtype())
        J = jnp.diag(jnp.array([10.0, 20.0, 30.0], dtype=get_dtype()))
        tau = torque_gravity_gradient(q, r_eci, J)
        assert tau.shape == (3,)

    def test_zero_torque_principal_axis_aligned(self):
        """Zero torque when a principal axis is aligned with nadir.

        With identity quaternion and r_eci along x-axis, r_hat_body = [1,0,0].
        I @ r_hat_body = [Ixx, 0, 0].
        r_hat_body x (I @ r_hat_body) = [1,0,0] x [Ixx, 0, 0] = [0, 0, 0].
        """
        q = _identity_q()
        r_eci = jnp.array([7000e3, 0.0, 0.0], dtype=get_dtype())
        J = jnp.diag(jnp.array([10.0, 20.0, 30.0], dtype=get_dtype()))
        tau = torque_gravity_gradient(q, r_eci, J)
        assert jnp.allclose(tau, jnp.zeros(3), atol=1e-10)

    def test_nonzero_torque_misaligned(self):
        """Nonzero torque when body axes are misaligned with nadir.

        With identity quaternion and r_eci off-axis, the torque should
        be nonzero for asymmetric inertia.
        """
        q = _identity_q()
        # Position at 45 degrees in xy plane
        r_mag = 7000e3
        r_eci = jnp.array(
            [r_mag / jnp.sqrt(2.0), r_mag / jnp.sqrt(2.0), 0.0],
            dtype=get_dtype(),
        )
        J = jnp.diag(jnp.array([10.0, 20.0, 30.0], dtype=get_dtype()))
        tau = torque_gravity_gradient(q, r_eci, J)
        assert float(jnp.linalg.norm(tau)) > 1e-10

    def test_inverse_cube_scaling(self):
        """Gravity gradient scales as 1/r^3."""
        q = _identity_q()
        J = jnp.diag(jnp.array([10.0, 20.0, 30.0], dtype=get_dtype()))

        r1 = 7000e3
        r2 = 14000e3  # double the distance

        r_eci_1 = jnp.array(
            [r1 / jnp.sqrt(2.0), r1 / jnp.sqrt(2.0), 0.0], dtype=get_dtype()
        )
        r_eci_2 = jnp.array(
            [r2 / jnp.sqrt(2.0), r2 / jnp.sqrt(2.0), 0.0], dtype=get_dtype()
        )

        tau_1 = torque_gravity_gradient(q, r_eci_1, J)
        tau_2 = torque_gravity_gradient(q, r_eci_2, J)

        mag_1 = float(jnp.linalg.norm(tau_1))
        mag_2 = float(jnp.linalg.norm(tau_2))

        # tau ~ 1/r^3, so doubling r should give 1/8 the torque
        ratio = mag_1 / mag_2
        assert ratio == pytest.approx(8.0, rel=1e-4)

    def test_symmetric_inertia_zero_torque(self):
        """Gravity gradient is zero for a spherically symmetric body.

        When I = c * eye(3), I @ r_hat = c * r_hat,
        so r_hat x (c * r_hat) = 0.
        """
        q = _identity_q()
        r_eci = jnp.array(
            [7000e3 / jnp.sqrt(3.0)] * 3, dtype=get_dtype()
        )
        J = 10.0 * jnp.eye(3, dtype=get_dtype())
        tau = torque_gravity_gradient(q, r_eci, J)
        assert jnp.allclose(tau, jnp.zeros(3), atol=1e-10)


# ===========================================================================
# Factory tests
# ===========================================================================


class TestCreateAttitudeDynamics:
    """Tests for create_attitude_dynamics factory."""

    def test_returns_callable(self):
        """Factory returns a callable."""
        inertia = _asymmetric_inertia()
        cfg = AttitudeDynamicsConfig.torque_free(inertia)
        dynamics = create_attitude_dynamics(cfg, _leo_position_fn())
        assert callable(dynamics)

    def test_derivative_shape(self):
        """Output is shape (7,)."""
        inertia = _asymmetric_inertia()
        cfg = AttitudeDynamicsConfig.torque_free(inertia)
        dynamics = create_attitude_dynamics(cfg, _leo_position_fn())

        x0 = jnp.concatenate([_identity_q(), jnp.array([0.1, 0.0, 0.0])])
        dx = dynamics(0.0, x0)
        assert dx.shape == (7,)

    def test_torque_free_no_nan(self):
        """Torque-free dynamics produces no NaN."""
        inertia = _asymmetric_inertia()
        cfg = AttitudeDynamicsConfig.torque_free(inertia)
        dynamics = create_attitude_dynamics(cfg, _leo_position_fn())

        x0 = jnp.concatenate([_identity_q(), jnp.array([0.1, -0.2, 0.3])])
        dx = dynamics(0.0, x0)
        assert not jnp.any(jnp.isnan(dx))

    def test_gravity_gradient_no_nan(self):
        """Gravity gradient dynamics produces no NaN."""
        inertia = _asymmetric_inertia()
        cfg = AttitudeDynamicsConfig.with_gravity_gradient(inertia)
        dynamics = create_attitude_dynamics(cfg, _leo_position_fn())

        x0 = jnp.concatenate([_identity_q(), jnp.array([0.01, 0.0, 0.0])])
        dx = dynamics(0.0, x0)
        assert not jnp.any(jnp.isnan(dx))


# ===========================================================================
# Propagation tests
# ===========================================================================


class TestTorqueFreePropagation:
    """Tests for torque-free rigid body propagation."""

    def test_angular_momentum_conservation(self):
        """Angular momentum magnitude is conserved over ~1000 seconds.

        For torque-free motion, L = I @ omega is constant.
        """
        _float = get_dtype()
        inertia = _asymmetric_inertia()
        J = inertia.I
        cfg = AttitudeDynamicsConfig.torque_free(inertia)
        dynamics = create_attitude_dynamics(cfg, _leo_position_fn())

        omega_0 = jnp.array([0.1, -0.05, 0.2], dtype=_float)
        x0 = jnp.concatenate([_identity_q(), omega_0])

        # Initial angular momentum
        L0 = J @ omega_0
        L0_mag = float(jnp.linalg.norm(L0))

        # Propagate with RK4 for 1000 seconds
        dt = 1.0
        n_steps = 1000

        def scan_step(carry, _):
            t, state = carry
            result = rk4_step(dynamics, t, state, dt)
            return (t + dt, result.state), None

        (_, x_final), _ = jax.lax.scan(
            scan_step, (jnp.asarray(0.0, dtype=_float), x0), None, length=n_steps
        )

        # Final angular momentum
        omega_final = x_final[4:7]
        L_final = J @ omega_final
        L_final_mag = float(jnp.linalg.norm(L_final))

        # Angular momentum should be conserved to high precision
        rel_error = abs(L_final_mag - L0_mag) / L0_mag
        assert rel_error < 1e-4, (
            f"Angular momentum not conserved: "
            f"|L0|={L0_mag:.6e}, |L_final|={L_final_mag:.6e}, "
            f"rel_error={rel_error:.6e}"
        )

    def test_quaternion_norm_drift(self):
        """Quaternion norm stays near 1.0 during propagation."""
        _float = get_dtype()
        inertia = _asymmetric_inertia()
        cfg = AttitudeDynamicsConfig.torque_free(inertia)
        dynamics = create_attitude_dynamics(cfg, _leo_position_fn())

        x0 = jnp.concatenate([
            _identity_q(),
            jnp.array([0.1, -0.05, 0.2], dtype=_float),
        ])

        dt = 1.0
        n_steps = 1000

        def scan_step(carry, _):
            t, state = carry
            result = rk4_step(dynamics, t, state, dt)
            return (t + dt, result.state), None

        (_, x_final), _ = jax.lax.scan(
            scan_step, (jnp.asarray(0.0, dtype=_float), x0), None, length=n_steps
        )

        q_norm = float(jnp.linalg.norm(x_final[:4]))
        # Without normalization, drift should still be small for 1000 steps
        assert abs(q_norm - 1.0) < 0.01, (
            f"Quaternion norm drifted to {q_norm}"
        )


class TestGravityGradientPropagation:
    """Tests for gravity gradient propagation."""

    def test_gravity_gradient_restoring_torque(self):
        """Gravity gradient creates a restoring torque toward stable equilibrium.

        Gravity gradient stabilizes the **maximum** inertia axis along nadir.
        Here the z-axis (Izz=30, largest) should be aligned with nadir.

        Setup: z-axis aligned with nadir (r along z in ECI), then pitch
        the body slightly about y.  The restoring torque should oppose
        the pitch displacement.
        """
        _float = get_dtype()
        inertia = SpacecraftInertia.from_principal(10.0, 20.0, 30.0)
        cfg = AttitudeDynamicsConfig.with_gravity_gradient(inertia)

        # Position along z-axis in ECI
        r_mag = R_EARTH + 500e3
        def pos_fn(t):
            return jnp.array([0.0, 0.0, r_mag], dtype=_float)

        dynamics = create_attitude_dynamics(cfg, pos_fn)

        # Small pitch about y-axis: displaces z-axis away from nadir
        theta = 0.05  # ~3 degrees
        q0 = jnp.array(
            [jnp.cos(theta / 2), 0.0, jnp.sin(theta / 2), 0.0],
            dtype=_float,
        )
        x0 = jnp.concatenate([q0, jnp.zeros(3, dtype=_float)])

        # Evaluate derivative at t=0
        dx = dynamics(0.0, x0)

        # The restoring torque should produce an angular acceleration
        # about y that opposes the pitch displacement.
        # Positive theta pitches z-axis away from +z nadir, so the
        # restoring omega_dot_y should be negative.
        omega_dot_y = float(dx[5])
        assert omega_dot_y < 0.0, (
            f"Expected negative restoring omega_dot_y, got {omega_dot_y}"
        )

    def test_gravity_gradient_propagation_no_nan(self):
        """Gravity gradient propagation produces no NaN over 100 steps."""
        _float = get_dtype()
        inertia = SpacecraftInertia.from_principal(10.0, 20.0, 30.0)
        cfg = AttitudeDynamicsConfig.with_gravity_gradient(inertia)

        r_mag = R_EARTH + 500e3
        def pos_fn(t):
            return jnp.array([r_mag, 0.0, 0.0], dtype=_float)
        dynamics = create_attitude_dynamics(cfg, pos_fn)

        theta = 0.05
        q0 = jnp.array(
            [jnp.cos(theta / 2), 0.0, jnp.sin(theta / 2), 0.0],
            dtype=_float,
        )
        x0 = jnp.concatenate([q0, jnp.zeros(3, dtype=_float)])

        dt = 1.0
        n_steps = 100

        def scan_step(carry, _):
            t, state = carry
            result = rk4_step(dynamics, t, state, dt)
            normed = normalize_attitude_state(result.state)
            return (t + dt, normed), normed

        (_, x_final), trajectory = jax.lax.scan(
            scan_step, (jnp.asarray(0.0, dtype=_float), x0), None, length=n_steps
        )

        assert not jnp.any(jnp.isnan(x_final))
        # Quaternion should still be unit norm after normalization
        q_norm = float(jnp.linalg.norm(x_final[:4]))
        assert q_norm == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# JIT compatibility tests
# ===========================================================================


class TestJITCompatibility:
    """Tests for JIT compilation of attitude dynamics."""

    def test_jit_torque_free(self):
        """Torque-free dynamics is JIT-compilable."""
        inertia = _asymmetric_inertia()
        cfg = AttitudeDynamicsConfig.torque_free(inertia)
        dynamics = create_attitude_dynamics(cfg, _leo_position_fn())

        x0 = jnp.concatenate([_identity_q(), jnp.array([0.1, 0.0, 0.0])])

        jit_dynamics = jax.jit(dynamics)
        dx = jit_dynamics(0.0, x0)
        assert dx.shape == (7,)
        assert not jnp.any(jnp.isnan(dx))

    def test_jit_gravity_gradient(self):
        """Gravity gradient dynamics is JIT-compilable."""
        inertia = _asymmetric_inertia()
        cfg = AttitudeDynamicsConfig.with_gravity_gradient(inertia)
        dynamics = create_attitude_dynamics(cfg, _leo_position_fn())

        x0 = jnp.concatenate([_identity_q(), jnp.array([0.01, 0.0, 0.0])])

        jit_dynamics = jax.jit(dynamics)
        dx = jit_dynamics(0.0, x0)
        assert dx.shape == (7,)
        assert not jnp.any(jnp.isnan(dx))

    def test_jit_quaternion_derivative(self):
        """quaternion_derivative is JIT-compilable."""
        q = _identity_q()
        omega = jnp.array([0.1, 0.0, 0.0], dtype=get_dtype())

        jit_qd = jax.jit(quaternion_derivative)
        q_dot = jit_qd(q, omega)
        assert q_dot.shape == (4,)

    def test_jit_euler_equation(self):
        """euler_equation is JIT-compilable."""
        J = jnp.diag(jnp.array([10.0, 20.0, 30.0], dtype=get_dtype()))
        omega = jnp.array([0.1, 0.2, 0.3], dtype=get_dtype())
        tau = jnp.zeros(3, dtype=get_dtype())

        jit_euler = jax.jit(euler_equation)
        omega_dot = jit_euler(omega, J, tau)
        assert omega_dot.shape == (3,)

    def test_jit_gravity_gradient_torque(self):
        """torque_gravity_gradient is JIT-compilable."""
        q = _identity_q()
        r_eci = jnp.array([7000e3, 0.0, 0.0], dtype=get_dtype())
        J = jnp.diag(jnp.array([10.0, 20.0, 30.0], dtype=get_dtype()))

        jit_gg = jax.jit(torque_gravity_gradient)
        tau = jit_gg(q, r_eci, J)
        assert tau.shape == (3,)


# ===========================================================================
# Integrator compatibility tests
# ===========================================================================


class TestIntegratorCompatibility:
    """Tests for dynamics function with actual integrators."""

    def test_rk4_step(self):
        """Single RK4 step produces valid result."""
        inertia = _asymmetric_inertia()
        cfg = AttitudeDynamicsConfig.torque_free(inertia)
        dynamics = create_attitude_dynamics(cfg, _leo_position_fn())

        x0 = jnp.concatenate([_identity_q(), jnp.array([0.1, -0.2, 0.3])])
        result = rk4_step(dynamics, 0.0, x0, 1.0)

        assert result.state.shape == (7,)
        assert not jnp.any(jnp.isnan(result.state))
        # State should have changed
        assert not jnp.allclose(result.state, x0)

    def test_dp54_step(self):
        """Single DP54 step produces valid result."""
        inertia = _asymmetric_inertia()
        cfg = AttitudeDynamicsConfig.torque_free(inertia)
        dynamics = create_attitude_dynamics(cfg, _leo_position_fn())

        x0 = jnp.concatenate([_identity_q(), jnp.array([0.1, -0.2, 0.3])])
        adaptive_cfg = AdaptiveConfig(abs_tol=1e-8, rel_tol=1e-6)
        result = dp54_step(dynamics, 0.0, x0, 1.0, config=adaptive_cfg)

        assert result.state.shape == (7,)
        assert not jnp.any(jnp.isnan(result.state))

    def test_rk4_with_gravity_gradient(self):
        """RK4 step with gravity gradient produces valid result."""
        inertia = _asymmetric_inertia()
        cfg = AttitudeDynamicsConfig.with_gravity_gradient(inertia)
        dynamics = create_attitude_dynamics(cfg, _leo_position_fn())

        x0 = jnp.concatenate([_identity_q(), jnp.array([0.01, 0.0, 0.0])])
        result = rk4_step(dynamics, 0.0, x0, 1.0)

        assert result.state.shape == (7,)
        assert not jnp.any(jnp.isnan(result.state))


# ===========================================================================
# Normalization utility tests
# ===========================================================================


class TestNormalizeAttitudeState:
    """Tests for normalize_attitude_state."""

    def test_already_normalized(self):
        """Normalizing a unit quaternion state is a no-op."""
        x = jnp.concatenate([_identity_q(), jnp.array([0.1, 0.2, 0.3])])
        x_normed = normalize_attitude_state(x)
        assert jnp.allclose(x, x_normed, atol=1e-6)

    def test_restores_unit_norm(self):
        """Normalizes a drifted quaternion back to unit norm."""
        q_drift = jnp.array([1.001, 0.002, -0.001, 0.003], dtype=get_dtype())
        omega = jnp.array([0.1, 0.2, 0.3], dtype=get_dtype())
        x = jnp.concatenate([q_drift, omega])

        x_normed = normalize_attitude_state(x)

        q_norm = float(jnp.linalg.norm(x_normed[:4]))
        assert q_norm == pytest.approx(1.0, abs=1e-6)

    def test_preserves_angular_velocity(self):
        """Normalization does not modify angular velocity."""
        q_drift = jnp.array([1.1, 0.0, 0.0, 0.0], dtype=get_dtype())
        omega = jnp.array([0.1, -0.2, 0.3], dtype=get_dtype())
        x = jnp.concatenate([q_drift, omega])

        x_normed = normalize_attitude_state(x)
        assert jnp.allclose(x_normed[4:7], omega, atol=1e-10)

    def test_output_shape(self):
        """Output is shape (7,)."""
        x = jnp.concatenate([_identity_q(), jnp.zeros(3)])
        x_normed = normalize_attitude_state(x)
        assert x_normed.shape == (7,)

    def test_jit_compatible(self):
        """normalize_attitude_state is JIT-compilable."""
        x = jnp.concatenate([
            jnp.array([1.01, 0.0, 0.0, 0.0]),
            jnp.array([0.1, 0.2, 0.3]),
        ])
        jit_norm = jax.jit(normalize_attitude_state)
        x_normed = jit_norm(x)
        q_norm = float(jnp.linalg.norm(x_normed[:4]))
        assert q_norm == pytest.approx(1.0, abs=1e-6)
