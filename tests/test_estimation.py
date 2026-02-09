"""Tests for the astrojax.estimation and astrojax.orbit_measurements modules.

Tests cover:
- FilterState, UKFConfig, FilterResult type construction and access
- EKF predict and update on linear and nonlinear systems
- UKF predict and update on linear and nonlinear systems
- EKF vs UKF consistency on linear systems
- GNSS measurement model functions and noise constructors
- JIT compatibility for all filter functions
- jax.lax.scan composition for sequential filtering
- End-to-end orbit determination with simulated GNSS
"""

import jax
import jax.numpy as jnp
import pytest

from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.estimation import (
    FilterResult,
    FilterState,
    UKFConfig,
    ekf_predict,
    ekf_update,
    ukf_predict,
    ukf_update,
)
from astrojax.orbit_measurements import (
    gnss_measurement_noise,
    gnss_position_measurement,
    gnss_position_velocity_measurement,
    gnss_position_velocity_noise,
)

# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────


def _linear_propagate(x):
    """Linear propagation: x_next = A @ x with A = [[1, dt], [0, 1]]."""
    dt = 0.1
    A = jnp.array([[1.0, dt], [0.0, 1.0]])
    return A @ x


def _linear_measurement(x):
    """Linear measurement: z = H @ x with H = [[1, 0]]."""
    return x[:1]


def _nonlinear_propagate(x):
    """Nonlinear propagation: simple harmonic oscillator Euler step."""
    dt = 0.01
    return x + jnp.array([x[1], -x[0]]) * dt


def _two_body_propagate(state):
    """Single RK4 step of two-body dynamics for testing."""
    r = state[:3]
    v = state[3:]
    r_norm = jnp.linalg.norm(r)
    a = -GM_EARTH * r / r_norm**3
    dt = 10.0
    # Euler step for simplicity in tests
    r_new = r + v * dt + 0.5 * a * dt**2
    v_new = v + a * dt
    return jnp.concatenate([r_new, v_new])


def _circular_orbit_state():
    """Create a circular equatorial orbit state at 500 km altitude."""
    sma = R_EARTH + 500e3
    v_circ = jnp.sqrt(GM_EARTH / sma)
    return jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])


# ──────────────────────────────────────────────
# Type tests
# ──────────────────────────────────────────────


class TestFilterState:
    def test_construction(self):
        """FilterState holds state vector and covariance."""
        x = jnp.array([1.0, 2.0])
        P = jnp.eye(2)
        fs = FilterState(x=x, P=P)
        assert fs.x.shape == (2,)
        assert fs.P.shape == (2, 2)

    def test_field_access(self):
        """FilterState fields are accessible by name."""
        x = jnp.array([3.0])
        P = jnp.array([[0.5]])
        fs = FilterState(x=x, P=P)
        assert float(fs.x[0]) == pytest.approx(3.0)
        assert float(fs.P[0, 0]) == pytest.approx(0.5)


class TestUKFConfig:
    def test_defaults(self):
        """UKFConfig has standard default values."""
        config = UKFConfig()
        assert config.alpha == pytest.approx(1.0)
        assert config.beta == pytest.approx(2.0)
        assert config.kappa == pytest.approx(0.0)

    def test_custom_values(self):
        """UKFConfig accepts custom values."""
        config = UKFConfig(alpha=0.5, beta=1.0, kappa=1.0)
        assert config.alpha == pytest.approx(0.5)
        assert config.beta == pytest.approx(1.0)
        assert config.kappa == pytest.approx(1.0)


class TestFilterResult:
    def test_construction(self):
        """FilterResult holds state and diagnostic fields."""
        fs = FilterState(x=jnp.array([1.0]), P=jnp.array([[0.1]]))
        result = FilterResult(
            state=fs,
            innovation=jnp.array([0.5]),
            innovation_covariance=jnp.array([[1.0]]),
            kalman_gain=jnp.array([[0.1]]),
        )
        assert result.state.x.shape == (1,)
        assert result.innovation.shape == (1,)
        assert result.innovation_covariance.shape == (1, 1)
        assert result.kalman_gain.shape == (1, 1)


# ──────────────────────────────────────────────
# EKF tests
# ──────────────────────────────────────────────


class TestEKFPredict:
    def test_linear_propagation(self):
        """EKF predict propagates state through linear dynamics."""
        x0 = jnp.array([1.0, 0.5])
        P0 = jnp.eye(2) * 0.01
        Q = jnp.eye(2) * 1e-6
        fs = FilterState(x=x0, P=P0)

        fs_pred = ekf_predict(fs, _linear_propagate, Q)

        # Check state: A @ x0
        expected_x = _linear_propagate(x0)
        assert jnp.allclose(fs_pred.x, expected_x, atol=1e-6)

    def test_covariance_grows(self):
        """EKF predict increases covariance through process noise."""
        x0 = jnp.array([1.0, 0.5])
        P0 = jnp.eye(2) * 0.01
        Q = jnp.eye(2) * 0.001
        fs = FilterState(x=x0, P=P0)

        fs_pred = ekf_predict(fs, _linear_propagate, Q)

        # Predicted covariance should be larger than initial
        assert jnp.all(jnp.diag(fs_pred.P) > jnp.diag(P0))

    def test_covariance_symmetric(self):
        """EKF predict produces symmetric covariance."""
        x0 = jnp.array([1.0, 0.5])
        P0 = jnp.eye(2) * 0.01
        Q = jnp.eye(2) * 1e-6
        fs = FilterState(x=x0, P=P0)

        fs_pred = ekf_predict(fs, _linear_propagate, Q)

        assert jnp.allclose(fs_pred.P, fs_pred.P.T, atol=1e-6)


class TestEKFUpdate:
    def test_measurement_reduces_uncertainty(self):
        """EKF update reduces covariance."""
        x = jnp.array([1.0, 0.5])
        P = jnp.eye(2) * 1.0
        R = jnp.array([[0.01]])
        z = jnp.array([1.1])
        fs = FilterState(x=x, P=P)

        result = ekf_update(fs, z, _linear_measurement, R)

        # Posterior variance should be less than prior
        assert float(result.state.P[0, 0]) < float(P[0, 0])

    def test_innovation_computation(self):
        """EKF update computes correct innovation."""
        x = jnp.array([1.0, 0.5])
        P = jnp.eye(2) * 0.01
        R = jnp.array([[0.01]])
        z = jnp.array([1.5])
        fs = FilterState(x=x, P=P)

        result = ekf_update(fs, z, _linear_measurement, R)

        expected_innovation = z - x[:1]
        assert jnp.allclose(result.innovation, expected_innovation, atol=1e-6)

    def test_state_moves_toward_measurement(self):
        """EKF update moves state estimate toward measurement."""
        x = jnp.array([1.0, 0.5])
        P = jnp.eye(2) * 1.0
        R = jnp.array([[0.01]])
        z = jnp.array([2.0])
        fs = FilterState(x=x, P=P)

        result = ekf_update(fs, z, _linear_measurement, R)

        # Updated x[0] should be closer to z than prior x[0]
        assert abs(float(result.state.x[0]) - 2.0) < abs(1.0 - 2.0)

    def test_joseph_form_symmetry(self):
        """EKF update produces symmetric covariance via Joseph form."""
        x = jnp.array([1.0, 0.5, -0.3])
        P = jnp.eye(3) * 0.5
        R = jnp.eye(2) * 0.1

        def measure(x):
            return x[:2]

        z = jnp.array([1.1, 0.4])
        fs = FilterState(x=x, P=P)

        result = ekf_update(fs, z, measure, R)

        assert jnp.allclose(result.state.P, result.state.P.T, atol=1e-6)

    def test_kalman_gain_shape(self):
        """EKF update returns Kalman gain with correct shape."""
        x = jnp.array([1.0, 0.5, -0.3])
        P = jnp.eye(3) * 0.5
        R = jnp.eye(2) * 0.1

        def measure(x):
            return x[:2]

        z = jnp.array([1.1, 0.4])
        fs = FilterState(x=x, P=P)

        result = ekf_update(fs, z, measure, R)

        assert result.kalman_gain.shape == (3, 2)

    def test_innovation_covariance_shape(self):
        """EKF update returns innovation covariance with correct shape."""
        x = jnp.array([1.0, 0.5])
        P = jnp.eye(2) * 0.5
        R = jnp.eye(1) * 0.1

        z = jnp.array([1.1])
        fs = FilterState(x=x, P=P)

        result = ekf_update(fs, z, _linear_measurement, R)

        assert result.innovation_covariance.shape == (1, 1)


# ──────────────────────────────────────────────
# UKF tests
# ──────────────────────────────────────────────


class TestUKFPredict:
    def test_linear_propagation(self):
        """UKF predict propagates state through linear dynamics."""
        x0 = jnp.array([1.0, 0.5])
        P0 = jnp.eye(2) * 0.01
        Q = jnp.eye(2) * 1e-6
        fs = FilterState(x=x0, P=P0)

        fs_pred = ukf_predict(fs, _linear_propagate, Q)

        expected_x = _linear_propagate(x0)
        assert jnp.allclose(fs_pred.x, expected_x, atol=1e-4)

    def test_covariance_grows(self):
        """UKF predict increases covariance through process noise."""
        x0 = jnp.array([1.0, 0.5])
        P0 = jnp.eye(2) * 0.01
        Q = jnp.eye(2) * 0.001
        fs = FilterState(x=x0, P=P0)

        fs_pred = ukf_predict(fs, _linear_propagate, Q)

        assert jnp.all(jnp.diag(fs_pred.P) > jnp.diag(P0))

    def test_covariance_symmetric(self):
        """UKF predict produces symmetric covariance."""
        x0 = jnp.array([1.0, 0.5])
        P0 = jnp.eye(2) * 0.01
        Q = jnp.eye(2) * 1e-6
        fs = FilterState(x=x0, P=P0)

        fs_pred = ukf_predict(fs, _linear_propagate, Q)

        assert jnp.allclose(fs_pred.P, fs_pred.P.T, atol=1e-6)


class TestUKFUpdate:
    def test_measurement_reduces_uncertainty(self):
        """UKF update reduces covariance."""
        x = jnp.array([1.0, 0.5])
        P = jnp.eye(2) * 1.0
        R = jnp.array([[0.01]])
        z = jnp.array([1.1])
        fs = FilterState(x=x, P=P)

        result = ukf_update(fs, z, _linear_measurement, R)

        assert float(result.state.P[0, 0]) < float(P[0, 0])

    def test_innovation_computation(self):
        """UKF update computes correct innovation."""
        x = jnp.array([1.0, 0.5])
        P = jnp.eye(2) * 0.01
        R = jnp.array([[0.01]])
        z = jnp.array([1.5])
        fs = FilterState(x=x, P=P)

        result = ukf_update(fs, z, _linear_measurement, R)

        expected_innovation = z - x[:1]
        assert jnp.allclose(result.innovation, expected_innovation, atol=1e-4)

    def test_state_moves_toward_measurement(self):
        """UKF update moves state estimate toward measurement."""
        x = jnp.array([1.0, 0.5])
        P = jnp.eye(2) * 1.0
        R = jnp.array([[0.01]])
        z = jnp.array([2.0])
        fs = FilterState(x=x, P=P)

        result = ukf_update(fs, z, _linear_measurement, R)

        assert abs(float(result.state.x[0]) - 2.0) < abs(1.0 - 2.0)

    def test_kalman_gain_shape(self):
        """UKF update returns Kalman gain with correct shape."""
        x = jnp.array([1.0, 0.5, -0.3])
        P = jnp.eye(3) * 0.5
        R = jnp.eye(2) * 0.1

        def measure(x):
            return x[:2]

        z = jnp.array([1.1, 0.4])
        fs = FilterState(x=x, P=P)

        result = ukf_update(fs, z, measure, R)

        assert result.kalman_gain.shape == (3, 2)


# ──────────────────────────────────────────────
# EKF vs UKF consistency on linear system
# ──────────────────────────────────────────────


class TestEKFvsUKF:
    def test_predict_agrees_on_linear_system(self):
        """EKF and UKF predict produce similar results on linear dynamics."""
        x0 = jnp.array([1.0, 0.5])
        P0 = jnp.eye(2) * 0.01
        Q = jnp.eye(2) * 1e-6
        fs = FilterState(x=x0, P=P0)

        ekf_pred = ekf_predict(fs, _linear_propagate, Q)
        ukf_pred = ukf_predict(fs, _linear_propagate, Q)

        assert jnp.allclose(ekf_pred.x, ukf_pred.x, atol=1e-4)
        assert jnp.allclose(ekf_pred.P, ukf_pred.P, atol=1e-3)

    def test_update_agrees_on_linear_system(self):
        """EKF and UKF update produce similar results on linear measurement."""
        x = jnp.array([1.0, 0.5])
        P = jnp.eye(2) * 0.5
        R = jnp.array([[0.1]])
        z = jnp.array([1.2])
        fs = FilterState(x=x, P=P)

        ekf_result = ekf_update(fs, z, _linear_measurement, R)
        ukf_result = ukf_update(fs, z, _linear_measurement, R)

        assert jnp.allclose(ekf_result.state.x, ukf_result.state.x, atol=1e-3)
        assert jnp.allclose(ekf_result.innovation, ukf_result.innovation, atol=1e-4)


# ──────────────────────────────────────────────
# GPS measurement model tests
# ──────────────────────────────────────────────


class TestGNSSMeasurements:
    def test_gnss_position_measurement(self):
        """GNSS position measurement extracts first 3 components."""
        state = jnp.array([6878e3, 100.0, -50.0, 0.0, 7612.0, 0.0])
        z = gnss_position_measurement(state)
        assert z.shape == (3,)
        assert jnp.allclose(z, state[:3])

    def test_gnss_measurement_noise(self):
        """GNSS position noise is sigma^2 * I_3."""
        R = gnss_measurement_noise(10.0)
        assert R.shape == (3, 3)
        assert jnp.allclose(R, 100.0 * jnp.eye(3), atol=1e-6)

    def test_gnss_position_velocity_measurement(self):
        """GNSS position-velocity measurement extracts first 6 components."""
        state = jnp.array([6878e3, 100.0, -50.0, 0.0, 7612.0, 0.0])
        z = gnss_position_velocity_measurement(state)
        assert z.shape == (6,)
        assert jnp.allclose(z, state[:6])

    def test_gnss_position_velocity_noise(self):
        """GNSS position-velocity noise has correct diagonal structure."""
        R = gnss_position_velocity_noise(10.0, 0.1)
        assert R.shape == (6, 6)
        # Position block
        assert float(R[0, 0]) == pytest.approx(100.0)
        assert float(R[1, 1]) == pytest.approx(100.0)
        assert float(R[2, 2]) == pytest.approx(100.0)
        # Velocity block
        assert float(R[3, 3]) == pytest.approx(0.01)
        assert float(R[4, 4]) == pytest.approx(0.01)
        assert float(R[5, 5]) == pytest.approx(0.01)
        # Off-diagonal zeros
        assert float(R[0, 3]) == pytest.approx(0.0)

    def test_gnss_measurement_noise_zero_sigma(self):
        """GNSS noise with zero sigma produces zero matrix."""
        R = gnss_measurement_noise(0.0)
        assert jnp.allclose(R, jnp.zeros((3, 3)), atol=1e-10)


# ──────────────────────────────────────────────
# JAX compatibility tests
# ──────────────────────────────────────────────


class TestJAXCompatibility:
    def test_jit_ekf_predict(self):
        """ekf_predict is JIT-compilable."""
        x0 = jnp.array([1.0, 0.5])
        P0 = jnp.eye(2) * 0.01
        Q = jnp.eye(2) * 1e-6

        @jax.jit
        def step(x, P, Q):
            fs = FilterState(x=x, P=P)
            return ekf_predict(fs, _linear_propagate, Q)

        result = step(x0, P0, Q)
        assert jnp.all(jnp.isfinite(result.x))
        assert jnp.all(jnp.isfinite(result.P))

    def test_jit_ekf_update(self):
        """ekf_update is JIT-compilable."""
        x = jnp.array([1.0, 0.5])
        P = jnp.eye(2) * 0.5
        R = jnp.array([[0.1]])
        z = jnp.array([1.2])

        @jax.jit
        def step(x, P, z, R):
            fs = FilterState(x=x, P=P)
            return ekf_update(fs, z, _linear_measurement, R)

        result = step(x, P, z, R)
        assert jnp.all(jnp.isfinite(result.state.x))

    def test_jit_ukf_predict(self):
        """ukf_predict is JIT-compilable."""
        x0 = jnp.array([1.0, 0.5])
        P0 = jnp.eye(2) * 0.01
        Q = jnp.eye(2) * 1e-6

        @jax.jit
        def step(x, P, Q):
            fs = FilterState(x=x, P=P)
            return ukf_predict(fs, _linear_propagate, Q)

        result = step(x0, P0, Q)
        assert jnp.all(jnp.isfinite(result.x))

    def test_jit_ukf_update(self):
        """ukf_update is JIT-compilable."""
        x = jnp.array([1.0, 0.5])
        P = jnp.eye(2) * 0.5
        R = jnp.array([[0.1]])
        z = jnp.array([1.2])

        @jax.jit
        def step(x, P, z, R):
            fs = FilterState(x=x, P=P)
            return ukf_update(fs, z, _linear_measurement, R)

        result = step(x, P, z, R)
        assert jnp.all(jnp.isfinite(result.state.x))

    def test_lax_scan_ekf(self):
        """EKF predict+update composes with jax.lax.scan."""
        x0 = jnp.array([1.0, 0.5])
        P0 = jnp.eye(2) * 1.0
        Q = jnp.eye(2) * 1e-4
        R = jnp.array([[0.01]])
        fs0 = FilterState(x=x0, P=P0)

        # Generate synthetic measurements from true state
        true_x = jnp.array([1.0, 0.5])
        n_steps = 10
        measurements = jnp.full((n_steps, 1), true_x[0])  # perfect position obs

        def filter_step(fs, z):
            fs = ekf_predict(fs, _linear_propagate, Q)
            result = ekf_update(fs, z, _linear_measurement, R)
            return result.state, result.innovation

        final_state, innovations = jax.lax.scan(filter_step, fs0, measurements)

        assert innovations.shape == (n_steps, 1)
        assert jnp.all(jnp.isfinite(final_state.x))
        # Uncertainty should decrease after multiple updates
        assert float(final_state.P[0, 0]) < float(P0[0, 0])

    def test_lax_scan_ukf(self):
        """UKF predict+update composes with jax.lax.scan."""
        x0 = jnp.array([1.0, 0.5])
        P0 = jnp.eye(2) * 1.0
        Q = jnp.eye(2) * 1e-4
        R = jnp.array([[0.01]])
        fs0 = FilterState(x=x0, P=P0)

        n_steps = 10
        measurements = jnp.full((n_steps, 1), 1.0)

        def filter_step(fs, z):
            fs = ukf_predict(fs, _linear_propagate, Q)
            result = ukf_update(fs, z, _linear_measurement, R)
            return result.state, result.innovation

        final_state, innovations = jax.lax.scan(filter_step, fs0, measurements)

        assert innovations.shape == (n_steps, 1)
        assert jnp.all(jnp.isfinite(final_state.x))
        assert float(final_state.P[0, 0]) < float(P0[0, 0])

    def test_grad_ekf(self):
        """EKF predict supports gradient computation."""

        def loss(x0):
            P0 = jnp.eye(2) * 0.01
            Q = jnp.eye(2) * 1e-6
            fs = FilterState(x=x0, P=P0)
            fs_pred = ekf_predict(fs, _linear_propagate, Q)
            return jnp.sum(fs_pred.x**2)

        x0 = jnp.array([1.0, 0.5])
        grad = jax.grad(loss)(x0)
        assert grad.shape == (2,)
        assert jnp.all(jnp.isfinite(grad))


# ──────────────────────────────────────────────
# End-to-end orbit determination tests
# ──────────────────────────────────────────────


class TestOrbitDetermination:
    def test_ekf_gnss_orbit_determination(self):
        """EKF converges on orbit with simulated GNSS position measurements."""
        # True orbit state
        true_state = _circular_orbit_state()

        # Initial estimate with 1 km position error
        x0 = true_state + jnp.array([1000.0, -500.0, 200.0, 0.0, 0.0, 0.0])
        P0 = jnp.diag(jnp.array([1e6, 1e6, 1e6, 1e2, 1e2, 1e2]))
        Q = jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]))
        R = gnss_measurement_noise(10.0)  # 10 m GNSS noise
        fs = FilterState(x=x0, P=P0)

        # Simulate 50 measurement updates (no actual propagation, just updates)
        key = jax.random.PRNGKey(42)
        n_steps = 50

        for _i in range(n_steps):
            # Propagate
            fs = ekf_predict(fs, _two_body_propagate, Q)

            # Generate noisy measurement from propagated true state
            true_state = _two_body_propagate(true_state)
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (3,)) * 10.0
            z = true_state[:3] + noise

            # Update
            result = ekf_update(fs, z, gnss_position_measurement, R)
            fs = result.state

        # Position error should be significantly reduced
        pos_error = jnp.linalg.norm(fs.x[:3] - true_state[:3])
        assert float(pos_error) < 500.0  # Should converge to < 500 m

    def test_ukf_gnss_orbit_determination(self):
        """UKF converges on orbit with simulated GNSS position measurements."""
        true_state = _circular_orbit_state()

        x0 = true_state + jnp.array([1000.0, -500.0, 200.0, 0.0, 0.0, 0.0])
        P0 = jnp.diag(jnp.array([1e6, 1e6, 1e6, 1e2, 1e2, 1e2]))
        Q = jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]))
        R = gnss_measurement_noise(10.0)
        fs = FilterState(x=x0, P=P0)

        key = jax.random.PRNGKey(42)
        n_steps = 50

        for _i in range(n_steps):
            fs = ukf_predict(fs, _two_body_propagate, Q)

            true_state = _two_body_propagate(true_state)
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (3,)) * 10.0
            z = true_state[:3] + noise

            result = ukf_update(fs, z, gnss_position_measurement, R)
            fs = result.state

        pos_error = jnp.linalg.norm(fs.x[:3] - true_state[:3])
        assert float(pos_error) < 500.0

    def test_ekf_lax_scan_orbit_determination(self):
        """EKF orbit determination works inside jax.lax.scan."""
        true_state_0 = _circular_orbit_state()

        x0 = true_state_0 + jnp.array([500.0, -300.0, 100.0, 0.0, 0.0, 0.0])
        P0 = jnp.diag(jnp.array([1e6, 1e6, 1e6, 1e2, 1e2, 1e2]))
        Q = jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]))
        R = gnss_measurement_noise(10.0)

        # Pre-generate truth trajectory and noisy measurements
        n_steps = 20
        key = jax.random.PRNGKey(123)
        true_states = [true_state_0]
        for _ in range(n_steps):
            true_states.append(_two_body_propagate(true_states[-1]))
        true_states = jnp.stack(true_states[1:])  # (n_steps, 6)

        keys = jax.random.split(key, n_steps)
        noise = jax.vmap(lambda k: jax.random.normal(k, (3,)) * 10.0)(keys)
        measurements = true_states[:, :3] + noise  # (n_steps, 3)

        fs0 = FilterState(x=x0, P=P0)

        def filter_step(fs, z):
            fs = ekf_predict(fs, _two_body_propagate, Q)
            result = ekf_update(fs, z, gnss_position_measurement, R)
            return result.state, result.innovation

        final_state, innovations = jax.lax.scan(filter_step, fs0, measurements)

        assert innovations.shape == (n_steps, 3)
        assert jnp.all(jnp.isfinite(final_state.x))
        # Covariance should decrease
        assert float(final_state.P[0, 0]) < float(P0[0, 0])
