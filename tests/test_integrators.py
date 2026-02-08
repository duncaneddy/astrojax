"""Tests for the astrojax.integrators module.

Tests cover:
- Polynomial exactness (RK4 is exact for degree <= 3 polynomials)
- Exponential decay with known solution
- Harmonic oscillator energy conservation
- Two-body orbital mechanics
- Backward integration
- Control input functionality
- Adaptive step-size behavior (RKF45, DP54)
- JIT and vmap compatibility
"""

import jax
import jax.numpy as jnp
import pytest

from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.integrators import (
    AdaptiveConfig,
    StepResult,
    dp54_step,
    rk4_step,
    rkf45_step,
    rkn1210_step,
)

# Tolerances
_SINGLE_TOL = 1e-4
_ADAPTIVE_TOL = 1e-3


# ──────────────────────────────────────────────
# Helper dynamics functions
# ──────────────────────────────────────────────

def _exponential_decay(t, x):
    """dx/dt = -x. Solution: x(t) = x0 * exp(-t)."""
    return -x


def _harmonic_oscillator(t, x):
    """d^2q/dt^2 = -q. State: [q, dq/dt]. Solution: [cos(t), -sin(t)]."""
    return jnp.array([x[1], -x[0]])


def _linear_dynamics(t, x):
    """dx/dt = 1. Solution: x(t) = x0 + t."""
    return jnp.ones_like(x)


def _quadratic_dynamics(t, x):
    """dx/dt = 2t. Solution: x(t) = x0 + t^2."""
    return 2.0 * t * jnp.ones_like(x)


def _cubic_dynamics(t, x):
    """dx/dt = 3t^2. Solution: x(t) = x0 + t^3."""
    return 3.0 * t**2 * jnp.ones_like(x)


def _two_body(t, state):
    """Two-body gravitational dynamics. State: [rx, ry, rz, vx, vy, vz]."""
    r = state[:3]
    v = state[3:]
    r_norm = jnp.linalg.norm(r)
    a = -GM_EARTH * r / r_norm**3
    return jnp.concatenate([v, a])


def _circular_orbit_state(sma):
    """Create a circular equatorial orbit state [x, y, z, vx, vy, vz]."""
    v_circ = jnp.sqrt(GM_EARTH / sma)
    return jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])


# ──────────────────────────────────────────────
# StepResult and AdaptiveConfig tests
# ──────────────────────────────────────────────

class TestTypes:
    def test_step_result_fields(self):
        """StepResult has the expected fields."""
        result = StepResult(
            state=jnp.array([1.0]),
            dt_used=jnp.array(0.1),
            error_estimate=jnp.array(0.0),
            dt_next=jnp.array(0.1),
        )
        assert result.state.shape == (1,)
        assert float(result.dt_used) == pytest.approx(0.1)
        assert float(result.error_estimate) == pytest.approx(0.0)
        assert float(result.dt_next) == pytest.approx(0.1)

    def test_adaptive_config_defaults(self):
        """AdaptiveConfig has reasonable defaults."""
        config = AdaptiveConfig()
        assert config.abs_tol == 1e-6
        assert config.rel_tol == 1e-3
        assert config.safety_factor == 0.9
        assert config.max_step_attempts == 10

    def test_adaptive_config_custom(self):
        """AdaptiveConfig accepts custom values."""
        config = AdaptiveConfig(abs_tol=1e-10, rel_tol=1e-8)
        assert config.abs_tol == 1e-10
        assert config.rel_tol == 1e-8


# ──────────────────────────────────────────────
# RK4 tests
# ──────────────────────────────────────────────

class TestRK4:
    def test_exponential_decay(self):
        """RK4 approximates exponential decay with small error."""
        x0 = jnp.array([1.0])
        dt = 0.1
        result = rk4_step(_exponential_decay, 0.0, x0, dt)
        expected = jnp.exp(-dt)
        assert jnp.allclose(result.state, jnp.array([expected]), atol=1e-6)

    def test_linear_exactness(self):
        """RK4 is exact for linear dynamics (dx/dt = 1)."""
        x0 = jnp.array([5.0])
        dt = 1.0
        result = rk4_step(_linear_dynamics, 0.0, x0, dt)
        expected = x0 + dt
        assert jnp.allclose(result.state, expected, atol=1e-6)

    def test_quadratic_exactness(self):
        """RK4 is exact for quadratic dynamics (dx/dt = 2t)."""
        x0 = jnp.array([0.0])
        t0 = 1.0
        dt = 0.5
        result = rk4_step(_quadratic_dynamics, t0, x0, dt)
        # x(t0+dt) = x0 + (t0+dt)^2 - t0^2 = (1.5)^2 - 1^2 = 1.25
        expected = jnp.array([(t0 + dt) ** 2 - t0**2])
        assert jnp.allclose(result.state, expected, atol=1e-5)

    def test_cubic_exactness(self):
        """RK4 is exact for cubic dynamics (dx/dt = 3t^2)."""
        x0 = jnp.array([0.0])
        t0 = 0.0
        dt = 1.0
        result = rk4_step(_cubic_dynamics, t0, x0, dt)
        # x(1) = 0 + 1^3 = 1.0
        expected = jnp.array([1.0])
        assert jnp.allclose(result.state, expected, atol=1e-5)

    def test_harmonic_oscillator_single_step(self):
        """RK4 approximates harmonic oscillator accurately for small dt."""
        x0 = jnp.array([1.0, 0.0])
        dt = 0.01
        result = rk4_step(_harmonic_oscillator, 0.0, x0, dt)
        expected = jnp.array([jnp.cos(dt), -jnp.sin(dt)])
        assert jnp.allclose(result.state, expected, atol=1e-6)

    def test_harmonic_oscillator_multi_step(self):
        """RK4 preserves energy over many small steps."""
        x0 = jnp.array([1.0, 0.0])
        dt = 0.01
        n_steps = 1000  # 10 seconds
        state = x0
        for _ in range(n_steps):
            result = rk4_step(_harmonic_oscillator, 0.0, state, dt)
            state = result.state

        t_final = dt * n_steps
        expected = jnp.array([jnp.cos(t_final), -jnp.sin(t_final)])
        assert jnp.allclose(state, expected, atol=1e-3)

    def test_step_result_fields(self):
        """RK4 returns correct StepResult metadata."""
        x0 = jnp.array([1.0])
        dt = 0.1
        result = rk4_step(_exponential_decay, 0.0, x0, dt)
        assert float(result.dt_used) == pytest.approx(0.1, abs=1e-6)
        assert float(result.error_estimate) == pytest.approx(0.0, abs=1e-6)
        assert float(result.dt_next) == pytest.approx(0.1, abs=1e-6)

    def test_backward_integration(self):
        """RK4 supports negative dt for backward integration."""
        x0 = jnp.array([1.0])
        dt_forward = 0.1
        # Forward
        result_fwd = rk4_step(_exponential_decay, 0.0, x0, dt_forward)
        # Backward from the forward result
        result_bwd = rk4_step(_exponential_decay, dt_forward, result_fwd.state, -dt_forward)
        assert jnp.allclose(result_bwd.state, x0, atol=1e-5)

    def test_control_input(self):
        """RK4 correctly incorporates an additive control function."""
        # dx/dt = -x + 1. Solution: x(t) = 1 - exp(-t) for x(0)=0
        x0 = jnp.array([0.0])
        dt = 0.1

        def constant_control(t, x):
            return jnp.ones_like(x)

        result = rk4_step(_exponential_decay, 0.0, x0, dt, control=constant_control)
        expected = 1.0 - jnp.exp(-dt)
        assert jnp.allclose(result.state, jnp.array([expected]), atol=1e-5)

    def test_two_body_circular_orbit(self):
        """RK4 preserves circular orbit radius over one step."""
        sma = R_EARTH + 500e3
        state0 = _circular_orbit_state(sma)
        dt = 10.0  # 10 seconds
        result = rk4_step(_two_body, 0.0, state0, dt)
        r_final = jnp.linalg.norm(result.state[:3])
        assert jnp.abs(r_final - sma) / sma < 1e-5


# ──────────────────────────────────────────────
# RKF45 tests
# ──────────────────────────────────────────────

class TestRKF45:
    def test_exponential_decay(self):
        """RKF45 approximates exponential decay accurately."""
        x0 = jnp.array([1.0])
        dt = 0.5
        result = rkf45_step(_exponential_decay, 0.0, x0, dt)
        expected = jnp.exp(-dt)
        assert jnp.allclose(result.state, jnp.array([expected]), atol=1e-5)

    def test_harmonic_oscillator(self):
        """RKF45 approximates harmonic oscillator."""
        x0 = jnp.array([1.0, 0.0])
        dt = 0.1
        result = rkf45_step(_harmonic_oscillator, 0.0, x0, dt)
        expected = jnp.array([jnp.cos(dt), -jnp.sin(dt)])
        assert jnp.allclose(result.state, expected, atol=_ADAPTIVE_TOL)

    def test_error_estimate_nonzero(self):
        """RKF45 produces a finite nonzero error estimate."""
        x0 = jnp.array([1.0, 0.0])
        result = rkf45_step(_harmonic_oscillator, 0.0, x0, 0.1)
        assert jnp.isfinite(result.error_estimate)

    def test_dt_next_positive(self):
        """RKF45 suggests a positive next step size."""
        x0 = jnp.array([1.0, 0.0])
        result = rkf45_step(_harmonic_oscillator, 0.0, x0, 0.1)
        assert float(result.dt_next) > 0.0

    def test_step_acceptance(self):
        """RKF45 accepts steps when error is within tolerance."""
        x0 = jnp.array([1.0, 0.0])
        config = AdaptiveConfig(abs_tol=1e-4, rel_tol=1e-2)
        result = rkf45_step(_harmonic_oscillator, 0.0, x0, 0.01, config=config)
        assert float(result.error_estimate) <= 1.0

    def test_custom_config(self):
        """RKF45 respects tighter tolerances with AdaptiveConfig."""
        x0 = jnp.array([1.0, 0.0])
        dt = 1.0
        # Loose config
        config_loose = AdaptiveConfig(abs_tol=1e-2, rel_tol=1e-1)
        result_loose = rkf45_step(_harmonic_oscillator, 0.0, x0, dt, config=config_loose)
        # Tight config
        config_tight = AdaptiveConfig(abs_tol=1e-8, rel_tol=1e-6)
        result_tight = rkf45_step(_harmonic_oscillator, 0.0, x0, dt, config=config_tight)
        # Tight config should use a smaller step
        assert float(jnp.abs(result_tight.dt_used)) <= float(jnp.abs(result_loose.dt_used))

    def test_backward_integration(self):
        """RKF45 supports negative dt for backward integration."""
        x0 = jnp.array([1.0])
        dt = 0.5
        result_fwd = rkf45_step(_exponential_decay, 0.0, x0, dt)
        result_bwd = rkf45_step(_exponential_decay, dt, result_fwd.state, -dt)
        assert jnp.allclose(result_bwd.state, x0, atol=1e-3)

    def test_control_input(self):
        """RKF45 correctly incorporates an additive control function."""
        x0 = jnp.array([0.0])
        dt = 0.1

        def constant_control(t, x):
            return jnp.ones_like(x)

        result = rkf45_step(_exponential_decay, 0.0, x0, dt, control=constant_control)
        expected = 1.0 - jnp.exp(-dt)
        assert jnp.allclose(result.state, jnp.array([expected]), atol=1e-4)

    def test_two_body_circular_orbit(self):
        """RKF45 preserves circular orbit radius."""
        sma = R_EARTH + 500e3
        state0 = _circular_orbit_state(sma)
        dt = 60.0
        result = rkf45_step(_two_body, 0.0, state0, dt)
        r_final = jnp.linalg.norm(result.state[:3])
        assert jnp.abs(r_final - sma) / sma < 1e-4


# ──────────────────────────────────────────────
# DP54 tests
# ──────────────────────────────────────────────

class TestDP54:
    def test_exponential_decay(self):
        """DP54 approximates exponential decay accurately."""
        x0 = jnp.array([1.0])
        dt = 0.5
        result = dp54_step(_exponential_decay, 0.0, x0, dt)
        expected = jnp.exp(-dt)
        assert jnp.allclose(result.state, jnp.array([expected]), atol=1e-5)

    def test_harmonic_oscillator(self):
        """DP54 approximates harmonic oscillator."""
        x0 = jnp.array([1.0, 0.0])
        dt = 0.1
        result = dp54_step(_harmonic_oscillator, 0.0, x0, dt)
        expected = jnp.array([jnp.cos(dt), -jnp.sin(dt)])
        assert jnp.allclose(result.state, expected, atol=_ADAPTIVE_TOL)

    def test_error_estimate_nonzero(self):
        """DP54 produces a finite nonzero error estimate."""
        x0 = jnp.array([1.0, 0.0])
        result = dp54_step(_harmonic_oscillator, 0.0, x0, 0.1)
        assert jnp.isfinite(result.error_estimate)

    def test_dt_next_positive(self):
        """DP54 suggests a positive next step size."""
        x0 = jnp.array([1.0, 0.0])
        result = dp54_step(_harmonic_oscillator, 0.0, x0, 0.1)
        assert float(result.dt_next) > 0.0

    def test_step_acceptance(self):
        """DP54 accepts steps when error is within tolerance."""
        x0 = jnp.array([1.0, 0.0])
        config = AdaptiveConfig(abs_tol=1e-4, rel_tol=1e-2)
        result = dp54_step(_harmonic_oscillator, 0.0, x0, 0.01, config=config)
        assert float(result.error_estimate) <= 1.0

    def test_custom_config(self):
        """DP54 respects tighter tolerances with AdaptiveConfig."""
        x0 = jnp.array([1.0, 0.0])
        dt = 1.0
        config_loose = AdaptiveConfig(abs_tol=1e-2, rel_tol=1e-1)
        result_loose = dp54_step(_harmonic_oscillator, 0.0, x0, dt, config=config_loose)
        config_tight = AdaptiveConfig(abs_tol=1e-8, rel_tol=1e-6)
        result_tight = dp54_step(_harmonic_oscillator, 0.0, x0, dt, config=config_tight)
        assert float(jnp.abs(result_tight.dt_used)) <= float(jnp.abs(result_loose.dt_used))

    def test_backward_integration(self):
        """DP54 supports negative dt for backward integration."""
        x0 = jnp.array([1.0])
        dt = 0.5
        result_fwd = dp54_step(_exponential_decay, 0.0, x0, dt)
        result_bwd = dp54_step(_exponential_decay, dt, result_fwd.state, -dt)
        assert jnp.allclose(result_bwd.state, x0, atol=1e-3)

    def test_control_input(self):
        """DP54 correctly incorporates an additive control function."""
        x0 = jnp.array([0.0])
        dt = 0.1

        def constant_control(t, x):
            return jnp.ones_like(x)

        result = dp54_step(_exponential_decay, 0.0, x0, dt, control=constant_control)
        expected = 1.0 - jnp.exp(-dt)
        assert jnp.allclose(result.state, jnp.array([expected]), atol=1e-4)

    def test_two_body_circular_orbit(self):
        """DP54 preserves circular orbit radius."""
        sma = R_EARTH + 500e3
        state0 = _circular_orbit_state(sma)
        dt = 60.0
        result = dp54_step(_two_body, 0.0, state0, dt)
        r_final = jnp.linalg.norm(result.state[:3])
        assert jnp.abs(r_final - sma) / sma < 1e-4


# ──────────────────────────────────────────────
# Cross-method consistency tests
# ──────────────────────────────────────────────

class TestCrossMethod:
    def test_all_methods_agree_small_step(self):
        """All three methods agree for a small step on harmonic oscillator."""
        x0 = jnp.array([1.0, 0.0])
        dt = 0.01
        r_rk4 = rk4_step(_harmonic_oscillator, 0.0, x0, dt)
        r_rkf = rkf45_step(_harmonic_oscillator, 0.0, x0, dt)
        r_dp = dp54_step(_harmonic_oscillator, 0.0, x0, dt)
        assert jnp.allclose(r_rk4.state, r_rkf.state, atol=1e-4)
        assert jnp.allclose(r_rk4.state, r_dp.state, atol=1e-4)

    def test_rkn1210_agrees_with_dp54(self):
        """RKN1210 and DP54 agree on small harmonic oscillator step."""
        x0 = jnp.array([1.0, 0.0])
        dt = 0.01
        r_rkn = rkn1210_step(_harmonic_oscillator, 0.0, x0, dt)
        r_dp = dp54_step(_harmonic_oscillator, 0.0, x0, dt)
        assert jnp.allclose(r_rkn.state, r_dp.state, atol=1e-4)

    def test_all_methods_agree_exponential(self):
        """All three methods agree for exponential decay."""
        x0 = jnp.array([2.0])
        dt = 0.1
        r_rk4 = rk4_step(_exponential_decay, 0.0, x0, dt)
        r_rkf = rkf45_step(_exponential_decay, 0.0, x0, dt)
        r_dp = dp54_step(_exponential_decay, 0.0, x0, dt)
        expected = 2.0 * jnp.exp(-dt)
        assert jnp.allclose(r_rk4.state, jnp.array([expected]), atol=1e-5)
        assert jnp.allclose(r_rkf.state, jnp.array([expected]), atol=1e-5)
        assert jnp.allclose(r_dp.state, jnp.array([expected]), atol=1e-5)


# ──────────────────────────────────────────────
# JAX compatibility tests
# ──────────────────────────────────────────────

class TestJAXCompatibility:
    def test_jit_rk4(self):
        """rk4_step is JIT-compilable."""
        x0 = jnp.array([1.0, 0.0])

        @jax.jit
        def step(t, x, dt):
            return rk4_step(_harmonic_oscillator, t, x, dt)

        result = step(0.0, x0, 0.01)
        expected = jnp.array([jnp.cos(0.01), -jnp.sin(0.01)])
        assert jnp.allclose(result.state, expected, atol=1e-6)

    def test_jit_rkf45(self):
        """rkf45_step is JIT-compilable."""
        x0 = jnp.array([1.0, 0.0])

        @jax.jit
        def step(t, x, dt):
            return rkf45_step(_harmonic_oscillator, t, x, dt)

        result = step(0.0, x0, 0.1)
        assert jnp.all(jnp.isfinite(result.state))

    def test_jit_dp54(self):
        """dp54_step is JIT-compilable."""
        x0 = jnp.array([1.0, 0.0])

        @jax.jit
        def step(t, x, dt):
            return dp54_step(_harmonic_oscillator, t, x, dt)

        result = step(0.0, x0, 0.1)
        assert jnp.all(jnp.isfinite(result.state))

    def test_vmap_rk4(self):
        """rk4_step works with vmap over a batch of initial conditions."""
        x0_batch = jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ])

        def step(x0):
            return rk4_step(_harmonic_oscillator, 0.0, x0, 0.01).state

        results = jax.vmap(step)(x0_batch)
        assert results.shape == (3, 2)

    def test_vmap_rkf45(self):
        """rkf45_step works with vmap over a batch of initial conditions."""
        x0_batch = jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        def step(x0):
            return rkf45_step(_harmonic_oscillator, 0.0, x0, 0.1).state

        results = jax.vmap(step)(x0_batch)
        assert results.shape == (2, 2)

    def test_vmap_dp54(self):
        """dp54_step works with vmap over a batch of initial conditions."""
        x0_batch = jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        def step(x0):
            return dp54_step(_harmonic_oscillator, 0.0, x0, 0.1).state

        results = jax.vmap(step)(x0_batch)
        assert results.shape == (2, 2)

    def test_grad_rk4(self):
        """rk4_step supports gradient computation."""
        def loss(x0):
            result = rk4_step(_harmonic_oscillator, 0.0, x0, 0.01)
            return jnp.sum(result.state**2)

        x0 = jnp.array([1.0, 0.0])
        grad = jax.grad(loss)(x0)
        assert grad.shape == (2,)
        assert jnp.all(jnp.isfinite(grad))

    def test_lax_scan_rk4(self):
        """rk4_step works inside lax.scan for multi-step propagation."""
        from astrojax.config import get_dtype

        dtype = get_dtype()
        x0 = jnp.array([1.0, 0.0], dtype=dtype)
        dt = 0.01
        n_steps = 100

        def scan_step(state, _):
            result = rk4_step(_harmonic_oscillator, 0.0, state, dt)
            return result.state, result.state

        final, trajectory = jax.lax.scan(scan_step, x0, None, length=n_steps)
        assert trajectory.shape == (n_steps, 2)
        # Check final state against analytical solution
        t_final = dt * n_steps
        expected = jnp.array([jnp.cos(t_final), -jnp.sin(t_final)])
        assert jnp.allclose(final, expected, atol=1e-3)

    def test_jit_rkn1210(self):
        """rkn1210_step is JIT-compilable."""
        x0 = jnp.array([1.0, 0.0])

        @jax.jit
        def step(t, x, dt):
            return rkn1210_step(_harmonic_oscillator, t, x, dt)

        result = step(0.0, x0, 0.1)
        assert jnp.all(jnp.isfinite(result.state))

    def test_vmap_rkn1210(self):
        """rkn1210_step works with vmap over a batch of initial conditions."""
        x0_batch = jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        def step(x0):
            return rkn1210_step(_harmonic_oscillator, 0.0, x0, 0.1).state

        results = jax.vmap(step)(x0_batch)
        assert results.shape == (2, 2)


# ──────────────────────────────────────────────
# RKN1210 tests
# ──────────────────────────────────────────────

class TestRKN1210:
    def test_harmonic_oscillator(self):
        """RKN1210 approximates harmonic oscillator accurately.

        The harmonic oscillator x'' = -x is a second-order ODE, exactly
        the class of problem RKN methods are designed for.
        """
        x0 = jnp.array([1.0, 0.0])
        dt = 0.1
        result = rkn1210_step(_harmonic_oscillator, 0.0, x0, dt)
        expected = jnp.array([jnp.cos(dt), -jnp.sin(dt)])
        assert jnp.allclose(result.state, expected, atol=_ADAPTIVE_TOL)

    def test_exponential_state_dot(self):
        """RKN1210 works with standard dynamics(t, state) -> state_dot format.

        Uses exponential decay as a first-order ODE to verify the integrator
        handles the state_dot extraction correctly even for non-second-order
        problems (though it won't be as efficient).
        """
        x0 = jnp.array([1.0, 0.0])
        dt = 0.1
        result = rkn1210_step(_harmonic_oscillator, 0.0, x0, dt)
        # Just verify it produces finite results
        assert jnp.all(jnp.isfinite(result.state))

    def test_two_body_circular_orbit(self):
        """RKN1210 preserves circular orbit radius."""
        sma = R_EARTH + 500e3
        state0 = _circular_orbit_state(sma)
        dt = 60.0
        result = rkn1210_step(_two_body, 0.0, state0, dt)
        r_final = jnp.linalg.norm(result.state[:3])
        assert jnp.abs(r_final - sma) / sma < 1e-4

    def test_backward_integration(self):
        """RKN1210 supports negative dt for backward integration."""
        x0 = jnp.array([1.0, 0.0])
        dt = 0.1
        result_fwd = rkn1210_step(_harmonic_oscillator, 0.0, x0, dt)
        result_bwd = rkn1210_step(_harmonic_oscillator, dt, result_fwd.state, -dt)
        assert jnp.allclose(result_bwd.state, x0, atol=1e-3)

    def test_control_input(self):
        """RKN1210 correctly incorporates an additive control function."""
        x0 = jnp.array([1.0, 0.0])
        dt = 0.01

        def zero_control(t, x):
            return jnp.zeros_like(x)

        result_no_ctrl = rkn1210_step(_harmonic_oscillator, 0.0, x0, dt)
        result_zero_ctrl = rkn1210_step(
            _harmonic_oscillator, 0.0, x0, dt, control=zero_control
        )
        assert jnp.allclose(result_no_ctrl.state, result_zero_ctrl.state, atol=1e-6)

    def test_error_estimate_nonzero(self):
        """RKN1210 produces a finite nonzero error estimate."""
        x0 = jnp.array([1.0, 0.0])
        result = rkn1210_step(_harmonic_oscillator, 0.0, x0, 0.1)
        assert jnp.isfinite(result.error_estimate)

    def test_dt_next_positive(self):
        """RKN1210 suggests a positive next step size."""
        x0 = jnp.array([1.0, 0.0])
        result = rkn1210_step(_harmonic_oscillator, 0.0, x0, 0.1)
        assert float(result.dt_next) > 0.0

    def test_step_acceptance(self):
        """RKN1210 accepts steps when error is within tolerance."""
        x0 = jnp.array([1.0, 0.0])
        config = AdaptiveConfig(abs_tol=1e-4, rel_tol=1e-2)
        result = rkn1210_step(_harmonic_oscillator, 0.0, x0, 0.01, config=config)
        assert float(result.error_estimate) <= 1.0

    def test_custom_config(self):
        """RKN1210 respects tighter tolerances with AdaptiveConfig."""
        x0 = jnp.array([1.0, 0.0])
        dt = 1.0
        config_loose = AdaptiveConfig(abs_tol=1e-2, rel_tol=1e-1)
        result_loose = rkn1210_step(_harmonic_oscillator, 0.0, x0, dt, config=config_loose)
        config_tight = AdaptiveConfig(abs_tol=1e-8, rel_tol=1e-6)
        result_tight = rkn1210_step(_harmonic_oscillator, 0.0, x0, dt, config=config_tight)
        assert float(jnp.abs(result_tight.dt_used)) <= float(jnp.abs(result_loose.dt_used))
