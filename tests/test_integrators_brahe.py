"""Cross-validation tests comparing astrojax.integrators output against brahe 1.0+.

Brahe uses float64 internally while astrojax defaults to float32. Tolerances
are set to accommodate the precision difference. The RK4 comparison is
tighter because fixed-step methods produce deterministic results given
the same coefficients. Adaptive methods may differ in step-size decisions
due to float32 vs float64 error norms.
"""

import brahe as bh
import jax.numpy as jnp
import numpy as np

from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.integrators import (
    dp54_step,
    rk4_step,
    rkf45_step,
)

# Float32 vs float64 tolerances
_RK4_ATOL = 1e-3  # RK4 state comparison (float32 rounding on ~7e6 m values)
_ADAPTIVE_ATOL = 1.0  # Adaptive methods may take different steps
_REL_TOL = 1e-4  # Relative tolerance for position comparisons


# ──────────────────────────────────────────────
# Shared dynamics
# ──────────────────────────────────────────────


def _jax_harmonic(t, x):
    return jnp.array([x[1], -x[0]])


def _np_harmonic(t, x):
    return np.array([x[1], -x[0]])


def _jax_exp_decay(t, x):
    return -x


def _np_exp_decay(t, x):
    return -x


def _jax_two_body(t, state):
    r = state[:3]
    v = state[3:]
    r_norm = jnp.linalg.norm(r)
    a = -GM_EARTH * r / r_norm**3
    return jnp.concatenate([v, a])


def _np_two_body(t, state):
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)
    a = -float(GM_EARTH) * r / r_norm**3
    return np.concatenate([v, a])


# ──────────────────────────────────────────────
# RK4 comparison
# ──────────────────────────────────────────────


class TestRK4Brahe:
    def test_harmonic_oscillator(self):
        """RK4 matches brahe for harmonic oscillator."""
        x0_jax = jnp.array([1.0, 0.0])
        x0_np = np.array([1.0, 0.0])
        dt = 0.01

        astrojax_result = rk4_step(_jax_harmonic, 0.0, x0_jax, dt)
        brahe_rk4 = bh.RK4Integrator(2, _np_harmonic)
        brahe_result = brahe_rk4.step(0.0, x0_np, dt)

        assert jnp.allclose(
            astrojax_result.state,
            jnp.array(brahe_result),
            atol=1e-5,
        )

    def test_exponential_decay(self):
        """RK4 matches brahe for exponential decay."""
        x0_jax = jnp.array([1.0])
        x0_np = np.array([1.0])
        dt = 0.1

        astrojax_result = rk4_step(_jax_exp_decay, 0.0, x0_jax, dt)
        brahe_rk4 = bh.RK4Integrator(1, _np_exp_decay)
        brahe_result = brahe_rk4.step(0.0, x0_np, dt)

        assert jnp.allclose(
            astrojax_result.state,
            jnp.array(brahe_result),
            atol=1e-6,
        )

    def test_two_body_orbit(self):
        """RK4 matches brahe for two-body orbit propagation."""
        sma = float(R_EARTH + 500e3)
        v_circ_jax = jnp.sqrt(GM_EARTH / sma)
        v_circ_np = np.sqrt(float(GM_EARTH) / sma)
        x0_jax = jnp.array([sma, 0.0, 0.0, 0.0, v_circ_jax, 0.0])
        x0_np = np.array([sma, 0.0, 0.0, 0.0, v_circ_np, 0.0])
        dt = 10.0

        astrojax_result = rk4_step(_jax_two_body, 0.0, x0_jax, dt)
        brahe_rk4 = bh.RK4Integrator(6, _np_two_body)
        brahe_result = brahe_rk4.step(0.0, x0_np, dt)

        # Position comparison (relative tolerance due to large values)
        pos_astro = np.array(astrojax_result.state[:3])
        pos_brahe = brahe_result[:3]
        assert np.allclose(pos_astro, pos_brahe, rtol=_REL_TOL, atol=_RK4_ATOL)

        # Velocity comparison
        vel_astro = np.array(astrojax_result.state[3:])
        vel_brahe = brahe_result[3:]
        assert np.allclose(vel_astro, vel_brahe, rtol=_REL_TOL, atol=0.1)


# ──────────────────────────────────────────────
# RKF45 comparison
# ──────────────────────────────────────────────


class TestRKF45Brahe:
    def test_harmonic_oscillator_state(self):
        """RKF45 state matches brahe for harmonic oscillator."""
        x0_jax = jnp.array([1.0, 0.0])
        x0_np = np.array([1.0, 0.0])
        dt = 0.1

        astrojax_result = rkf45_step(_jax_harmonic, 0.0, x0_jax, dt)
        brahe_rkf = bh.RKF45Integrator(2, _np_harmonic)
        brahe_result = brahe_rkf.step(0.0, x0_np, dt)

        assert jnp.allclose(
            astrojax_result.state,
            jnp.array(brahe_result.state),
            atol=1e-4,
        )

    def test_exponential_decay_state(self):
        """RKF45 state matches brahe for exponential decay."""
        x0_jax = jnp.array([1.0])
        x0_np = np.array([1.0])
        dt = 0.5

        astrojax_result = rkf45_step(_jax_exp_decay, 0.0, x0_jax, dt)
        brahe_rkf = bh.RKF45Integrator(1, _np_exp_decay)
        brahe_result = brahe_rkf.step(0.0, x0_np, dt)

        assert jnp.allclose(
            astrojax_result.state,
            jnp.array(brahe_result.state),
            atol=1e-4,
        )

    def test_two_body_orbit_state(self):
        """RKF45 state matches brahe for two-body orbit."""
        sma = float(R_EARTH + 500e3)
        v_circ_jax = jnp.sqrt(GM_EARTH / sma)
        v_circ_np = np.sqrt(float(GM_EARTH) / sma)
        x0_jax = jnp.array([sma, 0.0, 0.0, 0.0, v_circ_jax, 0.0])
        x0_np = np.array([sma, 0.0, 0.0, 0.0, v_circ_np, 0.0])
        dt = 60.0

        astrojax_result = rkf45_step(_jax_two_body, 0.0, x0_jax, dt)
        brahe_rkf = bh.RKF45Integrator(6, _np_two_body)
        brahe_result = brahe_rkf.step(0.0, x0_np, dt)

        # Position comparison
        pos_astro = np.array(astrojax_result.state[:3])
        pos_brahe = np.array(brahe_result.state[:3])
        assert np.allclose(pos_astro, pos_brahe, rtol=_REL_TOL, atol=_ADAPTIVE_ATOL)


# ──────────────────────────────────────────────
# DP54 comparison
# ──────────────────────────────────────────────


class TestDP54Brahe:
    def test_harmonic_oscillator_state(self):
        """DP54 state matches brahe for harmonic oscillator."""
        x0_jax = jnp.array([1.0, 0.0])
        x0_np = np.array([1.0, 0.0])
        dt = 0.1

        astrojax_result = dp54_step(_jax_harmonic, 0.0, x0_jax, dt)
        brahe_dp = bh.DP54Integrator(2, _np_harmonic)
        brahe_result = brahe_dp.step(0.0, x0_np, dt)

        assert jnp.allclose(
            astrojax_result.state,
            jnp.array(brahe_result.state),
            atol=1e-4,
        )

    def test_exponential_decay_state(self):
        """DP54 state matches brahe for exponential decay."""
        x0_jax = jnp.array([1.0])
        x0_np = np.array([1.0])
        dt = 0.5

        astrojax_result = dp54_step(_jax_exp_decay, 0.0, x0_jax, dt)
        brahe_dp = bh.DP54Integrator(1, _np_exp_decay)
        brahe_result = brahe_dp.step(0.0, x0_np, dt)

        assert jnp.allclose(
            astrojax_result.state,
            jnp.array(brahe_result.state),
            atol=1e-4,
        )

    def test_two_body_orbit_state(self):
        """DP54 state matches brahe for two-body orbit."""
        sma = float(R_EARTH + 500e3)
        v_circ_jax = jnp.sqrt(GM_EARTH / sma)
        v_circ_np = np.sqrt(float(GM_EARTH) / sma)
        x0_jax = jnp.array([sma, 0.0, 0.0, 0.0, v_circ_jax, 0.0])
        x0_np = np.array([sma, 0.0, 0.0, 0.0, v_circ_np, 0.0])
        dt = 60.0

        astrojax_result = dp54_step(_jax_two_body, 0.0, x0_jax, dt)
        brahe_dp = bh.DP54Integrator(6, _np_two_body)
        brahe_result = brahe_dp.step(0.0, x0_np, dt)

        # Position comparison
        pos_astro = np.array(astrojax_result.state[:3])
        pos_brahe = np.array(brahe_result.state[:3])
        assert np.allclose(pos_astro, pos_brahe, rtol=_REL_TOL, atol=_ADAPTIVE_ATOL)
