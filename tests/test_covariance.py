"""Tests for astrojax.covariance — variational equation STM propagation.

Tests cover:
- Shape correctness of augmented dynamics output
- Linear system: STM vs expm(A·dt) (exact for constant A)
- Two-body: STM matches jax.jacfwd through RK4 step
- JIT/vmap/lax.scan compatibility
- Covariance propagation arithmetic
"""

import jax
import jax.numpy as jnp

from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.covariance import (
    augmented_initial_state,
    create_variational_dynamics,
    extract_state_and_stm,
    propagate_covariance,
)
from astrojax.integrators import rk4_step

# ──────────────────────────────────────────────
# Helper dynamics
# ──────────────────────────────────────────────


def _harmonic(t, x):
    """Simple harmonic oscillator: dx/dt = [x1, -x0]."""
    return jnp.array([x[1], -x[0]])


def _linear_2d(t, x):
    """Linear system dx/dt = A @ x with constant A."""
    A = jnp.array([[0.0, 1.0], [-1.0, -0.1]])
    return A @ x


def _two_body(t, state):
    """Two-body point-mass dynamics (3D position + velocity)."""
    r = state[:3]
    v = state[3:]
    r_mag = jnp.linalg.norm(r)
    a = -GM_EARTH * r / r_mag**3
    return jnp.concatenate([v, a])


# ──────────────────────────────────────────────
# Shape tests
# ──────────────────────────────────────────────


class TestShapes:
    """Verify shapes of augmented dynamics, initial state, and extraction."""

    def test_augmented_initial_state_shape(self):
        n = 4
        x0 = jnp.ones(n)
        aug = augmented_initial_state(x0, n)
        assert aug.shape == (n + n * n,)

    def test_augmented_initial_state_identity(self):
        n = 3
        x0 = jnp.array([1.0, 2.0, 3.0])
        aug = augmented_initial_state(x0, n)
        x, Phi = extract_state_and_stm(aug, n)
        assert jnp.allclose(x, x0)
        assert jnp.allclose(Phi, jnp.eye(n))

    def test_augmented_dynamics_output_shape(self):
        n = 2
        aug_dyn = create_variational_dynamics(_harmonic, n)
        aug_x0 = augmented_initial_state(jnp.array([1.0, 0.0]), n)
        deriv = aug_dyn(0.0, aug_x0)
        assert deriv.shape == (n + n * n,)

    def test_augmented_dynamics_output_shape_6d(self):
        n = 6
        aug_dyn = create_variational_dynamics(_two_body, n)
        r0 = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        v0 = jnp.array([0.0, 7612.0, 0.0])
        x0 = jnp.concatenate([r0, v0])
        aug_x0 = augmented_initial_state(x0, n)
        deriv = aug_dyn(0.0, aug_x0)
        assert deriv.shape == (n + n * n,)

    def test_extract_roundtrip(self):
        n = 3
        x0 = jnp.array([1.0, 2.0, 3.0])
        aug = augmented_initial_state(x0, n)
        x, Phi = extract_state_and_stm(aug, n)
        assert x.shape == (n,)
        assert Phi.shape == (n, n)


# ──────────────────────────────────────────────
# Linear system: STM vs matrix exponential
# ──────────────────────────────────────────────


class TestLinearSystem:
    """For a constant-coefficient linear system dx/dt = Ax, the STM is expm(A·t)."""

    def test_stm_matches_expm(self):
        """Integrate STM for linear system and compare to expm(A·dt)."""
        from jax.scipy.linalg import expm

        n = 2
        A = jnp.array([[0.0, 1.0], [-1.0, -0.1]])
        dt = 0.01
        n_steps = 100
        total_time = dt * n_steps

        aug_dyn = create_variational_dynamics(_linear_2d, n)
        x0 = jnp.array([1.0, 0.0])
        aug_x0 = augmented_initial_state(x0, n)

        # Integrate with RK4
        def scan_step(carry, _):
            t, aug = carry
            result = rk4_step(aug_dyn, t, aug, dt)
            return (t + dt, result.state), None

        (_, aug_final), _ = jax.lax.scan(scan_step, (0.0, aug_x0), None, length=n_steps)
        _, Phi_numerical = extract_state_and_stm(aug_final, n)

        # Exact STM = expm(A * total_time)
        Phi_exact = expm(A * total_time)

        assert jnp.allclose(Phi_numerical, Phi_exact, atol=1e-4, rtol=1e-4)

    def test_state_matches_expm(self):
        """State propagated through augmented dynamics matches expm."""
        from jax.scipy.linalg import expm

        n = 2
        A = jnp.array([[0.0, 1.0], [-1.0, -0.1]])
        dt = 0.01
        n_steps = 100
        total_time = dt * n_steps

        aug_dyn = create_variational_dynamics(_linear_2d, n)
        x0 = jnp.array([1.0, 0.0])
        aug_x0 = augmented_initial_state(x0, n)

        def scan_step(carry, _):
            t, aug = carry
            result = rk4_step(aug_dyn, t, aug, dt)
            return (t + dt, result.state), None

        (_, aug_final), _ = jax.lax.scan(scan_step, (0.0, aug_x0), None, length=n_steps)
        x_numerical, _ = extract_state_and_stm(aug_final, n)

        # Exact: x(t) = expm(A*t) @ x0
        x_exact = expm(A * total_time) @ x0

        assert jnp.allclose(x_numerical, x_exact, atol=1e-4, rtol=1e-4)


# ──────────────────────────────────────────────
# Two-body: STM vs jacfwd through RK4
# ──────────────────────────────────────────────


class TestTwoBody:
    """STM from variational equations matches jacfwd through the propagation."""

    def test_stm_matches_jacfwd(self):
        """Single RK4 step: variational STM ≈ jacfwd(propagate)(x0)."""
        n = 6
        r0 = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        v0 = jnp.array([0.0, 7612.0, 0.0])
        x0 = jnp.concatenate([r0, v0])
        dt = 10.0

        # Method 1: Variational equations (one RK4 step)
        aug_dyn = create_variational_dynamics(_two_body, n)
        aug_x0 = augmented_initial_state(x0, n)
        result = rk4_step(aug_dyn, 0.0, aug_x0, dt)
        _, Phi_var = extract_state_and_stm(result.state, n)

        # Method 2: jacfwd through the RK4 step
        def propagate(x_):
            return rk4_step(_two_body, 0.0, x_, dt).state

        Phi_jac = jax.jacfwd(propagate)(x0)

        assert jnp.allclose(Phi_var, Phi_jac, atol=1e-3, rtol=1e-3)

    def test_stm_multi_step_matches_jacfwd(self):
        """Multi-step propagation: variational STM ≈ jacfwd."""
        n = 6
        r0 = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        v0 = jnp.array([0.0, 7612.0, 0.0])
        x0 = jnp.concatenate([r0, v0])
        dt = 10.0
        n_steps = 10

        # Variational
        aug_dyn = create_variational_dynamics(_two_body, n)
        aug_x0 = augmented_initial_state(x0, n)

        def scan_step(carry, _):
            t, aug = carry
            result = rk4_step(aug_dyn, t, aug, dt)
            return (t + dt, result.state), None

        (_, aug_final), _ = jax.lax.scan(scan_step, (0.0, aug_x0), None, length=n_steps)
        _, Phi_var = extract_state_and_stm(aug_final, n)

        # jacfwd through multi-step propagation
        def propagate_multi(x_):
            def step(carry, _):
                t, state = carry
                result = rk4_step(_two_body, t, state, dt)
                return (t + dt, result.state), None

            (_, x_final), _ = jax.lax.scan(step, (0.0, x_), None, length=n_steps)
            return x_final

        Phi_jac = jax.jacfwd(propagate_multi)(x0)

        assert jnp.allclose(Phi_var, Phi_jac, atol=1e-2, rtol=1e-2)


# ──────────────────────────────────────────────
# JIT / vmap / lax.scan compatibility
# ──────────────────────────────────────────────


class TestJAXCompatibility:
    """Verify that augmented dynamics work with JIT, vmap, and lax.scan."""

    def test_jit(self):
        n = 2
        aug_dyn = create_variational_dynamics(_harmonic, n)
        x0 = jnp.array([1.0, 0.0])
        aug_x0 = augmented_initial_state(x0, n)

        @jax.jit
        def step(aug):
            return rk4_step(aug_dyn, 0.0, aug, 0.01).state

        result = step(aug_x0)
        assert result.shape == (n + n * n,)

    def test_vmap(self):
        n = 2
        aug_dyn = create_variational_dynamics(_harmonic, n)

        x0_batch = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        aug_batch = jax.vmap(augmented_initial_state, in_axes=(0, None))(x0_batch, n)

        @jax.vmap
        def step(aug):
            return rk4_step(aug_dyn, 0.0, aug, 0.01).state

        results = step(aug_batch)
        assert results.shape == (
            3,
            n + n * n,
        )

    def test_lax_scan(self):
        n = 2
        aug_dyn = create_variational_dynamics(_harmonic, n)
        x0 = jnp.array([1.0, 0.0])
        aug_x0 = augmented_initial_state(x0, n)
        dt = 0.01

        def scan_step(carry, _):
            t, aug = carry
            result = rk4_step(aug_dyn, t, aug, dt)
            return (t + dt, result.state), None

        (t_final, aug_final), _ = jax.lax.scan(scan_step, (0.0, aug_x0), None, length=50)
        x, Phi = extract_state_and_stm(aug_final, n)
        assert x.shape == (n,)
        assert Phi.shape == (n, n)


# ──────────────────────────────────────────────
# Covariance propagation arithmetic
# ──────────────────────────────────────────────


class TestCovariancePropagation:
    """Verify propagate_covariance arithmetic."""

    def test_identity_stm(self):
        """Identity STM should leave covariance unchanged (no Q)."""
        n = 3
        P0 = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
        Phi = jnp.eye(n)
        P = propagate_covariance(Phi, P0)
        assert jnp.allclose(P, P0, atol=1e-6)

    def test_with_process_noise(self):
        """P = Φ P₀ Φᵀ + Q."""
        n = 2
        P0 = jnp.eye(n)
        Phi = jnp.array([[1.0, 0.1], [0.0, 1.0]])
        Q = jnp.eye(n) * 0.01
        P = propagate_covariance(Phi, P0, Q)
        P_expected = Phi @ P0 @ Phi.T + Q
        assert jnp.allclose(P, P_expected, atol=1e-6)

    def test_symmetry(self):
        """Propagated covariance should be symmetric."""
        n = 4
        # Random-ish PSD matrix
        A = jnp.array(
            [
                [2.0, 0.5, 0.1, 0.0],
                [0.5, 3.0, 0.2, 0.1],
                [0.1, 0.2, 1.0, 0.3],
                [0.0, 0.1, 0.3, 2.0],
            ]
        )
        P0 = A @ A.T
        Phi = jnp.eye(n) + 0.01 * jnp.ones((n, n))
        P = propagate_covariance(Phi, P0)
        assert jnp.allclose(P, P.T, atol=1e-5)

    def test_no_q_default(self):
        """Without Q, result is just Φ P₀ Φᵀ."""
        n = 2
        P0 = jnp.eye(n) * 4.0
        Phi = 2.0 * jnp.eye(n)
        P = propagate_covariance(Phi, P0)
        # Φ P₀ Φᵀ = 2I * 4I * 2I = 16I
        assert jnp.allclose(P, 16.0 * jnp.eye(n), atol=1e-5)

    def test_jit_compatible(self):
        """propagate_covariance works under JIT."""
        n = 3
        P0 = jnp.eye(n)
        Phi = jnp.eye(n)
        Q = jnp.eye(n) * 0.1

        @jax.jit
        def f(Phi, P0, Q):
            return propagate_covariance(Phi, P0, Q)

        P = f(Phi, P0, Q)
        assert jnp.allclose(P, jnp.eye(n) * 1.1, atol=1e-6)
