"""Brahe (SI float64) reference implementations used by the nondim
validation suites.  If brahe lacks a specific routine, this module
provides a self-contained float64 reference instead.
"""

from __future__ import annotations

import math

import numpy as np


def hcw_state_after(state_si_f64: np.ndarray, n_si_f64: float, t_si_f64: float) -> np.ndarray:
    """Analytical HCW propagation in pure float64 numpy.

    Args:
        state_si_f64: 6-element initial relative state (m, m/s).
        n_si_f64: mean motion of the chief (rad/s).
        t_si_f64: elapsed time (s).
    Returns:
        Propagated state, shape (6,), float64.
    """
    state = np.asarray(state_si_f64, dtype=np.float64).reshape(6)
    n = float(n_si_f64)
    t = float(t_si_f64)
    nt = n * t
    c, s = math.cos(nt), math.sin(nt)

    phi = np.array(
        [
            [4 - 3 * c, 0.0, 0.0, s / n, 2 * (1 - c) / n, 0.0],
            [6 * (s - nt), 1.0, 0.0, 2 * (c - 1) / n, (4 * s - 3 * nt) / n, 0.0],
            [0.0, 0.0, c, 0.0, 0.0, s / n],
            [3 * n * s, 0.0, 0.0, c, 2 * s, 0.0],
            [6 * n * (c - 1), 0.0, 0.0, -2 * s, 4 * c - 3, 0.0],
            [0.0, 0.0, -n * s, 0.0, 0.0, c],
        ],
        dtype=np.float64,
    )
    return phi @ state


def two_body_state_after(state_si_f64: np.ndarray, mu_si: float, t_si_f64: float) -> np.ndarray:
    """Two-body Keplerian propagation via brahe if available, else a
    self-contained float64 RK4 in pure numpy.

    Args:
        state_si_f64: initial ECI state (m, m/s), shape (6,).
        mu_si: central-body μ (m^3/s^2).
        t_si_f64: elapsed time (s).
    """
    try:
        import brahe as bh  # type: ignore
    except ImportError:
        bh = None

    if bh is not None and hasattr(bh, "state_keplerian_propagate"):
        return np.asarray(
            bh.state_keplerian_propagate(state_si_f64, t_si_f64),
            dtype=np.float64,
        )
    # Fallback: 4th-order Runge-Kutta in float64 with a small step.
    state = np.asarray(state_si_f64, dtype=np.float64).reshape(6)
    n_steps = max(int(abs(t_si_f64) / 1.0), 1)  # 1 second steps
    dt = t_si_f64 / n_steps

    def f(s):
        r = s[:3]
        v = s[3:]
        a = -mu_si * r / np.linalg.norm(r) ** 3
        return np.concatenate([v, a])

    for _ in range(n_steps):
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return state


def full_force_state_after(
    state_si_f64: np.ndarray,
    epoch_initial,
    t_si_f64: float,
    eop,
    space_weather,
) -> np.ndarray:
    """Reference propagation using astrojax SI at float64.

    This is *not* an independent reference -- it uses the same astrojax
    SI code path the tests compare against.  Its purpose is to provide
    a high-precision (float64) ground truth for assertion (B) without
    requiring brahe to expose a matching force-model API.

    If/when brahe exposes the matching ``accel_*`` SI functions, the
    body of this function can be replaced with a brahe-driven path to
    make assertion (B) truly independent.
    """
    import jax
    import jax.numpy as jnp

    from astrojax.config import get_dtype, set_dtype
    from astrojax.integrators import dp54_step
    from astrojax.orbit_dynamics import (
        ForceModelConfig,
        create_orbit_dynamics,
    )
    from astrojax.orbit_dynamics.gravity import GravityModel

    original = get_dtype()
    try:
        set_dtype(jnp.float64)
        gravity_model = GravityModel.from_type("JGM3")
        config = ForceModelConfig(
            gravity_type="spherical_harmonics",
            gravity_model=gravity_model,
            gravity_degree=5,
            gravity_order=5,
            drag=True,
            density_model="harris_priester",
            srp=True,
            third_body_sun=True,
            third_body_moon=True,
        )
        dyn = create_orbit_dynamics(
            eop=eop,
            epoch_0=epoch_initial,
            config=config,
            space_weather=space_weather,
        )
        state0 = jnp.asarray(state_si_f64, dtype=jnp.float64)
        t0 = jnp.float64(0.0)
        dt = jnp.float64(60.0)
        n_steps = int(t_si_f64 / 60.0)

        def step(carry, _):
            t, state = carry
            result = dp54_step(dyn, t, state, dt)
            return (t + result.dt_used, result.state), None

        @jax.jit
        def run(t0, state0):
            (_t_final, state_final), _ = jax.lax.scan(
                step,
                (t0, state0),
                None,
                length=n_steps,
            )
            return state_final

        state = run(t0, state0)
        return np.asarray(state, dtype=np.float64)
    finally:
        set_dtype(original)
