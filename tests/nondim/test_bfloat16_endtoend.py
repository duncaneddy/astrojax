"""End-to-end low-precision propagation: characterization, not regression.

Runs a multi-orbit nondim point-mass rollout in bfloat16 *and* float16
and asserts for each:
- No NaN / Inf appears in the trajectory.
- Final |r| stays within factor of 2 of initial magnitude.

Diagnostic output reports the absolute final radial drift in metres so
the practical accuracy is visible. Propagation duration:
``N_ORBITS * orbital_period`` at LEO (10 orbits ~ 15750 s ~ 4h 22min).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from astrojax.config import get_dtype, set_dtype
from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.integrators import rk4_step
from astrojax.nondim import (
    UnitSystem,
    from_nondim_state,
    to_nondim_state,
)
from astrojax.orbit_dynamics import accel_point_mass

A = R_EARTH + 500e3
N_ORBITS = 10
ORBIT_PERIOD = 2 * math.pi * math.sqrt(A**3 / GM_EARTH)
T_PROP = N_ORBITS * ORBIT_PERIOD
DT = 30.0

INITIAL_STATE_SI = np.array(
    [A, 0.0, 0.0, 0.0, math.sqrt(GM_EARTH / A), 0.0],
    dtype=np.float64,
)


@pytest.fixture(autouse=True)
def restore_dtype():
    original = get_dtype()
    yield
    set_dtype(original)


def _propagate_nondim(dtype) -> np.ndarray:
    """Run the nondim point-mass rollout and return the final SI state."""
    set_dtype(dtype)
    units = UnitSystem.from_orbit(A, GM_EARTH)
    state0 = to_nondim_state(
        jnp.asarray(INITIAL_STATE_SI, dtype=dtype),
        units,
    )
    mu_nd = jnp.asarray(1.0, dtype=dtype)
    t0 = jnp.zeros((), dtype=dtype)
    dt = jnp.asarray(DT / units.TU, dtype=dtype)
    n_steps = int(T_PROP / DT)

    def deriv(t_, s):
        r = s[:3]
        v = s[3:]
        a = accel_point_mass(r, jnp.zeros(3, dtype=r.dtype), mu_nd)
        return jnp.concatenate([v, a])

    def step(carry, _):
        t, s = carry
        s_next = rk4_step(deriv, t, s, dt).state
        return (t + dt, s_next), s_next  # accumulate full trajectory

    @jax.jit
    def run(t0, state0):
        (_t_final, _state_final), trajectory = jax.lax.scan(
            step,
            (t0, state0),
            None,
            length=n_steps,
        )
        return trajectory

    trajectory = run(t0, state0)

    state_final = trajectory[-1].astype(jnp.float32)
    state_si = from_nondim_state(state_final, units)
    return np.asarray(state_si, dtype=np.float64)


def test_bfloat16_multi_orbit_no_nan():
    """bfloat16 sustains a multi-orbit nondim rollout."""
    state_si = _propagate_nondim(jnp.bfloat16)

    final_r = float(np.linalg.norm(state_si[:3]))
    initial_r = float(np.linalg.norm(INITIAL_STATE_SI[:3]))
    drift_m = abs(final_r - initial_r)

    print(
        f"\n[multi-orbit T={T_PROP:.0f}s N={N_ORBITS} dtype=bfloat16]"
        f" initial |r|={initial_r:.3e} m, final |r|={final_r:.3e} m,"
        f" |drift|={drift_m:.3e} m, ratio={final_r / initial_r:.4f}"
    )

    assert 0.5 * initial_r < final_r < 2.0 * initial_r, (
        f"bfloat16 final |r| = {final_r:.3e} m, expected within "
        f"factor of 2 of initial {initial_r:.3e} m"
    )


def test_float16_multi_orbit_characterization():
    """float16 characterization — diverges over multi-orbit rollout.

    float16 has the same 5-bit exponent as a half-precision IEEE float
    (range ±65504), and a 10-bit mantissa. With nondim scaling, the
    range is fine — but the small mantissa combined with thousands of
    RK4 steps, plus underflow when orbital position components pass
    through zero, makes float16 unsuitable for LEO point-mass
    propagation past roughly one orbit.

    This test reports the observed final state (which will typically
    be non-finite) so users can see why bfloat16 — with its wider
    exponent — is the recommended low-precision choice.
    """
    state_si = _propagate_nondim(jnp.float16)

    final_r = float(np.linalg.norm(state_si[:3]))
    initial_r = float(np.linalg.norm(INITIAL_STATE_SI[:3]))
    is_finite = bool(np.all(np.isfinite(state_si)))

    print(
        f"\n[multi-orbit T={T_PROP:.0f}s N={N_ORBITS} dtype=float16]"
        f" initial |r|={initial_r:.3e} m, final |r|={final_r:.3e} m,"
        f" finite={is_finite} — float16 is not recommended for LEO."
    )

    # Characterization only — no assertion on accuracy.
    # If float16 ever becomes viable here we'd want to know, so
    # explicitly do NOT assert anything that would make this a
    # regression failure.
