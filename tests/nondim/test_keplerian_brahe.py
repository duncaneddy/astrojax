"""Two-body propagation: 3-way assertion across dtypes.

Propagates a 7000 km circular ECI orbit (|r| ~ 7e6 m) for **1800 s
(30 minutes)** using RK4 with dt=30s (60 steps), comparing astrojax SI
and nondim paths against a float64 RK4 reference with 1s steps.

Diagnostic output reports both relative position error (dimensionless)
and absolute position error (metres) so users can judge usability.

For SI float16/bfloat16, |r|=7e6 m exceeds float16's representable
range (max 65504) and approaches the limit where bfloat16 cannot
resolve sub-km positions. Both SI low-precision paths are expected to
fail; the nondim paths run with |r|~1 and remain bounded.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from _brahe_reference import two_body_state_after

from astrojax.config import get_dtype, set_dtype
from astrojax.constants import GM_EARTH
from astrojax.integrators import rk4_step
from astrojax.nondim import (
    UnitSystem,
    from_nondim_state,
    to_nondim_state,
)
from astrojax.orbit_dynamics import accel_point_mass

A = 7000e3
STATE_SI = np.array([A, 0.0, 0.0, 0.0, math.sqrt(GM_EARTH / A), 0.0], dtype=np.float64)
T_PROP = 1800.0  # ~30 minutes
DT = 30.0
N_STEPS = int(T_PROP / DT)

REFERENCE = two_body_state_after(STATE_SI, GM_EARTH, T_PROP)

# Tolerances refined empirically (see Task 5 report).
# Observed errors with DT=30s over T_PROP=1800s vs float64 RK4 1s-step ref:
#   float64:  err_SI=3.009e-08, err_nondim=3.009e-08, err_consistency=5.363e-16
#   float32:  err_SI=2.577e-07, err_nondim=2.248e-07, err_consistency=7.143e-08
#   bfloat16: err_SI=2.128e-02, err_nondim=1.322e-02, err_consistency=9.322e-03
# (tol_si, tol_nondim, tol_consistency); None = skip that check.
TOL_TABLE = {
    jnp.float64: (5e-8, 5e-8, 1e-15),
    jnp.float32: (5e-7, 5e-7, 1e-7),
    jnp.bfloat16: (None, 5e-2, 5e-2),
    jnp.float16: (None, None, None),  # characterization-only; SI overflows, nondim may diverge
}


@pytest.fixture(autouse=True)
def restore_dtype():
    original = get_dtype()
    yield
    set_dtype(original)


def _two_body_deriv(t, state, mu):
    """ODE right-hand side for two-body motion.

    Works for any consistent unit system: pass SI values (m, m/s, μ in
    m³/s²) for SI propagation, or nondim values (LU, LU/TU, μ_nd) for
    nondim propagation. The functional form is identical because the
    ODE is unit-invariant.
    """
    r = state[:3]
    v = state[3:]
    a = accel_point_mass(r, jnp.zeros(3, dtype=r.dtype), mu)
    return jnp.concatenate([v, a])


def _astrojax_si_two_body(dtype) -> np.ndarray:
    set_dtype(dtype)
    state = jnp.asarray(STATE_SI, dtype=dtype)
    mu = jnp.asarray(GM_EARTH, dtype=dtype)
    t = jnp.zeros((), dtype=dtype)
    dt = jnp.asarray(DT, dtype=dtype)
    for _ in range(N_STEPS):
        state = rk4_step(lambda t_, s: _two_body_deriv(t_, s, mu), t, state, dt).state
        t = t + dt
    return np.asarray(state, dtype=np.float64)


def _astrojax_nondim_two_body(dtype) -> np.ndarray:
    set_dtype(dtype)
    units = UnitSystem.from_orbit(A, GM_EARTH)  # μ_nd = 1, |r_nd| ≈ 1
    state_nd = to_nondim_state(jnp.asarray(STATE_SI, dtype=dtype), units)
    mu_nd = jnp.asarray(1.0, dtype=dtype)
    t_nd = jnp.zeros((), dtype=dtype)
    dt_nd = jnp.asarray(DT / units.TU, dtype=dtype)

    def deriv(t_, s):
        return _two_body_deriv(t_, s, mu_nd)

    for _ in range(N_STEPS):
        state_nd = rk4_step(deriv, t_nd, state_nd, dt_nd).state
        t_nd = t_nd + dt_nd

    out_si = from_nondim_state(state_nd, units)
    return np.asarray(out_si, dtype=np.float64)


def _rel_position_err(predicted, reference) -> float:
    return float(
        np.linalg.norm(predicted[:3] - reference[:3]) / max(np.linalg.norm(reference[:3]), 1e-12)
    )


def _abs_position_err_m(predicted, reference) -> float:
    """Absolute position error in metres."""
    return float(np.linalg.norm(predicted[:3] - reference[:3]))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32, jnp.bfloat16, jnp.float16])
def test_two_body_three_way_assertion(dtype):
    tol_si, tol_nd, tol_cons = TOL_TABLE[dtype]

    si_path = _astrojax_si_two_body(dtype)
    nd_path = _astrojax_nondim_two_body(dtype)

    err_si = _rel_position_err(si_path, REFERENCE)
    err_nd = _rel_position_err(nd_path, REFERENCE)
    err_cons = _rel_position_err(si_path, nd_path)
    abs_si_m = _abs_position_err_m(si_path, REFERENCE)
    abs_nd_m = _abs_position_err_m(nd_path, REFERENCE)

    print(
        f"\n[two-body T={T_PROP:.0f}s dtype={dtype.__name__}]"
        f" err_SI={err_si:.2e} ({abs_si_m:.2e} m)"
        f" err_nondim={err_nd:.2e} ({abs_nd_m:.2e} m)"
        f" err_consistency={err_cons:.2e}"
    )

    if tol_si is not None:
        assert err_si <= tol_si, f"SI: {err_si:.3e} > {tol_si:.3e}"
    if tol_nd is not None:
        assert err_nd <= tol_nd, f"Nondim: {err_nd:.3e} > {tol_nd:.3e}"
    if tol_cons is not None:
        assert err_cons <= tol_cons, f"Consistency: {err_cons:.3e} > {tol_cons:.3e}"
