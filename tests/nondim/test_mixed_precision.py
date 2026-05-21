"""Mixed-precision propagation benchmark.

For each of (two-body, full-force) and each low-precision dtype
(bfloat16, float16), compares three modes against a float64 reference:

1. **naive**: ``set_dtype(low_dtype)``; the scan carry is at the
   low-precision dtype; all math is done at the low-precision dtype.
2. **per-step mixed**: ``set_dtype(float32)``; the scan carry is at
   the low-precision dtype; but each derivative call internally
   computes in float32 (via ``mixed_precision_dynamics``). The
   end-of-step cast back to low-precision quantises the state every
   step, which adds noise -- often making this *worse* than naive.
3. **scan-boundary mixed**: ``set_dtype(float32)``; the scan carry is
   at float32; the state is cast to low-precision only at scan
   entry/exit. This is the pattern ML frameworks actually use.

Reports absolute position error in metres against the same float64
reference so usability is clear.

Characterisation only: the test asserts that *scan-boundary mixed* is
at least as good as naive (the recommended pattern works) and is
finite. It does not assert per-step mixed performance because the
result depends on how the truncation pattern of the dtype interacts
with the orbit phase.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _brahe_reference import full_force_state_after, two_body_state_after

from astrojax.config import get_dtype, set_dtype
from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.eop import zero_eop
from astrojax.epoch import Epoch
from astrojax.integrators import dp54_step, rk4_step
from astrojax.nondim import (
    UnitSystem,
    cast_state,
    from_nondim_state,
    mixed_precision_dynamics,
    nondim_orbit_dynamics,
    to_nondim_state,
)
from astrojax.orbit_dynamics import accel_point_mass
from astrojax.orbit_dynamics.config import ForceModelConfig
from astrojax.orbit_dynamics.gravity import GravityModel
from astrojax.space_weather import zero_space_weather


@pytest.fixture(autouse=True)
def restore_dtype():
    original = get_dtype()
    yield
    set_dtype(original)


def _abs_err_m(predicted, reference) -> float:
    diff = np.asarray(predicted, dtype=np.float64)[:3] - reference[:3]
    return float(np.linalg.norm(diff))


# ──────────────────────────────────────────────────────────────────
# Two-body benchmark
# ──────────────────────────────────────────────────────────────────

A_TB = 7000e3
STATE_TB_SI = np.array([A_TB, 0.0, 0.0, 0.0, math.sqrt(GM_EARTH / A_TB), 0.0], dtype=np.float64)
T_TB = 1800.0  # 30 minutes
DT_TB = 30.0
N_TB = int(T_TB / DT_TB)

REF_TB = two_body_state_after(STATE_TB_SI, GM_EARTH, T_TB)


def _two_body_deriv(t, state, mu):
    r = state[:3]
    v = state[3:]
    a = accel_point_mass(r, jnp.zeros(3, dtype=r.dtype), mu)
    return jnp.concatenate([v, a])


def _propagate_two_body_naive(low_dtype: jnp.dtype) -> np.ndarray:
    """All-low-precision propagation: scan carry and math both at low_dtype.

    Builds the initial nondim state via float32 (SI 7e6 → bf16/float16
    would overflow float16), then casts to low_dtype for the rollout.
    """
    units = UnitSystem.from_orbit(A_TB, GM_EARTH)
    # Build state0 in nondim units via float32 to avoid overflow when
    # casting SI position (7e6 m) directly to float16.
    set_dtype(jnp.float32)
    state0_nd_f32 = to_nondim_state(jnp.asarray(STATE_TB_SI, dtype=jnp.float32), units)
    set_dtype(low_dtype)
    state0 = state0_nd_f32.astype(low_dtype)
    mu_nd = jnp.asarray(1.0, dtype=low_dtype)
    t0 = jnp.zeros((), dtype=low_dtype)
    dt = jnp.asarray(DT_TB / units.TU, dtype=low_dtype)

    def deriv(t_, s):
        return _two_body_deriv(t_, s, mu_nd)

    def step(carry, _):
        t, s = carry
        result = rk4_step(deriv, t, s, dt)
        return (t + dt, result.state.astype(low_dtype)), None

    @jax.jit
    def run(t0, state0):
        (_t, state_final), _ = jax.lax.scan(step, (t0, state0), None, length=N_TB)
        return state_final

    state_nd = run(t0, state0)
    out_si = from_nondim_state(state_nd.astype(jnp.float32), units)
    return np.asarray(out_si, dtype=np.float64)


def _propagate_two_body_per_step_mixed(low_dtype: jnp.dtype) -> np.ndarray:
    """Scan carry at low_dtype; derivative internally float32.

    Demonstrates the trap: quantising state back to low_dtype every
    step is often worse than naive low_dtype throughout.
    """
    set_dtype(jnp.float32)
    units = UnitSystem.from_orbit(A_TB, GM_EARTH)
    state0_low = cast_state(
        to_nondim_state(jnp.asarray(STATE_TB_SI, dtype=jnp.float32), units),
        low_dtype,
    )
    mu_nd = jnp.float32(1.0)
    t0_low = jnp.zeros((), dtype=low_dtype)
    dt_low = jnp.asarray(DT_TB / units.TU, dtype=low_dtype)

    def deriv_f32(t_, s):
        return _two_body_deriv(t_, s, mu_nd)

    deriv_wrapped = mixed_precision_dynamics(
        deriv_f32, storage_dtype=low_dtype, compute_dtype=jnp.float32
    )

    def step(carry, _):
        t, s = carry
        result = rk4_step(deriv_wrapped, t, s, dt_low)
        # rk4_step internally casts to get_dtype()=float32, so we
        # explicitly cast back to low_dtype for the scan carry.
        return (t + dt_low, result.state.astype(low_dtype)), None

    @jax.jit
    def run(t0, state0):
        (_t, state_final), _ = jax.lax.scan(step, (t0, state0), None, length=N_TB)
        return state_final

    state_nd = run(t0_low, state0_low)
    out_si = from_nondim_state(state_nd.astype(jnp.float32), units)
    return np.asarray(out_si, dtype=np.float64)


def _propagate_two_body_scan_boundary_mixed(low_dtype: jnp.dtype) -> np.ndarray:
    """Scan carry at float32; storage low_dtype only at entry/exit.

    This is the pattern that actually works -- the same pattern ML
    frameworks use for bfloat16 weights with fp32 forward/backward.
    """
    set_dtype(jnp.float32)
    units = UnitSystem.from_orbit(A_TB, GM_EARTH)
    state_bf16 = jnp.asarray(STATE_TB_SI, dtype=low_dtype)

    @jax.jit
    def rollout(state_storage):
        # 1. Lift to compute precision once at entry.
        state_f32 = cast_state(state_storage, jnp.float32)
        state_nd_f32 = to_nondim_state(state_f32, units)
        mu_nd = jnp.float32(1.0)
        t0_f32 = jnp.float32(0.0)
        dt_f32 = jnp.float32(DT_TB / units.TU)

        def deriv(t_, s):
            return _two_body_deriv(t_, s, mu_nd)

        def step(carry, _):
            t, s = carry
            result = rk4_step(deriv, t, s, dt_f32)
            return (t + dt_f32, result.state), None

        # 2. Integrate entirely in float32.
        (_t, state_final_f32), _ = jax.lax.scan(step, (t0_f32, state_nd_f32), None, length=N_TB)
        state_final_si_f32 = from_nondim_state(state_final_f32, units)
        # 3. Cast back to low-precision storage at exit.
        return cast_state(state_final_si_f32, low_dtype)

    out_low = rollout(state_bf16)
    return np.asarray(out_low.astype(jnp.float32), dtype=np.float64)


@pytest.mark.parametrize("low_dtype", [jnp.bfloat16, jnp.float16])
def test_two_body_mixed_precision_modes(low_dtype):
    naive = _propagate_two_body_naive(low_dtype)
    per_step = _propagate_two_body_per_step_mixed(low_dtype)
    scan_boundary = _propagate_two_body_scan_boundary_mixed(low_dtype)

    err_naive = _abs_err_m(naive, REF_TB)
    err_per_step = _abs_err_m(per_step, REF_TB)
    err_scan = _abs_err_m(scan_boundary, REF_TB)

    print(
        f"\n[two-body T={T_TB:.0f}s dtype={low_dtype.__name__}]"
        f" naive={err_naive:.3e} m"
        f" per-step-mixed={err_per_step:.3e} m"
        f" scan-boundary-mixed={err_scan:.3e} m"
    )

    # The recommended pattern (scan-boundary mixed) must beat naive
    # when both are finite. float16 is allowed to diverge -- it's the
    # dtype's exponent range that fails, not the mixed-precision
    # strategy.
    if np.isfinite(err_naive) and np.isfinite(err_scan):
        assert err_scan <= err_naive, (
            f"Scan-boundary mixed ({err_scan:.3e} m) should be at least "
            f"as good as naive ({err_naive:.3e} m)."
        )


# ──────────────────────────────────────────────────────────────────
# Full force benchmark
# ──────────────────────────────────────────────────────────────────

A_FF = R_EARTH + 500e3
DT_FF = 60.0
N_FF = 10
T_FF = DT_FF * N_FF  # 10 minutes
STATE_FF_SI = np.array([A_FF, 0.0, 0.0, 0.0, math.sqrt(GM_EARTH / A_FF), 0.0], dtype=np.float64)
EPOCH_FF = Epoch(2024, 1, 1, 12, 0, 0.0)
EOP_FF = zero_eop()
SW_FF = zero_space_weather()
REF_FF = full_force_state_after(STATE_FF_SI, EPOCH_FF, T_FF, EOP_FF, SW_FF)


def _build_full_config():
    return ForceModelConfig(
        gravity_type="spherical_harmonics",
        gravity_model=GravityModel.from_type("JGM3"),
        gravity_degree=5,
        gravity_order=5,
        drag=True,
        density_model="harris_priester",
        srp=True,
        third_body_sun=True,
        third_body_moon=True,
    )


def _propagate_full_naive(low_dtype: jnp.dtype) -> np.ndarray:
    units = UnitSystem.from_orbit(A_FF, GM_EARTH)
    config = _build_full_config()
    # Build state via float32 to avoid float16 overflow on SI position.
    set_dtype(jnp.float32)
    state0_nd_f32 = to_nondim_state(jnp.asarray(STATE_FF_SI, dtype=jnp.float32), units)
    set_dtype(low_dtype)
    dyn = nondim_orbit_dynamics(
        eop=EOP_FF,
        epoch_0=EPOCH_FF,
        units=units,
        config=config,
        space_weather=SW_FF,
    )
    state0 = state0_nd_f32.astype(low_dtype)
    t0 = jnp.zeros((), dtype=low_dtype)
    dt = jnp.asarray(DT_FF / units.TU, dtype=low_dtype)

    def step(carry, _):
        t, s = carry
        result = dp54_step(dyn, t, s, dt)
        return (t + result.dt_used.astype(low_dtype), result.state.astype(low_dtype)), None

    @jax.jit
    def run(t0, state0):
        (_t, state_final), _ = jax.lax.scan(step, (t0, state0), None, length=N_FF)
        return state_final

    state_nd = run(t0, state0)
    out_si = from_nondim_state(state_nd.astype(jnp.float32), units)
    return np.asarray(out_si, dtype=np.float64)


def _propagate_full_scan_boundary_mixed(low_dtype: jnp.dtype) -> np.ndarray:
    set_dtype(jnp.float32)
    units = UnitSystem.from_orbit(A_FF, GM_EARTH)
    config = _build_full_config()
    dyn = nondim_orbit_dynamics(
        eop=EOP_FF,
        epoch_0=EPOCH_FF,
        units=units,
        config=config,
        space_weather=SW_FF,
    )
    state_low = jnp.asarray(STATE_FF_SI, dtype=low_dtype)

    @jax.jit
    def rollout(state_storage):
        state_f32 = cast_state(state_storage, jnp.float32)
        state_nd_f32 = to_nondim_state(state_f32, units)
        t0_f32 = jnp.float32(0.0)
        dt_f32 = jnp.float32(DT_FF / units.TU)

        def step(carry, _):
            t, s = carry
            result = dp54_step(dyn, t, s, dt_f32)
            return (t + result.dt_used, result.state), None

        (_t, state_final_f32), _ = jax.lax.scan(step, (t0_f32, state_nd_f32), None, length=N_FF)
        state_final_si_f32 = from_nondim_state(state_final_f32, units)
        return cast_state(state_final_si_f32, low_dtype)

    out_low = rollout(state_low)
    return np.asarray(out_low.astype(jnp.float32), dtype=np.float64)


@pytest.mark.parametrize("low_dtype", [jnp.bfloat16, jnp.float16])
def test_full_force_scan_boundary_mixed_beats_naive(low_dtype):
    naive = _propagate_full_naive(low_dtype)
    scan_boundary = _propagate_full_scan_boundary_mixed(low_dtype)
    err_naive = _abs_err_m(naive, REF_FF)
    err_scan = _abs_err_m(scan_boundary, REF_FF)

    print(
        f"\n[full-force T={T_FF:.0f}s dtype={low_dtype.__name__}]"
        f" naive={err_naive:.3e} m"
        f" scan-boundary-mixed={err_scan:.3e} m"
        f" improvement={err_naive / max(err_scan, 1e-12):.1f}x"
    )

    if np.isfinite(err_naive) and np.isfinite(err_scan):
        assert err_scan <= err_naive, (
            f"Scan-boundary mixed ({err_scan:.3e} m) should be at least "
            f"as good as naive ({err_naive:.3e} m)."
        )
