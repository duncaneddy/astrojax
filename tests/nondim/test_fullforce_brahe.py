"""Full force model (5x5 + Sun + Moon + SRP + drag): 3-way assertion.

The full-force reference comes from astrojax-SI at float64 (see
``tests/nondim/brahe_reference.py``). Until brahe exposes a matching
SI force-model API, this is the best ground truth available.

What this test buys us:
- Assertion (A) [SI dtype vs SI float64]: catches dtype-induced drift
  in the astrojax SI path.
- Assertion (B) [nondim -> denorm vs SI float64]: catches nondim layer
  drift.
- Assertion (C) [SI dtype vs nondim dtype]: catches nondim layer
  *relative to its own SI path* even if both share an absolute bias.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brahe_reference import full_force_state_after

from astrojax.config import get_dtype, set_dtype
from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.eop import zero_eop
from astrojax.epoch import Epoch
from astrojax.integrators import dp54_step
from astrojax.nondim import (
    UnitSystem,
    from_nondim_state,
    nondim_orbit_dynamics,
    to_nondim_state,
)
from astrojax.orbit_dynamics import ForceModelConfig, create_orbit_dynamics
from astrojax.orbit_dynamics.gravity import GravityModel
from astrojax.space_weather import zero_space_weather

A = R_EARTH + 500e3
# Propagation duration chosen to keep JIT-cached integration tractable
# (~1 min wall-clock for the float64 reference). The test validates
# factory composition correctness across dtypes; absolute long-duration
# accuracy is reported in the diagnostic output rather than asserted.
DT = 60.0
N_STEPS = 10
T_PROP = DT * N_STEPS  # 10 minutes

INITIAL_STATE_SI = np.array(
    [A, 0.0, 0.0, 0.0, math.sqrt(GM_EARTH / A), 0.0],
    dtype=np.float64,
)

# Tolerances refined empirically (uv run pytest ... -v -s -o addopts=, 2026-05-21):
#   float64:   err_SI=0,        err_nondim=2e-16,  err_consistency=2e-16
#   float32:   err_SI=3.2e-7,   err_nondim=4.5e-7, err_consistency=1.6e-7
#   bfloat16:  err_SI=6.4e-1,   err_nondim=2.5e-1, err_consistency=4.0e-1
# At bfloat16 with a full perturbed force model, both paths diverge
# substantially over the 10-minute window — but nondim degrades 2.5×
# slower than SI, demonstrating the precision-magnitude advantage.
# tol_consistency is loose at bfloat16 because the two paths take
# different operation-ordering routes through unit-converted scalars.
TOL_TABLE = {
    jnp.float64: (1e-10, 1e-10, 1e-10),
    jnp.float32: (5e-6, 5e-6, 5e-6),
    jnp.bfloat16: (None, 5e-1, 5e-1),  # characterization, not regression
    jnp.float16: (None, None, None),  # characterization-only; SI overflows in |r|^3 term
}


@pytest.fixture(autouse=True)
def restore_dtype():
    original = get_dtype()
    yield
    set_dtype(original)


@pytest.fixture(scope="module")
def context():
    return dict(
        epoch=Epoch(2024, 1, 1, 12, 0, 0.0),
        eop=zero_eop(),
        space_weather=zero_space_weather(),
    )


@pytest.fixture(scope="module")
def reference(context):
    return full_force_state_after(
        INITIAL_STATE_SI,
        context["epoch"],
        T_PROP,
        context["eop"],
        context["space_weather"],
    )


def _build_config():
    """Build the 5x5 LEO force model config; must match the reference.

    Uses Harris-Priester (analytical) rather than NRLMSISE-00 to keep
    JIT compile times tractable. The point of this test is factory
    composition correctness, not atmospheric-model accuracy.
    """
    gravity_model = GravityModel.from_type("JGM3")
    return ForceModelConfig(
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


def _scan_propagate(dyn, t0, state0, dt, n_steps):
    """JIT-compiled fixed-step DP54 propagation via jax.lax.scan."""

    def step(carry, _):
        t, state = carry
        result = dp54_step(dyn, t, state, dt)
        return (t + result.dt_used, result.state), None

    @jax.jit
    def run(t0, state0):
        (t_final, state_final), _ = jax.lax.scan(
            step,
            (t0, state0),
            None,
            length=n_steps,
        )
        return state_final

    return run(t0, state0)


def _propagate_si(dtype, context) -> np.ndarray:
    set_dtype(dtype)
    config = _build_config()
    dyn = create_orbit_dynamics(
        eop=context["eop"],
        epoch_0=context["epoch"],
        config=config,
        space_weather=context["space_weather"],
    )
    state0 = jnp.asarray(INITIAL_STATE_SI, dtype=dtype)
    t0 = jnp.asarray(0.0, dtype=dtype)
    dt = jnp.asarray(DT, dtype=dtype)
    n_steps = int(T_PROP / DT)
    state = _scan_propagate(dyn, t0, state0, dt, n_steps)
    return np.asarray(state, dtype=np.float64)


def _propagate_nondim(dtype, context) -> np.ndarray:
    set_dtype(dtype)
    units = UnitSystem.from_orbit(A, GM_EARTH)
    config = _build_config()
    dyn = nondim_orbit_dynamics(
        eop=context["eop"],
        epoch_0=context["epoch"],
        units=units,
        config=config,
        space_weather=context["space_weather"],
    )
    state0 = to_nondim_state(
        jnp.asarray(INITIAL_STATE_SI, dtype=dtype),
        units,
    )
    t0 = jnp.asarray(0.0, dtype=dtype)
    dt_nd = jnp.asarray(DT / units.TU, dtype=dtype)
    n_steps = int(T_PROP / DT)
    state_nd = _scan_propagate(dyn, t0, state0, dt_nd, n_steps)
    state_si = from_nondim_state(state_nd, units)
    return np.asarray(state_si, dtype=np.float64)


def _rel_position_err(predicted, reference) -> float:
    return float(
        np.linalg.norm(predicted[:3] - reference[:3]) / max(np.linalg.norm(reference[:3]), 1e-12)
    )


def _abs_position_err_m(predicted, reference) -> float:
    """Absolute position error in metres."""
    return float(np.linalg.norm(predicted[:3] - reference[:3]))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32, jnp.bfloat16, jnp.float16])
def test_full_force_three_way_assertion(dtype, context, reference):
    tol_si, tol_nd, tol_cons = TOL_TABLE[dtype]

    si_path = _propagate_si(dtype, context)
    nd_path = _propagate_nondim(dtype, context)

    err_si = _rel_position_err(si_path, reference)
    err_nd = _rel_position_err(nd_path, reference)
    err_cons = _rel_position_err(si_path, nd_path)
    abs_si_m = _abs_position_err_m(si_path, reference)
    abs_nd_m = _abs_position_err_m(nd_path, reference)

    print(
        f"\n[full-force T={T_PROP:.0f}s dtype={dtype.__name__}]"
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
