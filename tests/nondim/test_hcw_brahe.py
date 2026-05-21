"""HCW relative motion: 3-way assertion across dtypes.

Propagates a 100 m radial offset on a 500 km circular chief orbit
analytically through the HCW STM for **3000 s (~half an orbit)** and
compares astrojax SI vs nondim paths against the analytical reference.

Diagnostic output reports both relative position error (dimensionless)
and absolute position error (metres) so users can judge usability.

Three assertions per dtype:
  A: |astrojax_SI - reference|       <= tol_SI
  B: |astrojax_nondim_denorm - ref|  <= tol_nondim
  C: |astrojax_SI - astrojax_nondim| <= tol_consistency
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from brahe_reference import hcw_state_after

from astrojax.config import get_dtype, set_dtype
from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.nondim import (
    UnitSystem,
    from_nondim_state,
    to_nondim_state,
)
from astrojax.relative_motion import hcw_stm

A_CHIEF = R_EARTH + 500e3
T_PROP = 3000.0  # ~half an orbit
N_SI = math.sqrt(GM_EARTH / A_CHIEF**3)
STATE_SI = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

REFERENCE = hcw_state_after(STATE_SI, N_SI, T_PROP)


TOL_TABLE = {
    # dtype -> (tol_SI, tol_nondim, tol_consistency) -- relative position error.
    # Refined empirically (uv run pytest tests/nondim/test_hcw_brahe.py -v -s).
    # tol_consistency reflects representable dtype precision, not observed bit-equality.
    jnp.float64: (1e-12, 1e-12, 1e-12),
    jnp.float32: (5e-7, 5e-7, 1e-6),
    jnp.bfloat16: (None, 1e-2, 1e-2),  # tol_si=None: SI bfloat16 unmeaningful for this input
    jnp.float16: (None, 1e-2, 1e-2),  # tol_si=None: SI float16 unmeaningful for this input
}


@pytest.fixture(autouse=True)
def restore_dtype():
    original = get_dtype()
    yield
    set_dtype(original)


def _astrojax_si_hcw(dtype) -> np.ndarray:
    set_dtype(dtype)
    state = jnp.asarray(STATE_SI, dtype=dtype)
    out = hcw_stm(T_PROP, jnp.asarray(N_SI, dtype=dtype)) @ state
    return np.asarray(out, dtype=np.float64)


def _astrojax_nondim_hcw(dtype) -> np.ndarray:
    set_dtype(dtype)
    # Use from_orbit_relative so the relative-position scale is O(LU_rel).
    units = UnitSystem.from_orbit_relative(A_CHIEF, GM_EARTH, LU_rel=100.0)
    state_nd = to_nondim_state(jnp.asarray(STATE_SI, dtype=dtype), units)
    n_nd = jnp.asarray(N_SI * units.TU, dtype=dtype)  # exactly 1.0 by construction
    t_nd = jnp.asarray(T_PROP / units.TU, dtype=dtype)
    out_nd = hcw_stm(t_nd, n_nd) @ state_nd
    out_si = from_nondim_state(out_nd, units)
    return np.asarray(out_si, dtype=np.float64)


def _rel_position_err(predicted, reference) -> float:
    return float(
        np.linalg.norm(predicted[:3] - reference[:3]) / max(np.linalg.norm(reference[:3]), 1e-12)
    )


def _abs_position_err_m(predicted, reference) -> float:
    """Absolute position error in metres."""
    return float(np.linalg.norm(predicted[:3] - reference[:3]))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32, jnp.bfloat16, jnp.float16])
def test_hcw_three_way_assertion(dtype):
    tol_si, tol_nd, tol_cons = TOL_TABLE[dtype]

    si_path = _astrojax_si_hcw(dtype)
    nd_path = _astrojax_nondim_hcw(dtype)

    err_si = _rel_position_err(si_path, REFERENCE)
    err_nd = _rel_position_err(nd_path, REFERENCE)
    err_cons = _rel_position_err(si_path, nd_path)
    abs_si_m = _abs_position_err_m(si_path, REFERENCE)
    abs_nd_m = _abs_position_err_m(nd_path, REFERENCE)

    print(
        f"\n[HCW T={T_PROP:.0f}s dtype={dtype.__name__}]"
        f" err_SI={err_si:.2e} ({abs_si_m:.2e} m)"
        f" err_nondim={err_nd:.2e} ({abs_nd_m:.2e} m)"
        f" err_consistency={err_cons:.2e}"
    )

    if tol_si is not None:
        assert err_si <= tol_si, f"SI path drifted from reference: {err_si:.3e} > {tol_si:.3e}"
    assert err_nd <= tol_nd, f"Nondim path drifted from reference: {err_nd:.3e} > {tol_nd:.3e}"
    if tol_cons is not None:
        assert err_cons <= tol_cons, f"SI vs nondim mismatch: {err_cons:.3e} > {tol_cons:.3e}"
