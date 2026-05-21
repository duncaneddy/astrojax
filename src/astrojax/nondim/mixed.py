"""Mixed-precision helpers for nondim integration.

Lets you store the propagation state in low-precision (bfloat16 for ML
training and RL rollouts) while doing the math at module precision
(typically float32). Mirrors the TensorCore-style pattern used by ML
training frameworks: low-precision storage, higher-precision math.

**The right pattern is to keep the entire rollout carry in
high-precision and only cast at the boundary.** Quantizing the state
back to bfloat16 between every integrator step adds rounding noise
that defeats the purpose. Run the whole ``jax.lax.scan`` in float32 and
cast to bfloat16 once at scan entry/exit::

    set_dtype(jnp.float32)
    units = UnitSystem.from_orbit(a, mu)
    dyn = nondim_orbit_dynamics(eop, epoch_0, units, config)

    @jax.jit
    def rollout(state_bf16, n_steps):
        # 1. lift to compute precision once
        state_f32 = state_bf16.astype(jnp.float32)
        t0_f32 = jnp.float32(0.0)
        dt_f32 = jnp.float32(DT / units.TU)

        def step(carry, _):
            t, s = carry
            return (t + dt_f32, dp54_step(dyn, t, s, dt_f32).state), None

        # 2. integrate entirely in float32
        (_t_final, state_final_f32), _ = jax.lax.scan(
            step, (t0_f32, state_f32), None, length=n_steps,
        )
        # 3. cast back to storage precision once at the boundary
        return state_final_f32.astype(jnp.bfloat16)

The two helpers below cover narrower cases:

- ``mixed_precision_dynamics``: wraps a single derivative call so its
  caller can hand it bfloat16 and receive bfloat16, while internal math
  stays at compute precision. Useful when you want to call the
  derivative as a black box from code that lives in bfloat16. Note that
  if you embed this in a scan with a bfloat16 carry, you pay the
  per-step quantization cost described above.

- ``cast_state``: a tiny convenience for the scan-boundary cast.
  Equivalent to ``state.astype(target_dtype)`` but reads more clearly
  in user code that mixes precisions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from astrojax.config import get_dtype


def mixed_precision_dynamics(
    dynamics_fn: Callable[..., Any],
    storage_dtype: DTypeLike,
    compute_dtype: DTypeLike | None = None,
) -> Callable[..., Any]:
    """Wrap a dynamics function so its I/O is in ``storage_dtype`` but
    internal derivative computation runs in ``compute_dtype``.

    The compute precision defaults to ``get_dtype()`` at wrap time, so
    the wrapper composes naturally with ``set_dtype()``: set the module
    to compute precision, build the dynamics, then wrap.

    Args:
        dynamics_fn: A dynamics closure with signature
            ``(t, state) -> deriv``, e.g. one returned by
            ``nondim_orbit_dynamics``.
        storage_dtype: The dtype exposed at the function boundary
            (typically a low-precision dtype like ``bfloat16``).
        compute_dtype: The dtype used internally; defaults to
            ``get_dtype()`` at wrap time.

    Returns:
        A wrapped dynamics function with ``storage_dtype`` I/O.
    """
    if compute_dtype is None:
        compute_dtype = get_dtype()

    def wrapped(t, state, *args, **kwargs):
        t_c = jnp.asarray(t, dtype=compute_dtype)
        state_c = jnp.asarray(state, dtype=compute_dtype)
        deriv_c = dynamics_fn(t_c, state_c, *args, **kwargs)
        return deriv_c.astype(storage_dtype)

    return wrapped


def cast_state(state: ArrayLike, target_dtype: DTypeLike) -> Any:
    """Cast a state vector to ``target_dtype``.

    Trivial wrapper around ``jnp.asarray(state, dtype=target_dtype)`` that
    reads clearly in mixed-precision rollout code. Use at scan
    entry/exit to switch between storage and compute precisions::

        state_f32 = cast_state(state_bf16, jnp.float32)   # scan entry
        # ... integrate in float32 ...
        state_bf16 = cast_state(state_f32, jnp.bfloat16)  # scan exit
    """
    return jnp.asarray(state, dtype=target_dtype)
