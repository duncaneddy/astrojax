"""Module-wide floating-point precision configuration.

Provides ``set_dtype`` and ``get_dtype`` to control the float dtype used
throughout astrojax.  The default is ``jnp.float32`` for GPU/TPU
compatibility.  Switching to ``jnp.float64`` automatically enables
JAX's 64-bit mode (``jax_enable_x64``).

Call ``set_dtype`` **before** any JIT compilation, just like JAX's own
``jax.config.update("jax_enable_x64", True)``.  Under JIT, ``get_dtype()``
runs during tracing and its result is baked into the compiled program.
JAX retraces when input dtypes change, so passing float64 inputs after
``set_dtype(jnp.float64)`` triggers a correct retrace.

Integer components (e.g. Epoch ``_jd``) are always ``jnp.int32``
regardless of this setting.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_VALID_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64)

_dtype = jnp.float32


def set_dtype(dtype) -> None:
    """Set the module-wide float dtype for astrojax.

    Must be called **before** any ``jax.jit`` compilation.  In eager mode
    the change takes effect immediately.  Under JIT, ``get_dtype()`` runs
    during tracing and its value is baked into the compiled program.

    If *dtype* is ``jnp.float64``, JAX's 64-bit mode is automatically
    enabled via ``jax.config.update("jax_enable_x64", True)``.

    Args:
        dtype: One of ``jnp.float16``, ``jnp.bfloat16``, ``jnp.float32``,
            or ``jnp.float64``.

    Raises:
        ValueError: If *dtype* is not a supported float type.
    """
    global _dtype
    if dtype not in _VALID_DTYPES:
        raise ValueError(
            f"Unsupported dtype {dtype}. Must be one of: "
            f"jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64"
        )
    if dtype == jnp.float64:
        jax.config.update("jax_enable_x64", True)
    _dtype = dtype


def get_dtype():
    """Return the current module-wide float dtype.

    Returns:
        The active float dtype (default ``jnp.float32``).
    """
    return _dtype


def get_epoch_eq_tolerance() -> float:
    """Return the dtype-adaptive tolerance for Epoch equality comparisons.

    The tolerance scales with the precision of the configured float dtype:

    - ``float16``:  0.1 s
    - ``bfloat16``: 0.1 s
    - ``float32``:  1e-3 s
    - ``float64``:  1e-9 s

    Returns:
        float: Tolerance in seconds.
    """
    if _dtype == jnp.float64:
        return 1e-9
    if _dtype == jnp.float32:
        return 1e-3
    # float16 and bfloat16
    return 0.1
