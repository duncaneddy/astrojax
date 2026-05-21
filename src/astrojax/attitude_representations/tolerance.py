"""Dtype-adaptive tolerance for attitude representation comparisons.

Provides ``get_attitude_epsilon`` which returns a tolerance that scales
with the configured float dtype, following the same pattern as
:func:`astrojax.config.get_epoch_eq_tolerance`.
"""

from __future__ import annotations

import jax.numpy as jnp

from astrojax.config import get_dtype


def get_attitude_epsilon() -> float:
    """Return the dtype-adaptive tolerance for attitude comparisons.

    The tolerance scales with the precision of the configured float dtype:

    - ``float64``:  1e-12
    - ``float32``:  1e-6
    - ``float16``:  1e-3
    - ``bfloat16``: 1e-3

    Returns:
        float: Absolute tolerance for element-wise comparisons.
    """
    dtype = get_dtype()
    if dtype == jnp.float64:
        return 1e-12
    if dtype == jnp.float32:
        return 1e-6
    # float16 and bfloat16
    return 1e-3
