"""Type definitions for Earth Orientation Parameters (EOP).

Provides the core data types for EOP storage and lookup:

- :class:`EOPData`: Immutable container holding sorted EOP arrays for
  JIT-compatible interpolation via ``jnp.searchsorted``.
- :class:`EOPExtrapolation`: Controls behavior when querying outside the
  data range.

``EOPData`` is a :class:`~typing.NamedTuple`, which JAX treats as a pytree
automatically. This means it works seamlessly with ``jax.jit``,
``jax.vmap``, and ``jax.lax`` control flow primitives.
"""

from __future__ import annotations

import enum
from typing import NamedTuple

from jax import Array


class EOPData(NamedTuple):
    """Earth Orientation Parameter data for JIT-compatible lookups.

    Stores EOP values as sorted JAX arrays, enabling O(log n) interpolation
    inside ``jax.jit`` via ``jnp.searchsorted``. Missing optional values
    (dX, dY, lod in prediction regions) are stored as NaN.

    Attributes:
        mjd: Sorted Modified Julian Dates, shape ``(N,)``.
        pm_x: Polar motion x-component [rad], shape ``(N,)``.
        pm_y: Polar motion y-component [rad], shape ``(N,)``.
        ut1_utc: UT1-UTC offset [seconds], shape ``(N,)``.
        dX: Celestial pole offset X [rad], shape ``(N,)``. NaN where missing.
        dY: Celestial pole offset Y [rad], shape ``(N,)``. NaN where missing.
        lod: Length of day excess [seconds], shape ``(N,)``. NaN where missing.
        mjd_min: Scalar, first MJD in the dataset.
        mjd_max: Scalar, last MJD in the dataset.
        mjd_last_lod: Scalar, last MJD with valid LOD data.
        mjd_last_dxdy: Scalar, last MJD with valid dX/dY data.
    """

    mjd: Array
    pm_x: Array
    pm_y: Array
    ut1_utc: Array
    dX: Array
    dY: Array
    lod: Array
    mjd_min: Array
    mjd_max: Array
    mjd_last_lod: Array
    mjd_last_dxdy: Array


class EOPExtrapolation(enum.Enum):
    """Extrapolation mode for EOP queries outside the data range.

    Resolved at trace time (Python string), not at runtime. This follows
    the same pattern as ``ForceModelConfig`` booleans.

    Attributes:
        HOLD: Clamp to the nearest boundary value.
        ZERO: Return zero for out-of-range queries.
    """

    HOLD = "hold"
    ZERO = "zero"
