"""Type definitions for Space Weather data.

Provides the core data type for space weather storage and lookup:

- :class:`SpaceWeatherData`: Immutable container holding sorted space weather
  arrays for JIT-compatible lookup via ``jnp.searchsorted``.

``SpaceWeatherData`` is a :class:`~typing.NamedTuple`, which JAX treats as a
pytree automatically. This means it works seamlessly with ``jax.jit``,
``jax.vmap``, and ``jax.lax`` control flow primitives.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class SpaceWeatherData(NamedTuple):
    """Space weather data for JIT-compatible lookups.

    Stores space weather values as sorted JAX arrays, enabling O(log n) lookup
    inside ``jax.jit`` via ``jnp.searchsorted``. Missing values (e.g. Kp/Ap
    in monthly-predicted regions) are stored as NaN.

    Attributes:
        mjd: Sorted Modified Julian Dates (one per day), shape ``(N,)``.
        kp: 3-hourly Kp indices (0.0-9.0 scale), shape ``(N, 8)``.
        ap: 3-hourly Ap indices, shape ``(N, 8)``.
        ap_daily: Daily average Ap index, shape ``(N,)``.
        f107_obs: Observed 10.7 cm solar radio flux [sfu], shape ``(N,)``.
        f107_adj: Adjusted 10.7 cm solar radio flux [sfu], shape ``(N,)``.
        f107_obs_ctr81: 81-day centered average observed F10.7, shape ``(N,)``.
        f107_obs_lst81: 81-day last average observed F10.7, shape ``(N,)``.
        f107_adj_ctr81: 81-day centered average adjusted F10.7, shape ``(N,)``.
        f107_adj_lst81: 81-day last average adjusted F10.7, shape ``(N,)``.
        mjd_min: Scalar, first MJD in the dataset.
        mjd_max: Scalar, last MJD in the dataset.
    """

    mjd: Array
    kp: Array
    ap: Array
    ap_daily: Array
    f107_obs: Array
    f107_adj: Array
    f107_obs_ctr81: Array
    f107_obs_lst81: Array
    f107_adj_ctr81: Array
    f107_adj_lst81: Array
    mjd_min: Array
    mjd_max: Array
