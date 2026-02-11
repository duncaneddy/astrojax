"""JIT-compatible EOP interpolation and query functions.

All functions use only JAX primitives (``jnp.searchsorted``, array indexing,
``jnp.where``) and are fully compatible with ``jax.jit``, ``jax.vmap``,
and ``jax.grad``.

The ``extrapolation`` parameter is a Python string resolved at trace time,
following the same pattern as ``ForceModelConfig`` booleans.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.eop._types import EOPData, EOPExtrapolation


def _interpolate_scalar(
    eop: EOPData,
    mjd: Array,
    values: Array,
    extrapolation: EOPExtrapolation = EOPExtrapolation.HOLD,
) -> Array:
    """Linearly interpolate a single EOP field at the given MJD.

    Uses ``jnp.searchsorted`` for O(log n) lookup, then linear interpolation
    between bracketing points. Out-of-range queries are handled by the
    extrapolation mode.

    Args:
        eop: EOP dataset with sorted MJD array.
        mjd: Scalar MJD to query.
        values: The EOP field array to interpolate, shape ``(N,)``.
        extrapolation: Extrapolation mode for out-of-range queries.

    Returns:
        Interpolated scalar value.
    """
    n = eop.mjd.shape[0]

    # Binary search: idx is the insertion point (right side)
    idx = jnp.searchsorted(eop.mjd, mjd, side="right")

    # Bracket indices, clamped to valid range
    idx_lo = jnp.clip(idx - 1, 0, n - 1)
    idx_hi = jnp.clip(idx, 0, n - 1)

    mjd_lo = eop.mjd[idx_lo]
    mjd_hi = eop.mjd[idx_hi]
    val_lo = values[idx_lo]
    val_hi = values[idx_hi]

    # Linear interpolation fraction (safe division: if mjd_lo == mjd_hi, frac=0)
    dmjd = mjd_hi - mjd_lo
    frac = jnp.where(dmjd > 0.0, (mjd - mjd_lo) / dmjd, 0.0)
    interpolated = val_lo + frac * (val_hi - val_lo)

    # Extrapolation handling
    in_range = (mjd >= eop.mjd_min) & (mjd <= eop.mjd_max)

    if extrapolation == EOPExtrapolation.ZERO:
        return jnp.where(in_range, interpolated, 0.0)
    else:
        # HOLD: clamp to boundary values (already done by clip)
        return interpolated


def get_ut1_utc(
    eop: EOPData,
    mjd: ArrayLike,
    extrapolation: EOPExtrapolation = EOPExtrapolation.HOLD,
) -> Array:
    """Query UT1-UTC offset at the given MJD.

    Args:
        eop: EOP dataset.
        mjd: Modified Julian Date to query.
        extrapolation: Extrapolation mode for out-of-range queries.

    Returns:
        UT1-UTC offset [seconds].

    Examples:
        ```python
        from astrojax.eop import zero_eop, get_ut1_utc
        eop = zero_eop()
        ut1_utc = get_ut1_utc(eop, 59569.0)
        ```
    """
    mjd = jnp.asarray(mjd, dtype=eop.mjd.dtype)
    return _interpolate_scalar(eop, mjd, eop.ut1_utc, extrapolation)


def get_pm(
    eop: EOPData,
    mjd: ArrayLike,
    extrapolation: EOPExtrapolation = EOPExtrapolation.HOLD,
) -> tuple[Array, Array]:
    """Query polar motion components at the given MJD.

    Args:
        eop: EOP dataset.
        mjd: Modified Julian Date to query.
        extrapolation: Extrapolation mode for out-of-range queries.

    Returns:
        Tuple of (pm_x, pm_y) polar motion components [rad].

    Examples:
        ```python
        from astrojax.eop import zero_eop, get_pm
        eop = zero_eop()
        pm_x, pm_y = get_pm(eop, 59569.0)
        ```
    """
    mjd = jnp.asarray(mjd, dtype=eop.mjd.dtype)
    pm_x = _interpolate_scalar(eop, mjd, eop.pm_x, extrapolation)
    pm_y = _interpolate_scalar(eop, mjd, eop.pm_y, extrapolation)
    return pm_x, pm_y


def get_dxdy(
    eop: EOPData,
    mjd: ArrayLike,
    extrapolation: EOPExtrapolation = EOPExtrapolation.HOLD,
) -> tuple[Array, Array]:
    """Query celestial pole offsets at the given MJD.

    Values may be NaN if the data source does not include dX/dY for the
    requested epoch (e.g. far-future predictions).

    Args:
        eop: EOP dataset.
        mjd: Modified Julian Date to query.
        extrapolation: Extrapolation mode for out-of-range queries.

    Returns:
        Tuple of (dX, dY) celestial pole offsets [rad].

    Examples:
        ```python
        from astrojax.eop import zero_eop, get_dxdy
        eop = zero_eop()
        dx, dy = get_dxdy(eop, 59569.0)
        ```
    """
    mjd = jnp.asarray(mjd, dtype=eop.mjd.dtype)
    dx = _interpolate_scalar(eop, mjd, eop.dX, extrapolation)
    dy = _interpolate_scalar(eop, mjd, eop.dY, extrapolation)
    return dx, dy


def get_lod(
    eop: EOPData,
    mjd: ArrayLike,
    extrapolation: EOPExtrapolation = EOPExtrapolation.HOLD,
) -> Array:
    """Query length-of-day excess at the given MJD.

    The value may be NaN if the data source does not include LOD for the
    requested epoch (e.g. far-future predictions).

    Args:
        eop: EOP dataset.
        mjd: Modified Julian Date to query.
        extrapolation: Extrapolation mode for out-of-range queries.

    Returns:
        Length of day excess [seconds].

    Examples:
        ```python
        from astrojax.eop import zero_eop, get_lod
        eop = zero_eop()
        lod = get_lod(eop, 59569.0)
        ```
    """
    mjd = jnp.asarray(mjd, dtype=eop.mjd.dtype)
    return _interpolate_scalar(eop, mjd, eop.lod, extrapolation)


def get_eop(
    eop: EOPData,
    mjd: ArrayLike,
    extrapolation: EOPExtrapolation = EOPExtrapolation.HOLD,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Query all EOP values at the given MJD.

    Convenience function that returns all six EOP parameters at once.

    Args:
        eop: EOP dataset.
        mjd: Modified Julian Date to query.
        extrapolation: Extrapolation mode for out-of-range queries.

    Returns:
        Tuple of (pm_x, pm_y, ut1_utc, lod, dX, dY).
        Units: pm_x/pm_y [rad], ut1_utc [s], lod [s], dX/dY [rad].

    Examples:
        ```python
        from astrojax.eop import zero_eop, get_eop
        eop = zero_eop()
        pm_x, pm_y, ut1_utc, lod, dx, dy = get_eop(eop, 59569.0)
        ```
    """
    mjd = jnp.asarray(mjd, dtype=eop.mjd.dtype)
    pm_x = _interpolate_scalar(eop, mjd, eop.pm_x, extrapolation)
    pm_y = _interpolate_scalar(eop, mjd, eop.pm_y, extrapolation)
    ut1_utc = _interpolate_scalar(eop, mjd, eop.ut1_utc, extrapolation)
    lod = _interpolate_scalar(eop, mjd, eop.lod, extrapolation)
    dx = _interpolate_scalar(eop, mjd, eop.dX, extrapolation)
    dy = _interpolate_scalar(eop, mjd, eop.dY, extrapolation)
    return pm_x, pm_y, ut1_utc, lod, dx, dy
