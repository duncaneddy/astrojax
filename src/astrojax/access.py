"""Satellite-ground station access (visibility) prediction.

Computes time windows when a satellite is above a minimum elevation angle
as seen from a ground station.  The approach is a three-stage hybrid:

1. **JIT-accelerated elevation grid** — ``vmap`` over a time array
   (the expensive topocentric math).
2. **Python-level window detection** — sign-change detection with NumPy
   (cheap, avoids JAX's fixed-shape constraint).
3. **JIT-accelerated bisection refinement** — ``jax.lax.while_loop``
   per boundary, vmapped across all boundaries.

All angles are in radians and distances in metres unless noted otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.coordinates.geodetic import position_geodetic_to_ecef
from astrojax.coordinates.topocentric import (
    position_enz_to_azel,
    rotation_ellipsoid_to_enz,
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class GroundLocation(NamedTuple):
    """A ground station / observer location.

    Stores geodetic coordinates together with pre-computed ECEF position
    and ECEF-to-ENZ rotation matrix for efficient repeated use.

    Construct via :func:`ground_location`.
    """

    lon: float  # Longitude [rad]
    lat: float  # Latitude [rad]
    alt: float  # Altitude above WGS84 [m]
    ecef: Array  # Pre-computed ECEF position [x,y,z] [m]
    rot_enz: Array  # Pre-computed 3x3 ECEF→ENZ rotation matrix


class AccessWindow(NamedTuple):
    """A single satellite access (visibility) window.

    All times are in seconds since the reference epoch used by the caller.
    Angles in radians, distances in metres.
    """

    t_rise: float  # Window open time [s]
    t_set: float  # Window close time [s]
    t_max_el: float  # Time of max elevation [s]
    az_rise: float  # Azimuth at rise [rad]
    el_rise: float  # Elevation at rise [rad]
    rng_rise: float  # Range at rise [m]
    az_set: float  # Azimuth at set [rad]
    el_set: float  # Elevation at set [rad]
    rng_set: float  # Range at set [m]
    az_max: float  # Azimuth at max elevation [rad]
    el_max: float  # Max elevation [rad]
    rng_max: float  # Range at max elevation [m]
    duration: float  # Window duration [s]


class AccessResult(NamedTuple):
    """Fixed-shape result from :func:`find_access_windows_jit`.

    All arrays have leading dimension ``max_windows``.  Slots where
    ``valid[i]`` is ``False`` contain dummy values and should be ignored.
    """

    t_rise: Array  # (max_windows,)
    t_set: Array  # (max_windows,)
    t_max_el: Array  # (max_windows,)
    az_rise: Array  # (max_windows,)
    el_rise: Array  # (max_windows,)
    rng_rise: Array  # (max_windows,)
    az_set: Array  # (max_windows,)
    el_set: Array  # (max_windows,)
    rng_set: Array  # (max_windows,)
    az_max: Array  # (max_windows,)
    el_max: Array  # (max_windows,)
    rng_max: Array  # (max_windows,)
    duration: Array  # (max_windows,)
    valid: Array  # (max_windows,) bool
    n_windows: Array  # () int32 scalar


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def ground_location(
    lon: float,
    lat: float,
    alt: float = 0.0,
    use_degrees: bool = True,
) -> GroundLocation:
    """Create a :class:`GroundLocation` with pre-computed ECEF and ENZ rotation.

    Args:
        lon: Longitude.
        lat: Latitude.
        alt: Altitude above WGS84 ellipsoid in metres.
        use_degrees: If ``True`` (default), *lon* and *lat* are in degrees.

    Returns:
        A :class:`GroundLocation` instance.
    """
    dtype = get_dtype()

    if use_degrees:
        lon_rad = dtype(jnp.deg2rad(lon))
        lat_rad = dtype(jnp.deg2rad(lat))
    else:
        lon_rad = dtype(lon)
        lat_rad = dtype(lat)

    x_geod = jnp.array([lon_rad, lat_rad, dtype(alt)])
    ecef = position_geodetic_to_ecef(x_geod)
    rot_enz = rotation_ellipsoid_to_enz(x_geod)

    return GroundLocation(
        lon=float(lon_rad),
        lat=float(lat_rad),
        alt=float(alt),
        ecef=ecef,
        rot_enz=rot_enz,
    )


# ---------------------------------------------------------------------------
# JIT-compatible building blocks
# ---------------------------------------------------------------------------


def compute_elevation(
    sat_ecef: ArrayLike,
    station_ecef: ArrayLike,
    rot_enz: ArrayLike | None = None,
) -> Array:
    """Compute the elevation angle of a satellite as seen from a station.

    This is a JIT-compatible, vmappable building block.

    Args:
        sat_ecef: Satellite ECEF position ``[x, y, z]`` in metres.
        station_ecef: Station ECEF position ``[x, y, z]`` in metres.
        rot_enz: Optional pre-computed 3x3 ECEF-to-ENZ rotation matrix.
            If ``None``, it is computed from the station position (slower
            for repeated calls with the same station).

    Returns:
        Scalar elevation angle in radians.
    """
    dtype = get_dtype()
    sat_ecef = jnp.asarray(sat_ecef, dtype=dtype)
    station_ecef = jnp.asarray(station_ecef, dtype=dtype)

    if rot_enz is None:
        from astrojax.coordinates.geodetic import position_ecef_to_geodetic

        x_geod = position_ecef_to_geodetic(station_ecef)
        rot_enz = rotation_ellipsoid_to_enz(x_geod)

    rot_enz = jnp.asarray(rot_enz, dtype=dtype)
    r_enz = rot_enz @ (sat_ecef - station_ecef)

    e, n, z = r_enz[0], r_enz[1], r_enz[2]
    horiz = jnp.sqrt(e * e + n * n)
    return jnp.arctan2(z, horiz)


def compute_azel(
    sat_ecef: ArrayLike,
    station_ecef: ArrayLike,
    rot_enz: ArrayLike | None = None,
) -> Array:
    """Compute azimuth, elevation, and range of a satellite from a station.

    Args:
        sat_ecef: Satellite ECEF position ``[x, y, z]`` in metres.
        station_ecef: Station ECEF position ``[x, y, z]`` in metres.
        rot_enz: Optional pre-computed 3x3 ECEF-to-ENZ rotation matrix.

    Returns:
        ``[azimuth, elevation, range]`` — azimuth in ``[0, 2pi)`` rad,
        elevation in ``[-pi/2, pi/2]`` rad, range in metres.
    """
    dtype = get_dtype()
    sat_ecef = jnp.asarray(sat_ecef, dtype=dtype)
    station_ecef = jnp.asarray(station_ecef, dtype=dtype)

    if rot_enz is None:
        from astrojax.coordinates.geodetic import position_ecef_to_geodetic

        x_geod = position_ecef_to_geodetic(station_ecef)
        rot_enz = rotation_ellipsoid_to_enz(x_geod)

    rot_enz = jnp.asarray(rot_enz, dtype=dtype)
    r_enz = rot_enz @ (sat_ecef - station_ecef)
    return position_enz_to_azel(r_enz)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _elevation_at_time(
    position_ecef_fn: Callable,
    station_ecef: Array,
    rot_enz: Array,
    t: ArrayLike,
) -> Array:
    """Evaluate elevation at a single time (JIT-compatible)."""
    sat_ecef = position_ecef_fn(t)
    return compute_elevation(sat_ecef, station_ecef, rot_enz)


def _azel_at_time(
    position_ecef_fn: Callable,
    station_ecef: Array,
    rot_enz: Array,
    t: ArrayLike,
) -> Array:
    """Evaluate az/el/range at a single time (JIT-compatible)."""
    sat_ecef = position_ecef_fn(t)
    return compute_azel(sat_ecef, station_ecef, rot_enz)


def _refine_boundary(
    elevation_fn: Callable,
    t_lo: float,
    t_hi: float,
    tol: float = 0.001,
    max_iter: int = 50,
) -> Array:
    """Bisection search for an elevation zero crossing.

    Uses ``jax.lax.while_loop`` for JIT compatibility.

    Args:
        elevation_fn: ``f(t) -> elevation`` (scalar in, scalar out).
        t_lo: Lower bound (elevation < threshold at this end).
        t_hi: Upper bound (elevation >= threshold at this end).
        tol: Convergence tolerance in seconds.
        max_iter: Maximum iterations.

    Returns:
        Refined time of the zero crossing.
    """
    dtype = get_dtype()
    t_lo = jnp.asarray(t_lo, dtype=dtype)
    t_hi = jnp.asarray(t_hi, dtype=dtype)
    tol = jnp.asarray(tol, dtype=dtype)

    el_lo = elevation_fn(t_lo)

    def cond(state):
        lo, hi, i = state
        return ((hi - lo) > tol) & (i < max_iter)

    def body(state):
        lo, hi, i = state
        mid = (lo + hi) * dtype(0.5)
        el_mid = elevation_fn(mid)
        # If el_mid has same sign as el_lo, the crossing is in [mid, hi]
        same_sign = (el_mid * el_lo) >= dtype(0.0)
        lo = jnp.where(same_sign, mid, lo)
        hi = jnp.where(same_sign, hi, mid)
        return (lo, hi, i + 1)

    init = (t_lo, t_hi, jnp.int32(0))
    lo_final, hi_final, _ = jax.lax.while_loop(cond, body, init)
    return (lo_final + hi_final) * dtype(0.5)


def _find_max_elevation(
    elevation_fn: Callable,
    t_lo: float,
    t_hi: float,
    n_samples: int = 20,
) -> Array:
    """Find time of maximum elevation in an interval using golden section.

    Uses a coarse grid to find the initial bracket, then golden section
    search to refine.

    Args:
        elevation_fn: ``f(t) -> elevation``.
        t_lo: Start of interval.
        t_hi: End of interval.
        n_samples: Number of initial grid samples.

    Returns:
        Time of maximum elevation within the interval.
    """
    dtype = get_dtype()
    t_lo = jnp.asarray(t_lo, dtype=dtype)
    t_hi = jnp.asarray(t_hi, dtype=dtype)

    # Coarse grid to find approximate max
    ts = jnp.linspace(t_lo, t_hi, n_samples)
    els = jax.vmap(elevation_fn)(ts)
    i_max = jnp.argmax(els)

    # Bracket around maximum
    bracket_lo = jnp.where(i_max > 0, ts[i_max - 1], t_lo)
    bracket_hi = jnp.where(i_max < n_samples - 1, ts[i_max + 1], t_hi)

    # Golden section search
    gr = dtype((jnp.sqrt(dtype(5.0)) - dtype(1.0)) / dtype(2.0))

    def cond(state):
        a, b, i = state
        return ((b - a) > dtype(0.1)) & (i < 50)

    def body(state):
        a, b, i = state
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = elevation_fn(c)
        fd = elevation_fn(d)
        # We want max, so narrow toward higher value
        a = jnp.where(fc < fd, c, a)
        b = jnp.where(fc < fd, b, d)
        return (a, b, i + 1)

    init = (bracket_lo, bracket_hi, jnp.int32(0))
    a_final, b_final, _ = jax.lax.while_loop(cond, body, init)
    return (a_final + b_final) * dtype(0.5)


def _detect_windows(
    elevations: np.ndarray,
    times: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    """Detect access windows from an elevation time series (Python/NumPy).

    Finds contiguous intervals where ``elevations >= threshold``.

    Args:
        elevations: 1-D elevation array in radians.
        times: Corresponding 1-D time array.
        threshold: Minimum elevation threshold in radians.

    Returns:
        List of ``(i_rise, i_set)`` index pairs where a rising crossing
        occurs at index ``i_rise`` and a setting crossing at ``i_set``.
    """
    above = elevations >= threshold
    # Detect transitions: +1 = rise, -1 = set
    transitions = np.diff(above.astype(np.int8))

    rises = np.where(transitions == 1)[0]  # index before crossing
    sets = np.where(transitions == -1)[0]

    windows = []

    # Handle case where satellite is already above at t_start
    if above[0]:
        if len(sets) > 0:
            first_set = sets[0]
            windows.append((0, first_set))
            sets = sets[1:]
        else:
            # Visible for entire span
            windows.append((0, len(elevations) - 1))
            return windows

    # Pair rises with sets
    for rise_idx in rises:
        # Find the next set after this rise
        later_sets = sets[sets > rise_idx]
        if len(later_sets) > 0:
            set_idx = later_sets[0]
            sets = sets[sets > set_idx]  # consume it
            windows.append((rise_idx, set_idx))
        else:
            # Satellite still above at t_end
            windows.append((rise_idx, len(elevations) - 1))
            break

    return windows


def _detect_crossings_jit(
    elevations: Array,
    min_el: ArrayLike,
    max_windows: int,
) -> tuple[Array, Array, Array, Array]:
    """Detect access windows from an elevation grid (fully JIT-compatible).

    Replaces :func:`_detect_windows` with a ``lax.scan`` that writes into
    fixed-size buffers, making all output shapes static.

    Args:
        elevations: 1-D elevation array in radians, shape ``(n_steps,)``.
        min_el: Minimum elevation threshold in radians (scalar).
        max_windows: Maximum number of windows to detect (static).

    Returns:
        ``(rise_indices, set_indices, valid, n_windows)`` where
        ``rise_indices`` and ``set_indices`` are ``(max_windows,)`` int32,
        ``valid`` is ``(max_windows,)`` bool, and ``n_windows`` is a
        scalar int32.
    """
    n_steps = elevations.shape[0]
    above = elevations >= min_el

    # Pre-allocate fixed-size output buffers
    rise_indices = jnp.zeros(max_windows, dtype=jnp.int32)
    set_indices = jnp.full(max_windows, n_steps - 1, dtype=jnp.int32)

    # Handle "visible at start": if above[0], the first window starts at 0
    start_above = above[0]
    wp_init = jnp.where(start_above, jnp.int32(1), jnp.int32(0))
    # rise_indices[0] already 0, which is correct for "visible at start"

    # Carry: (was_above, in_window, rise_indices, set_indices, write_ptr)
    init_carry = (start_above, start_above, rise_indices, set_indices, wp_init)

    def scan_fn(carry, x):
        was_above, in_window, rises, sets, wp = carry
        above_i, idx = x

        rising = ~was_above & above_i
        falling = was_above & ~above_i

        # Clamp write_ptr to valid range for indexing safety
        safe_wp = jnp.clip(wp, 0, max_windows - 1)
        safe_wp_prev = jnp.clip(wp - 1, 0, max_windows - 1)

        # Rising edge: start a new window at idx-1 (the last below-threshold sample)
        can_write_rise = rising & (wp < max_windows)
        rises = rises.at[safe_wp].set(jnp.where(can_write_rise, idx - 1, rises[safe_wp]))
        wp = jnp.where(can_write_rise, wp + 1, wp)

        # Falling edge: close the current window at idx-1 (last above-threshold sample)
        # This matches _detect_windows convention where set_idx is the last
        # index that is still above threshold.
        can_write_set = falling & (wp > 0)
        # After a rise, wp was incremented, so the set index goes at wp-1
        # Recompute safe_wp_prev after potential wp increment
        safe_wp_prev = jnp.clip(wp - 1, 0, max_windows - 1)
        sets = sets.at[safe_wp_prev].set(jnp.where(can_write_set, idx - 1, sets[safe_wp_prev]))

        new_in_window = jnp.where(rising, True, jnp.where(falling, False, in_window))
        return (above_i, new_in_window, rises, sets, wp), None

    indices = jnp.arange(1, n_steps)
    (_, _, rise_indices, set_indices, n_found), _ = jax.lax.scan(
        scan_fn, init_carry, (above[1:], indices)
    )

    valid = jnp.arange(max_windows) < n_found
    return rise_indices, set_indices, valid, n_found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_access_windows(
    position_ecef_fn: Callable,
    location: GroundLocation,
    t_start: float,
    t_end: float,
    min_elevation: float = 0.0,
    dt: float = 60.0,
    tol: float = 0.001,
    use_degrees: bool = False,
) -> list[AccessWindow]:
    """Find satellite visibility windows from a ground station.

    Takes a callable that returns ECEF position at a given time and finds
    all windows where the satellite exceeds the minimum elevation.

    Args:
        position_ecef_fn: Callable ``f(t) -> Array[3]`` returning the
            satellite ECEF position in metres at time *t* (seconds).
        location: Ground station as a :class:`GroundLocation`.
        t_start: Start time in seconds.
        t_end: End time in seconds.
        min_elevation: Minimum elevation threshold.  Radians by default,
            or degrees if ``use_degrees=True``.
        dt: Coarse grid step size in seconds (default 60).
        tol: Bisection convergence tolerance in seconds (default 0.001).
        use_degrees: If ``True``, interpret *min_elevation* as degrees.

    Returns:
        List of :class:`AccessWindow` instances (may be empty).
    """
    dtype = get_dtype()

    if use_degrees:
        min_el_rad = float(jnp.deg2rad(min_elevation))
    else:
        min_el_rad = float(min_elevation)

    station_ecef = location.ecef
    rot_enz = location.rot_enz

    # --- Stage 1: JIT-accelerated elevation grid ---
    n_steps = int(jnp.ceil((t_end - t_start) / dt)) + 1
    times_jax = jnp.linspace(dtype(t_start), dtype(t_end), n_steps)

    def el_at_t(t):
        return _elevation_at_time(position_ecef_fn, station_ecef, rot_enz, t)

    elevations_jax = jax.vmap(el_at_t)(times_jax)

    # Convert to numpy for window detection
    elevations_np = np.asarray(elevations_jax)
    times_np = np.asarray(times_jax)

    # --- Stage 2: Python-level window detection ---
    raw_windows = _detect_windows(elevations_np, times_np, min_el_rad)

    if not raw_windows:
        return []

    # --- Stage 3: Refine boundaries and collect window info ---
    def shifted_el(t):
        return el_at_t(t) - dtype(min_el_rad)

    results = []
    for i_rise, i_set in raw_windows:
        # Refine rise time
        if i_rise == 0 and elevations_np[0] >= min_el_rad:
            t_rise = float(times_np[0])
        else:
            t_rise = float(
                _refine_boundary(shifted_el, times_np[i_rise], times_np[i_rise + 1], tol)
            )

        # Refine set time
        if i_set == len(elevations_np) - 1 and elevations_np[-1] >= min_el_rad:
            t_set = float(times_np[-1])
        else:
            t_set = float(_refine_boundary(shifted_el, times_np[i_set + 1], times_np[i_set], tol))

        # Find max elevation
        t_max = float(_find_max_elevation(el_at_t, t_rise, t_set))

        # Compute azel at rise, set, max
        def azel_at_t(t):
            return _azel_at_time(position_ecef_fn, station_ecef, rot_enz, t)

        azel_rise = azel_at_t(dtype(t_rise))
        azel_set = azel_at_t(dtype(t_set))
        azel_max = azel_at_t(dtype(t_max))

        results.append(
            AccessWindow(
                t_rise=t_rise,
                t_set=t_set,
                t_max_el=t_max,
                az_rise=float(azel_rise[0]),
                el_rise=float(azel_rise[1]),
                rng_rise=float(azel_rise[2]),
                az_set=float(azel_set[0]),
                el_set=float(azel_set[1]),
                rng_set=float(azel_set[2]),
                az_max=float(azel_max[0]),
                el_max=float(azel_max[1]),
                rng_max=float(azel_max[2]),
                duration=t_set - t_rise,
            )
        )

    return results


def find_access_windows_jit(
    position_ecef_fn: Callable,
    station_ecef: ArrayLike,
    rot_enz: ArrayLike,
    t_start: ArrayLike,
    t_end: ArrayLike,
    min_elevation: ArrayLike,
    max_windows: int,
    n_steps: int,
    tol: float = 0.001,
) -> AccessResult:
    """Find satellite visibility windows — fully JIT-compilable.

    Unlike :func:`find_access_windows`, this function uses fixed-shape
    outputs and ``lax.scan`` for window detection, so it can be composed
    inside larger ``jax.jit`` pipelines.

    Args:
        position_ecef_fn: Callable ``f(t) -> Array[3]`` returning the
            satellite ECEF position in metres at time *t* (seconds).
        station_ecef: Station ECEF position ``[x, y, z]`` in metres.
        rot_enz: Pre-computed 3x3 ECEF-to-ENZ rotation matrix.
        t_start: Start time in seconds (scalar).
        t_end: End time in seconds (scalar).
        min_elevation: Minimum elevation threshold in radians (scalar).
        max_windows: Maximum windows to detect.  **Static** — determines
            output array shapes.  Must be passed via ``static_argnums``.
        n_steps: Number of coarse grid samples.  **Static** — determines
            ``lax.scan`` length.  Must be passed via ``static_argnums``.
        tol: Bisection convergence tolerance in seconds (default 0.001).

    Returns:
        An :class:`AccessResult` with fixed-shape arrays.  Only slots
        where ``valid[i]`` is ``True`` contain meaningful values.
    """
    dtype = get_dtype()
    station_ecef = jnp.asarray(station_ecef, dtype=dtype)
    rot_enz = jnp.asarray(rot_enz, dtype=dtype)
    t_start = jnp.asarray(t_start, dtype=dtype)
    t_end = jnp.asarray(t_end, dtype=dtype)
    min_el = jnp.asarray(min_elevation, dtype=dtype)
    tol = jnp.asarray(tol, dtype=dtype)

    # --- Stage 1: elevation grid via vmap ---
    times = jnp.linspace(t_start, t_end, n_steps)

    def el_at_t(t):
        return _elevation_at_time(position_ecef_fn, station_ecef, rot_enz, t)

    elevations = jax.vmap(el_at_t)(times)

    # --- Stage 2: JIT-compatible window detection ---
    rise_idx, set_idx, valid, n_windows = _detect_crossings_jit(elevations, min_el, max_windows)

    # --- Stage 3: refine boundaries via vmap ---
    def shifted_el(t):
        return el_at_t(t) - min_el

    def refine_one_window(i):
        ri = rise_idx[i]
        si = set_idx[i]

        # Rise refinement: bracket is [times[ri], times[ri+1]]
        # If ri==0 and visible at start, use times[0] directly
        ri_safe = jnp.clip(ri + 1, 0, n_steps - 1)
        at_start = (ri == 0) & (elevations[0] >= min_el)
        t_rise_refined = jnp.where(
            at_start,
            times[0],
            _refine_boundary(shifted_el, times[ri], times[ri_safe], tol),
        )

        # Set refinement: bracket is [times[si+1], times[si]]
        # (reversed: si+1 is below threshold, si is above)
        # If si==n_steps-1 and visible at end, use times[-1] directly
        si_safe = jnp.clip(si + 1, 0, n_steps - 1)
        at_end = (si == n_steps - 1) & (elevations[-1] >= min_el)
        t_set_refined = jnp.where(
            at_end,
            times[-1],
            _refine_boundary(shifted_el, times[si_safe], times[si], tol),
        )

        return t_rise_refined, t_set_refined

    t_rises, t_sets = jax.vmap(refine_one_window)(jnp.arange(max_windows))

    # --- Stage 4: find max elevation via vmap ---
    def max_el_one_window(i):
        return _find_max_elevation(el_at_t, t_rises[i], t_sets[i])

    t_maxes = jax.vmap(max_el_one_window)(jnp.arange(max_windows))

    # --- Stage 5: compute az/el/range at rise, set, max via vmap ---
    def azel_at_t(t):
        return _azel_at_time(position_ecef_fn, station_ecef, rot_enz, t)

    azel_rises = jax.vmap(azel_at_t)(t_rises)
    azel_sets = jax.vmap(azel_at_t)(t_sets)
    azel_maxes = jax.vmap(azel_at_t)(t_maxes)

    durations = t_sets - t_rises

    return AccessResult(
        t_rise=t_rises,
        t_set=t_sets,
        t_max_el=t_maxes,
        az_rise=azel_rises[:, 0],
        el_rise=azel_rises[:, 1],
        rng_rise=azel_rises[:, 2],
        az_set=azel_sets[:, 0],
        el_set=azel_sets[:, 1],
        rng_set=azel_sets[:, 2],
        az_max=azel_maxes[:, 0],
        el_max=azel_maxes[:, 1],
        rng_max=azel_maxes[:, 2],
        duration=durations,
        valid=valid,
        n_windows=n_windows,
    )


def find_all_access_windows(
    position_ecef_fn: Callable,
    station_ecef: ArrayLike,
    rot_enz: ArrayLike,
    t_start: float,
    t_end: float,
    min_elevation: float,
    max_windows: int = 100,
    n_steps: int = 181,
    batch_size: int = 10,
    tol: float = 0.001,
) -> list[AccessWindow]:
    """Find all access windows by paging through the time span.

    Repeatedly calls :func:`find_access_windows_jit` with ``batch_size``
    windows per call, advancing ``t_start`` past the last found window
    each time the batch fills up.  Stops when the batch has room left
    (time span exhausted) or ``max_windows`` total have been collected.

    This is a Python-level convenience wrapper — not itself JIT-compilable
    — but uses the JIT-compiled path for the heavy computation.

    Args:
        position_ecef_fn: Callable ``f(t) -> Array[3]`` returning the
            satellite ECEF position in metres at time *t* (seconds).
        station_ecef: Station ECEF position ``[x, y, z]`` in metres
            (e.g. ``location.ecef``).
        rot_enz: Pre-computed 3x3 ECEF-to-ENZ rotation matrix
            (e.g. ``location.rot_enz``).
        t_start: Start time in seconds.
        t_end: End time in seconds.
        min_elevation: Minimum elevation threshold in radians.
        max_windows: Maximum total windows to return (default 100).
        n_steps: Grid samples per JIT call (default 181).
        batch_size: Windows per JIT call (default 10).  Kept constant
            across calls to avoid JIT recompilation.
        tol: Bisection tolerance in seconds (default 0.001).

    Returns:
        List of :class:`AccessWindow` instances, at most ``max_windows``.
    """
    collected: list[AccessWindow] = []
    current_start = float(t_start)
    t_end_f = float(t_end)

    while current_start < t_end_f and len(collected) < max_windows:
        result = find_access_windows_jit(
            position_ecef_fn,
            station_ecef,
            rot_enz,
            current_start,
            t_end_f,
            min_elevation,
            max_windows=batch_size,
            n_steps=n_steps,
            tol=tol,
        )

        n = int(result.n_windows)
        if n == 0:
            break

        for i in range(n):
            if len(collected) >= max_windows:
                break
            collected.append(
                AccessWindow(
                    t_rise=float(result.t_rise[i]),
                    t_set=float(result.t_set[i]),
                    t_max_el=float(result.t_max_el[i]),
                    az_rise=float(result.az_rise[i]),
                    el_rise=float(result.el_rise[i]),
                    rng_rise=float(result.rng_rise[i]),
                    az_set=float(result.az_set[i]),
                    el_set=float(result.el_set[i]),
                    rng_set=float(result.rng_set[i]),
                    az_max=float(result.az_max[i]),
                    el_max=float(result.el_max[i]),
                    rng_max=float(result.rng_max[i]),
                    duration=float(result.duration[i]),
                )
            )

        if n < batch_size:
            break

        # Advance past the last window by one grid step so the next call
        # does not re-detect a marginal crossing at the boundary.
        dt_grid = (t_end_f - current_start) / max(n_steps - 1, 1)
        current_start = float(result.t_set[n - 1]) + dt_grid

    return collected


def find_access_windows_from_ephemeris(
    positions_gcrf: ArrayLike,
    times: ArrayLike,
    location: GroundLocation,
    eop,
    epochs: list,
    min_elevation: float = 0.0,
    dt: float = 60.0,
    tol: float = 0.001,
    use_degrees: bool = False,
) -> list[AccessWindow]:
    """Find access windows from pre-computed GCRF ephemeris positions.

    Transforms GCRF positions to ITRF/ECEF, builds a linear-interpolation
    function for boundary refinement, and delegates to
    :func:`find_access_windows`.

    Args:
        positions_gcrf: Array of shape ``(N, 3)`` with GCRF positions in metres.
        times: Array of shape ``(N,)`` with times in seconds since the
            reference epoch.
        location: Ground station as a :class:`GroundLocation`.
        eop: :class:`~astrojax.eop.EOPData` for frame transformation.
        epochs: List of :class:`~astrojax.epoch.Epoch` instances
            corresponding to each time step (needed for GCRF→ITRF).
        min_elevation: Minimum elevation threshold (radians, or degrees
            if ``use_degrees=True``).
        dt: Coarse grid step size in seconds (default 60).
        tol: Bisection tolerance in seconds (default 0.001).
        use_degrees: If ``True``, interpret *min_elevation* as degrees.

    Returns:
        List of :class:`AccessWindow` instances.
    """
    from astrojax.frames import rotation_gcrf_to_itrf

    dtype = get_dtype()
    positions_gcrf = jnp.asarray(positions_gcrf, dtype=dtype)
    times_arr = jnp.asarray(times, dtype=dtype)

    # Transform all positions GCRF → ITRF
    positions_ecef = []
    for i, epc in enumerate(epochs):
        rot = rotation_gcrf_to_itrf(eop, epc)
        pos_ecef = rot @ positions_gcrf[i]
        positions_ecef.append(pos_ecef)
    positions_ecef = jnp.stack(positions_ecef)

    # Build interpolated ECEF position function
    def position_ecef_fn(t):
        t = jnp.asarray(t, dtype=dtype)
        # Linear interpolation: find bracketing indices
        idx_float = jnp.interp(t, times_arr, jnp.arange(len(times_arr), dtype=dtype))
        idx_lo = jnp.floor(idx_float).astype(jnp.int32)
        idx_lo = jnp.clip(idx_lo, 0, len(times_arr) - 2)
        idx_hi = idx_lo + 1

        t_lo = times_arr[idx_lo]
        t_hi = times_arr[idx_hi]
        frac = jnp.where(t_hi > t_lo, (t - t_lo) / (t_hi - t_lo), dtype(0.0))
        frac = jnp.clip(frac, dtype(0.0), dtype(1.0))

        return (dtype(1.0) - frac) * positions_ecef[idx_lo] + frac * positions_ecef[idx_hi]

    t_start = float(times_arr[0])
    t_end = float(times_arr[-1])

    return find_access_windows(
        position_ecef_fn,
        location,
        t_start,
        t_end,
        min_elevation=min_elevation,
        dt=dt,
        tol=tol,
        use_degrees=use_degrees,
    )
