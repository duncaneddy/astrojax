"""Tests for the astrojax.access module.

Covers GroundLocation construction, compute_elevation, compute_azel,
_refine_boundary, _detect_windows, find_access_windows (single and
multi-window), and find_access_windows_from_ephemeris.
"""

import jax
import jax.numpy as jnp
import numpy as np

from astrojax.access import (
    AccessResult,
    AccessWindow,
    GroundLocation,
    _detect_crossings_jit,
    _detect_windows,
    _find_max_elevation,
    _refine_boundary,
    compute_azel,
    compute_elevation,
    find_access_windows,
    find_access_windows_from_ephemeris,
    find_access_windows_jit,
    find_all_access_windows,
    ground_location,
)
from astrojax.config import get_dtype
from astrojax.constants import R_EARTH, WGS84_a

# ──────────────────────────────────────────────
# Tolerance constants (float32-appropriate)
# ──────────────────────────────────────────────

_ANGLE_TOL = 1e-3  # radians
_POS_TOL = 10.0  # metres
_TIME_TOL = 1.0  # seconds


# ──────────────────────────────────────────────
# GroundLocation construction
# ──────────────────────────────────────────────


class TestGroundLocation:
    def test_from_degrees(self):
        """Construct from degrees; verify lon/lat stored in radians."""
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0, use_degrees=True)
        assert isinstance(loc, GroundLocation)
        assert abs(loc.lon) < 1e-6
        assert abs(loc.lat) < 1e-6
        assert loc.ecef.shape == (3,)
        assert loc.rot_enz.shape == (3, 3)

    def test_from_radians(self):
        """Construct from radians."""
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0, use_degrees=False)
        assert abs(loc.lon) < 1e-6
        assert abs(loc.lat) < 1e-6

    def test_ecef_on_equator(self):
        """Station at lon=0, lat=0 should have ECEF near [WGS84_a, 0, 0]."""
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        assert jnp.abs(loc.ecef[0] - WGS84_a) < _POS_TOL
        assert jnp.abs(loc.ecef[1]) < _POS_TOL
        assert jnp.abs(loc.ecef[2]) < _POS_TOL

    def test_rot_enz_shape_and_orthogonal(self):
        """ENZ rotation matrix should be orthogonal."""
        loc = ground_location(lon=30.0, lat=45.0, alt=100.0)
        R = loc.rot_enz
        eye = R @ R.T
        assert jnp.allclose(eye, jnp.eye(3), atol=1e-5)


# ──────────────────────────────────────────────
# compute_elevation
# ──────────────────────────────────────────────


class TestComputeElevation:
    def test_directly_overhead(self):
        """Satellite directly above station → elevation ≈ 90°."""
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        sat_ecef = loc.ecef + jnp.array([0.0, 0.0, 0.0])
        # Move satellite radially outward (zenith direction)
        # At lon=0, lat=0, zenith is along +x
        sat_ecef = loc.ecef * (1.0 + 500e3 / jnp.linalg.norm(loc.ecef))

        el = compute_elevation(sat_ecef, loc.ecef, loc.rot_enz)
        assert jnp.abs(el - jnp.pi / 2) < _ANGLE_TOL

    def test_below_horizon(self):
        """Satellite on the opposite side of Earth → negative elevation."""
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        sat_ecef = -loc.ecef  # Antipodal point at surface level
        el = compute_elevation(sat_ecef, loc.ecef, loc.rot_enz)
        assert el < 0.0

    def test_on_horizon(self):
        """Satellite near the geometric horizon → elevation near 0°."""
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        # Station is at [WGS84_a, 0, 0].  Place satellite far along +y
        # at the same Earth radius + altitude → nearly on the horizon.
        sat_ecef = jnp.array([0.0, R_EARTH + 800e3, R_EARTH + 800e3])
        el = compute_elevation(sat_ecef, loc.ecef, loc.rot_enz)
        # Elevation should be between -45° and +45°
        assert jnp.abs(el) < jnp.pi / 4

    def test_without_precomputed_rot(self):
        """compute_elevation works without pre-computed rotation matrix."""
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        sat_ecef = loc.ecef * 1.05  # 5% further out = directly overhead
        el_with = compute_elevation(sat_ecef, loc.ecef, loc.rot_enz)
        el_without = compute_elevation(sat_ecef, loc.ecef)
        assert jnp.abs(el_with - el_without) < _ANGLE_TOL

    def test_jit_compatible(self):
        """compute_elevation works under jax.jit."""
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        sat_ecef = loc.ecef * 1.05

        @jax.jit
        def f(sat, sta, rot):
            return compute_elevation(sat, sta, rot)

        eager = compute_elevation(sat_ecef, loc.ecef, loc.rot_enz)
        jitted = f(sat_ecef, loc.ecef, loc.rot_enz)
        assert jnp.allclose(eager, jitted, atol=1e-6)

    def test_vmap_compatible(self):
        """compute_elevation works with jax.vmap over satellite positions."""
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        sats = jnp.stack([loc.ecef * (1.0 + (i + 1) * 0.01) for i in range(5)])

        el_batch = jax.vmap(lambda s: compute_elevation(s, loc.ecef, loc.rot_enz))(sats)

        assert el_batch.shape == (5,)
        # All directly overhead → all ~90°
        assert jnp.all(el_batch > jnp.pi / 4)


# ──────────────────────────────────────────────
# compute_azel
# ──────────────────────────────────────────────


class TestComputeAzel:
    def test_directly_overhead(self):
        """Satellite directly overhead → el ≈ 90°, range > 0."""
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        sat_ecef = loc.ecef * 1.05
        azel = compute_azel(sat_ecef, loc.ecef, loc.rot_enz)
        assert azel.shape == (3,)
        assert jnp.abs(azel[1] - jnp.pi / 2) < _ANGLE_TOL  # elevation
        assert azel[2] > 0  # range

    def test_known_north(self):
        """Satellite due north at similar altitude → az ≈ 0 (north)."""
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        # Place satellite slightly north (increase latitude)
        sat_geod = jnp.array([0.0, 0.01, 500e3])  # ~0.6° latitude north
        from astrojax.coordinates.geodetic import position_geodetic_to_ecef

        sat_ecef = position_geodetic_to_ecef(sat_geod)
        azel = compute_azel(sat_ecef, loc.ecef, loc.rot_enz)
        # Azimuth should be near 0 (north) or 2pi
        az = float(azel[0])
        assert az < 0.1 or az > 2 * jnp.pi - 0.1

    def test_jit_compatible(self):
        """compute_azel works under jax.jit."""
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        sat_ecef = loc.ecef * 1.05

        @jax.jit
        def f(sat, sta, rot):
            return compute_azel(sat, sta, rot)

        eager = compute_azel(sat_ecef, loc.ecef, loc.rot_enz)
        jitted = f(sat_ecef, loc.ecef, loc.rot_enz)
        assert jnp.allclose(eager, jitted, atol=1e-6)


# ──────────────────────────────────────────────
# _refine_boundary (bisection)
# ──────────────────────────────────────────────


class TestRefineBoundary:
    def test_sine_crossing(self):
        """Bisect a sine function crossing zero."""

        # sin(t) crosses zero at t = pi
        def el_fn(t):
            return jnp.sin(t)

        # Bracket: sin(3) > 0, sin(3.5) < 0 → crossing near pi ≈ 3.1416
        t_cross = float(_refine_boundary(el_fn, 3.0, 3.5, tol=1e-6))
        assert abs(t_cross - jnp.pi) < 1e-4

    def test_linear_crossing(self):
        """Bisect a linear function crossing zero at t=5."""

        def el_fn(t):
            return t - 5.0

        t_cross = float(_refine_boundary(el_fn, 0.0, 10.0, tol=1e-6))
        assert abs(t_cross - 5.0) < 1e-4

    def test_convergence_tolerance(self):
        """Verify the result respects the tolerance."""

        def el_fn(t):
            return t - 100.0

        t_cross_coarse = float(_refine_boundary(el_fn, 0.0, 200.0, tol=1.0))
        t_cross_fine = float(_refine_boundary(el_fn, 0.0, 200.0, tol=0.001))
        assert abs(t_cross_coarse - 100.0) < 1.0
        assert abs(t_cross_fine - 100.0) < 0.001


# ──────────────────────────────────────────────
# _find_max_elevation
# ──────────────────────────────────────────────


class TestFindMaxElevation:
    def test_quadratic_peak(self):
        """Find the peak of a downward quadratic."""

        def el_fn(t):
            return -((t - 50.0) ** 2) + 100.0

        t_max = float(_find_max_elevation(el_fn, 0.0, 100.0))
        assert abs(t_max - 50.0) < 1.0

    def test_sine_peak(self):
        """Find the peak of a sine function in [0, pi]."""

        def el_fn(t):
            return jnp.sin(t)

        t_max = float(_find_max_elevation(el_fn, 0.0, float(jnp.pi)))
        assert abs(t_max - jnp.pi / 2) < 0.5


# ──────────────────────────────────────────────
# _detect_windows
# ──────────────────────────────────────────────


class TestDetectWindows:
    def test_single_window(self):
        """One pass above threshold."""
        times = np.linspace(0, 100, 101)
        # Elevation: starts negative, crosses 0 around t≈6, peaks at t=50,
        # crosses 0 again around t≈94
        elevations = np.sin(np.pi * times / 100) * 0.5 - 0.1
        threshold = 0.0
        windows = _detect_windows(elevations, times, threshold)
        assert len(windows) == 1
        i_rise, i_set = windows[0]
        assert times[i_rise] < 15
        assert times[i_set] > 85

    def test_no_window(self):
        """Satellite never above threshold."""
        times = np.linspace(0, 100, 101)
        elevations = np.full_like(times, -0.5)  # Always below
        windows = _detect_windows(elevations, times, 0.0)
        assert len(windows) == 0

    def test_always_visible(self):
        """Satellite above threshold for entire span."""
        times = np.linspace(0, 100, 101)
        elevations = np.full_like(times, 0.5)  # Always above
        windows = _detect_windows(elevations, times, 0.0)
        assert len(windows) == 1
        assert windows[0] == (0, 100)

    def test_multiple_windows(self):
        """Two separate passes."""
        times = np.linspace(0, 200, 201)
        elevations = np.sin(2 * np.pi * times / 100) * 0.5
        windows = _detect_windows(elevations, times, 0.0)
        assert len(windows) == 2

    def test_visible_at_start(self):
        """Satellite already above at t_start."""
        times = np.linspace(0, 100, 101)
        elevations = np.cos(np.pi * times / 100) * 0.5  # Starts high, goes low
        windows = _detect_windows(elevations, times, 0.0)
        assert len(windows) == 1
        assert windows[0][0] == 0  # Starts at beginning

    def test_visible_at_end(self):
        """Satellite still above at t_end."""
        times = np.linspace(0, 100, 101)
        elevations = -np.cos(np.pi * times / 100) * 0.5  # Starts low, ends high
        windows = _detect_windows(elevations, times, 0.0)
        assert len(windows) == 1
        assert windows[0][1] == 100  # Ends at last index


# ──────────────────────────────────────────────
# find_access_windows end-to-end
# ──────────────────────────────────────────────


class TestFindAccessWindows:
    def _make_circular_orbit_fn(self, alt_km=500.0, period_s=5400.0):
        """Create a simple circular orbit position function in ECEF.

        The orbit is in the x-z plane (inclination 90°), with Earth rotating
        separately. For testing, we ignore Earth rotation and just move the
        satellite in a great circle.
        """
        dtype = get_dtype()
        r = dtype(R_EARTH + alt_km * 1e3)
        omega = dtype(2.0 * jnp.pi / period_s)

        def pos_fn(t):
            t = jnp.asarray(t, dtype=dtype)
            theta = omega * t
            x = r * jnp.cos(theta)
            y = dtype(0.0)
            z = r * jnp.sin(theta)
            return jnp.array([x, y, z])

        return pos_fn

    def test_single_pass(self):
        """A polar orbit should produce access windows over a station."""
        pos_fn = self._make_circular_orbit_fn()
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

        # One orbit period
        windows = find_access_windows(pos_fn, loc, t_start=0.0, t_end=5400.0, dt=30.0)

        # Should find at least one window
        assert len(windows) >= 1
        w = windows[0]
        assert isinstance(w, AccessWindow)
        assert w.duration > 0
        assert w.t_rise < w.t_set
        assert w.t_max_el >= w.t_rise
        assert w.t_max_el <= w.t_set
        assert w.el_max > 0
        assert w.el_rise >= -_ANGLE_TOL
        assert w.el_set >= -_ANGLE_TOL

    def test_no_access(self):
        """Satellite never visible (orbit plane doesn't cross station)."""
        dtype = get_dtype()

        # Place station at north pole, orbit in equatorial plane
        loc = ground_location(lon=0.0, lat=89.9, alt=0.0)

        r = dtype(R_EARTH + 500e3)
        omega = dtype(2.0 * jnp.pi / 5400.0)

        def pos_fn(t):
            t = jnp.asarray(t, dtype=dtype)
            theta = omega * t
            return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta), dtype(0.0)])

        windows = find_access_windows(pos_fn, loc, t_start=0.0, t_end=5400.0, dt=30.0)
        assert len(windows) == 0

    def test_min_elevation_filter(self):
        """Higher min_elevation should yield fewer/shorter windows."""
        pos_fn = self._make_circular_orbit_fn()
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

        windows_0 = find_access_windows(pos_fn, loc, 0.0, 5400.0, min_elevation=0.0, dt=30.0)
        windows_10 = find_access_windows(
            pos_fn, loc, 0.0, 5400.0, min_elevation=10.0, use_degrees=True, dt=30.0
        )

        # Higher threshold → shorter or fewer windows
        if len(windows_0) > 0 and len(windows_10) > 0:
            assert windows_10[0].duration <= windows_0[0].duration + _TIME_TOL

    def test_window_properties(self):
        """Verify AccessWindow fields are self-consistent."""
        pos_fn = self._make_circular_orbit_fn()
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

        windows = find_access_windows(pos_fn, loc, 0.0, 5400.0, dt=30.0)
        if len(windows) > 0:
            w = windows[0]
            # Duration matches rise/set times
            assert abs(w.duration - (w.t_set - w.t_rise)) < 0.01
            # Ranges should be positive
            assert w.rng_rise > 0
            assert w.rng_set > 0
            assert w.rng_max > 0
            # Max elevation should be >= rise/set elevation
            assert w.el_max >= w.el_rise - _ANGLE_TOL
            assert w.el_max >= w.el_set - _ANGLE_TOL

    def test_use_degrees_min_elevation(self):
        """use_degrees flag correctly converts min_elevation."""
        pos_fn = self._make_circular_orbit_fn()
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

        w_rad = find_access_windows(
            pos_fn, loc, 0.0, 5400.0, min_elevation=jnp.deg2rad(5.0), dt=30.0
        )
        w_deg = find_access_windows(
            pos_fn, loc, 0.0, 5400.0, min_elevation=5.0, use_degrees=True, dt=30.0
        )

        assert len(w_rad) == len(w_deg)
        if len(w_rad) > 0:
            assert abs(w_rad[0].t_rise - w_deg[0].t_rise) < _TIME_TOL


# ──────────────────────────────────────────────
# find_access_windows_from_ephemeris
# ──────────────────────────────────────────────


class TestFindAccessWindowsFromEphemeris:
    def test_basic_ephemeris(self):
        """Smoke test with synthetic GCRF ephemeris and zero EOP."""
        from astrojax.eop import zero_eop
        from astrojax.epoch import Epoch

        dtype = get_dtype()
        eop = zero_eop()
        epoch0 = Epoch(2024, 1, 1, 12, 0, 0.0)

        # Simple polar orbit in GCRF (x-z plane)
        n_steps = 100
        period = 5400.0
        times = jnp.linspace(dtype(0.0), dtype(period), n_steps)
        r = dtype(R_EARTH + 500e3)

        positions_gcrf = jnp.stack(
            [
                jnp.array(
                    [
                        r * jnp.cos(dtype(2.0 * jnp.pi) * t / dtype(period)),
                        dtype(0.0),
                        r * jnp.sin(dtype(2.0 * jnp.pi) * t / dtype(period)),
                    ]
                )
                for t in times
            ]
        )

        epochs = [epoch0 + float(t) for t in times]
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

        windows = find_access_windows_from_ephemeris(
            positions_gcrf, times, loc, eop, epochs, dt=60.0
        )

        # Should find at least one window for this geometry
        assert isinstance(windows, list)
        # Each element should be an AccessWindow
        for w in windows:
            assert isinstance(w, AccessWindow)
            assert w.duration > 0


# ──────────────────────────────────────────────
# Import smoke test
# ──────────────────────────────────────────────


class TestDetectCrossingsJit:
    """Mirror TestDetectWindows cases using the JIT-compatible implementation."""

    def test_single_window(self):
        """One pass above threshold matches _detect_windows."""
        times = np.linspace(0, 100, 101)
        elevations = np.sin(np.pi * times / 100) * 0.5 - 0.1
        threshold = 0.0

        # Reference
        ref_windows = _detect_windows(elevations, times, threshold)
        assert len(ref_windows) == 1

        # JIT version
        el_jax = jnp.array(elevations)
        rise_idx, set_idx, valid, n_win = _detect_crossings_jit(el_jax, threshold, 5)
        assert int(n_win) == 1
        assert bool(valid[0])
        # Indices should match reference
        assert int(rise_idx[0]) == ref_windows[0][0]
        assert int(set_idx[0]) == ref_windows[0][1]

    def test_no_window(self):
        """Satellite never above threshold."""
        elevations = jnp.full(101, -0.5)
        rise_idx, set_idx, valid, n_win = _detect_crossings_jit(elevations, 0.0, 5)
        assert int(n_win) == 0
        assert not bool(valid[0])

    def test_always_visible(self):
        """Satellite above threshold for entire span."""
        elevations = jnp.full(101, 0.5)
        rise_idx, set_idx, valid, n_win = _detect_crossings_jit(elevations, 0.0, 5)
        assert int(n_win) == 1
        assert int(rise_idx[0]) == 0
        assert int(set_idx[0]) == 100  # n_steps - 1

    def test_multiple_windows(self):
        """Two separate passes."""
        times = np.linspace(0, 200, 201)
        elevations = np.sin(2 * np.pi * times / 100) * 0.5

        ref_windows = _detect_windows(elevations, times, 0.0)
        el_jax = jnp.array(elevations)
        rise_idx, set_idx, valid, n_win = _detect_crossings_jit(el_jax, 0.0, 5)

        assert int(n_win) == len(ref_windows)
        for i, (r_ref, s_ref) in enumerate(ref_windows):
            assert int(rise_idx[i]) == r_ref
            assert int(set_idx[i]) == s_ref

    def test_visible_at_start(self):
        """Satellite already above at t_start."""
        times = np.linspace(0, 100, 101)
        elevations = np.cos(np.pi * times / 100) * 0.5
        el_jax = jnp.array(elevations)
        rise_idx, set_idx, valid, n_win = _detect_crossings_jit(el_jax, 0.0, 5)
        assert int(n_win) == 1
        assert int(rise_idx[0]) == 0

    def test_visible_at_end(self):
        """Satellite still above at t_end."""
        times = np.linspace(0, 100, 101)
        elevations = -np.cos(np.pi * times / 100) * 0.5
        el_jax = jnp.array(elevations)
        rise_idx, set_idx, valid, n_win = _detect_crossings_jit(el_jax, 0.0, 5)
        assert int(n_win) == 1
        assert int(set_idx[0]) == 100

    def test_max_windows_capped(self):
        """More windows than max_windows are silently capped."""
        times = np.linspace(0, 400, 401)
        elevations = np.sin(2 * np.pi * times / 100) * 0.5  # 4 windows
        el_jax = jnp.array(elevations)
        rise_idx, set_idx, valid, n_win = _detect_crossings_jit(el_jax, 0.0, 2)
        # Only 2 slots available
        assert int(n_win) <= 2
        assert valid.shape == (2,)


class TestFindAccessWindowsJit:
    """Mirror TestFindAccessWindows cases against the JIT version."""

    def _make_circular_orbit_fn(self, alt_km=500.0, period_s=5400.0):
        dtype = get_dtype()
        r = dtype(R_EARTH + alt_km * 1e3)
        omega = dtype(2.0 * jnp.pi / period_s)

        def pos_fn(t):
            t = jnp.asarray(t, dtype=dtype)
            theta = omega * t
            x = r * jnp.cos(theta)
            y = dtype(0.0)
            z = r * jnp.sin(theta)
            return jnp.array([x, y, z])

        return pos_fn

    def test_single_pass(self):
        """JIT version finds windows matching the hybrid version."""
        pos_fn = self._make_circular_orbit_fn()
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

        # Hybrid reference
        ref = find_access_windows(pos_fn, loc, 0.0, 5400.0, dt=30.0)
        assert len(ref) >= 1

        # JIT version
        n_steps = int(jnp.ceil(5400.0 / 30.0)) + 1
        result = find_access_windows_jit(
            pos_fn,
            loc.ecef,
            loc.rot_enz,
            0.0,
            5400.0,
            0.0,
            max_windows=10,
            n_steps=n_steps,
        )
        assert isinstance(result, AccessResult)
        n = int(result.n_windows)
        assert n == len(ref)

        for i in range(n):
            assert abs(float(result.t_rise[i]) - ref[i].t_rise) < _TIME_TOL
            assert abs(float(result.t_set[i]) - ref[i].t_set) < _TIME_TOL
            assert abs(float(result.duration[i]) - ref[i].duration) < _TIME_TOL
            assert abs(float(result.el_max[i]) - ref[i].el_max) < _ANGLE_TOL

    def test_no_access(self):
        """No windows returned when satellite not visible."""
        dtype = get_dtype()
        loc = ground_location(lon=0.0, lat=89.9, alt=0.0)

        r = dtype(R_EARTH + 500e3)
        omega = dtype(2.0 * jnp.pi / 5400.0)

        def pos_fn(t):
            t = jnp.asarray(t, dtype=dtype)
            theta = omega * t
            return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta), dtype(0.0)])

        result = find_access_windows_jit(
            pos_fn,
            loc.ecef,
            loc.rot_enz,
            0.0,
            5400.0,
            0.0,
            max_windows=10,
            n_steps=181,
        )
        assert int(result.n_windows) == 0
        assert not bool(result.valid[0])

    def test_valid_mask(self):
        """valid mask correctly marks used and unused slots."""
        pos_fn = self._make_circular_orbit_fn()
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

        result = find_access_windows_jit(
            pos_fn,
            loc.ecef,
            loc.rot_enz,
            0.0,
            5400.0,
            0.0,
            max_windows=10,
            n_steps=181,
        )
        n = int(result.n_windows)
        for i in range(10):
            if i < n:
                assert bool(result.valid[i])
            else:
                assert not bool(result.valid[i])

    def test_window_properties(self):
        """AccessResult fields are self-consistent for valid windows."""
        pos_fn = self._make_circular_orbit_fn()
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

        result = find_access_windows_jit(
            pos_fn,
            loc.ecef,
            loc.rot_enz,
            0.0,
            5400.0,
            0.0,
            max_windows=10,
            n_steps=181,
        )
        n = int(result.n_windows)
        for i in range(n):
            t_r = float(result.t_rise[i])
            t_s = float(result.t_set[i])
            t_m = float(result.t_max_el[i])
            assert t_r < t_s
            assert t_m >= t_r - 0.01
            assert t_m <= t_s + 0.01
            assert float(result.duration[i]) > 0
            assert float(result.rng_rise[i]) > 0
            assert float(result.rng_set[i]) > 0
            assert float(result.rng_max[i]) > 0
            assert float(result.el_max[i]) >= float(result.el_rise[i]) - _ANGLE_TOL


class TestFindAccessWindowsJitCompilability:
    """Verify find_access_windows_jit works under jax.jit and jax.vmap."""

    def _make_circular_orbit_fn(self):
        dtype = get_dtype()
        r = dtype(R_EARTH + 500e3)
        omega = dtype(2.0 * jnp.pi / 5400.0)

        def pos_fn(t):
            t = jnp.asarray(t, dtype=dtype)
            theta = omega * t
            return jnp.array([r * jnp.cos(theta), dtype(0.0), r * jnp.sin(theta)])

        return pos_fn

    def test_jit_smoke(self):
        """jax.jit(find_access_windows_jit) produces correct results."""
        pos_fn = self._make_circular_orbit_fn()
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

        jitted = jax.jit(
            find_access_windows_jit,
            static_argnums=(0, 6, 7),
        )
        result = jitted(
            pos_fn,
            loc.ecef,
            loc.rot_enz,
            0.0,
            5400.0,
            0.0,
            10,
            181,
        )
        assert isinstance(result, AccessResult)
        assert int(result.n_windows) >= 1
        assert bool(result.valid[0])

    def test_vmap_over_stations(self):
        """vmap over multiple stations works."""
        pos_fn = self._make_circular_orbit_fn()

        # Two stations at different longitudes
        loc0 = ground_location(lon=0.0, lat=0.0, alt=0.0)
        loc1 = ground_location(lon=90.0, lat=0.0, alt=0.0)

        stations = jnp.stack([loc0.ecef, loc1.ecef])
        rots = jnp.stack([loc0.rot_enz, loc1.rot_enz])

        def access_for_station(station, rot):
            return find_access_windows_jit(
                pos_fn,
                station,
                rot,
                0.0,
                5400.0,
                0.0,
                max_windows=10,
                n_steps=181,
            )

        results = jax.vmap(access_for_station)(stations, rots)
        # Should return batched AccessResult with shape (2, max_windows)
        assert results.t_rise.shape == (2, 10)
        assert results.valid.shape == (2, 10)
        assert results.n_windows.shape == (2,)


class TestFindAllAccessWindows:
    """Tests for the paging convenience wrapper."""

    def _make_circular_orbit_fn(self, alt_km=500.0, period_s=5400.0):
        dtype = get_dtype()
        r = dtype(R_EARTH + alt_km * 1e3)
        omega = dtype(2.0 * jnp.pi / period_s)

        def pos_fn(t):
            t = jnp.asarray(t, dtype=dtype)
            theta = omega * t
            x = r * jnp.cos(theta)
            y = dtype(0.0)
            z = r * jnp.sin(theta)
            return jnp.array([x, y, z])

        return pos_fn

    def test_matches_hybrid(self):
        """Results match find_access_windows for a single-orbit case."""
        pos_fn = self._make_circular_orbit_fn()
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

        ref = find_access_windows(pos_fn, loc, 0.0, 5400.0, dt=30.0)
        result = find_all_access_windows(
            pos_fn,
            loc.ecef,
            loc.rot_enz,
            0.0,
            5400.0,
            0.0,
            n_steps=181,
            batch_size=10,
        )

        assert len(result) == len(ref)
        for r, w in zip(result, ref, strict=False):
            assert isinstance(r, AccessWindow)
            assert abs(r.t_rise - w.t_rise) < _TIME_TOL
            assert abs(r.t_set - w.t_set) < _TIME_TOL

    def test_paging_with_small_batch(self):
        """A batch_size=1 still finds all windows via paging."""
        pos_fn = self._make_circular_orbit_fn()
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

        ref = find_access_windows(pos_fn, loc, 0.0, 5400.0, dt=30.0)

        result = find_all_access_windows(
            pos_fn,
            loc.ecef,
            loc.rot_enz,
            0.0,
            5400.0,
            0.0,
            n_steps=181,
            batch_size=1,
        )

        assert len(result) == len(ref)
        for r, w in zip(result, ref, strict=False):
            assert abs(r.t_rise - w.t_rise) < _TIME_TOL
            assert abs(r.t_set - w.t_set) < _TIME_TOL

    def test_max_windows_cap(self):
        """Never returns more than max_windows."""
        pos_fn = self._make_circular_orbit_fn()
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

        result = find_all_access_windows(
            pos_fn,
            loc.ecef,
            loc.rot_enz,
            0.0,
            5400.0,
            0.0,
            max_windows=1,
            n_steps=181,
            batch_size=10,
        )
        assert len(result) <= 1

    def test_no_access(self):
        """Returns empty list when satellite is never visible."""
        dtype = get_dtype()
        loc = ground_location(lon=0.0, lat=89.9, alt=0.0)
        r = dtype(R_EARTH + 500e3)
        omega = dtype(2.0 * jnp.pi / 5400.0)

        def pos_fn(t):
            t = jnp.asarray(t, dtype=dtype)
            theta = omega * t
            return jnp.array([r * jnp.cos(theta), r * jnp.sin(theta), dtype(0.0)])

        result = find_all_access_windows(
            pos_fn,
            loc.ecef,
            loc.rot_enz,
            0.0,
            5400.0,
            0.0,
            n_steps=181,
            batch_size=10,
        )
        assert result == []


class TestImports:
    def test_top_level_imports(self):
        """All access symbols importable from astrojax top level."""
        import astrojax

        assert hasattr(astrojax, "GroundLocation")
        assert hasattr(astrojax, "AccessWindow")
        assert hasattr(astrojax, "AccessResult")
        assert hasattr(astrojax, "ground_location")
        assert hasattr(astrojax, "compute_elevation")
        assert hasattr(astrojax, "compute_azel")
        assert hasattr(astrojax, "find_access_windows")
        assert hasattr(astrojax, "find_access_windows_jit")
        assert hasattr(astrojax, "find_all_access_windows")
        assert hasattr(astrojax, "find_access_windows_from_ephemeris")
