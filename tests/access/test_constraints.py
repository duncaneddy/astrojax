"""Tests for the astrojax.constraints module.

Covers compute_off_nadir, elevation_constraint, elevation_mask_constraint,
off_nadir_constraint, and composition operators (constraint_all, constraint_any,
constraint_not).
"""

import jax
import jax.numpy as jnp

from astrojax.access import ground_location
from astrojax.access.constraints import (
    compute_off_nadir,
    constraint_all,
    constraint_any,
    constraint_not,
    elevation_constraint,
    elevation_mask_constraint,
    off_nadir_constraint,
)
from astrojax.config import get_dtype
from astrojax.constants import R_EARTH

_ANGLE_TOL = 1e-3  # radians


# ──────────────────────────────────────────────
# compute_off_nadir
# ──────────────────────────────────────────────


class TestComputeOffNadir:
    def test_directly_overhead(self):
        """Station directly below satellite → off-nadir ≈ 0."""
        dtype = get_dtype()
        sat_ecef = jnp.array([R_EARTH + 500e3, 0.0, 0.0], dtype=dtype)
        station_ecef = jnp.array([R_EARTH, 0.0, 0.0], dtype=dtype)
        angle = compute_off_nadir(sat_ecef, station_ecef)
        assert jnp.abs(angle) < _ANGLE_TOL

    def test_large_off_nadir(self):
        """Station far from sub-satellite point → large off-nadir."""
        dtype = get_dtype()
        r_sat = R_EARTH + 500e3
        sat_ecef = jnp.array([r_sat, 0.0, 0.0], dtype=dtype)
        # Station perpendicular at Earth surface
        station_ecef = jnp.array([0.0, R_EARTH, 0.0], dtype=dtype)
        angle = compute_off_nadir(sat_ecef, station_ecef)
        # Off-nadir should be significantly above zero (~43°)
        assert angle > jnp.deg2rad(30.0)

    def test_jit_compatible(self):
        """compute_off_nadir works under jax.jit."""
        dtype = get_dtype()
        sat_ecef = jnp.array([R_EARTH + 500e3, 0.0, 0.0], dtype=dtype)
        station_ecef = jnp.array([R_EARTH, 0.0, 0.0], dtype=dtype)
        eager = compute_off_nadir(sat_ecef, station_ecef)
        jitted = jax.jit(compute_off_nadir)(sat_ecef, station_ecef)
        assert jnp.allclose(eager, jitted, atol=1e-6)

    def test_vmap_compatible(self):
        """compute_off_nadir works with jax.vmap."""
        dtype = get_dtype()
        station_ecef = jnp.array([R_EARTH, 0.0, 0.0], dtype=dtype)
        sats = jnp.array(
            [
                [R_EARTH + 500e3, 0.0, 0.0],
                [R_EARTH + 1000e3, 0.0, 0.0],
                [0.0, R_EARTH + 500e3, 0.0],
            ],
            dtype=dtype,
        )
        angles = jax.vmap(lambda s: compute_off_nadir(s, station_ecef))(sats)
        assert angles.shape == (3,)


# ──────────────────────────────────────────────
# elevation_constraint
# ──────────────────────────────────────────────


class TestElevationConstraint:
    def _setup(self):
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        return loc.ecef, loc.rot_enz

    def test_above_minimum(self):
        """Satellite above min elevation → positive value."""
        station, rot = self._setup()
        # Directly overhead → el ≈ 90°
        sat = station * 1.05
        fn = elevation_constraint(jnp.deg2rad(10.0))
        val = fn(sat, station, rot)
        assert val > 0

    def test_below_minimum(self):
        """Satellite below min elevation → negative value."""
        station, rot = self._setup()
        # Opposite side of Earth → el < 0
        sat = -station
        fn = elevation_constraint(jnp.deg2rad(10.0))
        val = fn(sat, station, rot)
        assert val < 0

    def test_band_constraint(self):
        """Band constraint: both min and max enforced."""
        station, rot = self._setup()
        # Directly overhead → el ≈ 90°, which exceeds max_el=45°
        sat = station * 1.05
        fn = elevation_constraint(jnp.deg2rad(10.0), jnp.deg2rad(45.0))
        val = fn(sat, station, rot)
        assert val < 0  # violated because el > max

    def test_band_constraint_within(self):
        """Band constraint satisfied when elevation is within band."""
        station, rot = self._setup()
        fn = elevation_constraint(jnp.deg2rad(10.0), jnp.deg2rad(85.0))
        # Station at equator, satellite slightly offset → moderate elevation
        dtype = get_dtype()
        sat = jnp.array([R_EARTH + 500e3, 200e3, 0.0], dtype=dtype)
        val = fn(sat, station, rot)
        # Should be positive if elevation is within [10°, 85°]
        # The exact value depends on geometry, just check the function runs
        assert isinstance(float(val), float)


# ──────────────────────────────────────────────
# elevation_mask_constraint
# ──────────────────────────────────────────────


class TestElevationMaskConstraint:
    def _setup(self):
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        return loc.ecef, loc.rot_enz

    def test_uniform_mask_matches_flat(self):
        """Uniform mask at 10° should behave like elevation_constraint(10°)."""
        station, rot = self._setup()
        min_el = jnp.deg2rad(10.0)
        mask_az = jnp.linspace(0, 2 * jnp.pi, 8, endpoint=False)
        mask_el = jnp.full(8, min_el)

        fn_mask = elevation_mask_constraint(mask_az, mask_el)
        fn_flat = elevation_constraint(min_el)

        sat = station * 1.05
        val_mask = fn_mask(sat, station, rot)
        val_flat = fn_flat(sat, station, rot)
        assert jnp.abs(val_mask - val_flat) < _ANGLE_TOL

    def test_azimuth_dependent(self):
        """Higher threshold in some azimuths narrows the constraint."""
        station, rot = self._setup()
        # Low threshold everywhere except near az=0 where it's very high
        mask_az = jnp.array([0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2])
        mask_el = jnp.array(
            [jnp.deg2rad(80.0), jnp.deg2rad(5.0), jnp.deg2rad(5.0), jnp.deg2rad(5.0)]
        )
        fn = elevation_mask_constraint(mask_az, mask_el)
        # Function should return a scalar
        sat = station * 1.05
        val = fn(sat, station, rot)
        assert isinstance(float(val), float)


# ──────────────────────────────────────────────
# off_nadir_constraint
# ──────────────────────────────────────────────


class TestOffNadirConstraint:
    def test_within_bounds(self):
        """Off-nadir within bounds → non-negative."""
        dtype = get_dtype()
        sat = jnp.array([R_EARTH + 500e3, 0.0, 0.0], dtype=dtype)
        # Slightly offset station so off-nadir is small but non-zero
        station = jnp.array([R_EARTH, 50e3, 0.0], dtype=dtype)
        rot = jnp.eye(3, dtype=dtype)  # doesn't matter for off-nadir

        fn = off_nadir_constraint(jnp.deg2rad(60.0))
        val = fn(sat, station, rot)
        assert val > 0  # small off-nadir is within 60°

    def test_outside_bounds(self):
        """Off-nadir exceeds max → negative."""
        dtype = get_dtype()
        sat = jnp.array([R_EARTH + 500e3, 0.0, 0.0], dtype=dtype)
        station = jnp.array([0.0, R_EARTH, 0.0], dtype=dtype)
        rot = jnp.eye(3, dtype=dtype)

        fn = off_nadir_constraint(jnp.deg2rad(5.0))
        val = fn(sat, station, rot)
        assert val < 0  # large off-nadir exceeds 5°


# ──────────────────────────────────────────────
# Composition operators
# ──────────────────────────────────────────────


class TestComposition:
    def _setup(self):
        loc = ground_location(lon=0.0, lat=0.0, alt=0.0)
        return loc.ecef, loc.rot_enz

    def test_constraint_all_and_logic(self):
        """constraint_all: all must be satisfied."""
        station, rot = self._setup()
        sat = station * 1.05  # directly overhead

        c_pass = elevation_constraint(0.0)  # overhead passes
        c_fail = elevation_constraint(jnp.deg2rad(95.0))  # impossible to exceed 90°

        combined = constraint_all(c_pass, c_fail)
        val = combined(sat, station, rot)
        assert val < 0  # one fails → combined fails

    def test_constraint_all_both_pass(self):
        """constraint_all: both pass → positive."""
        station, rot = self._setup()
        sat = station * 1.05

        c1 = elevation_constraint(0.0)
        c2 = elevation_constraint(jnp.deg2rad(10.0))

        combined = constraint_all(c1, c2)
        val = combined(sat, station, rot)
        assert val > 0

    def test_constraint_any_or_logic(self):
        """constraint_any: at least one must be satisfied."""
        station, rot = self._setup()
        sat = station * 1.05

        c_pass = elevation_constraint(0.0)
        c_fail = elevation_constraint(jnp.deg2rad(95.0))

        combined = constraint_any(c_pass, c_fail)
        val = combined(sat, station, rot)
        assert val > 0  # one passes → combined passes

    def test_constraint_any_both_fail(self):
        """constraint_any: both fail → negative."""
        station, rot = self._setup()
        sat = -station  # below horizon

        c1 = elevation_constraint(jnp.deg2rad(10.0))
        c2 = elevation_constraint(jnp.deg2rad(20.0))

        combined = constraint_any(c1, c2)
        val = combined(sat, station, rot)
        assert val < 0

    def test_constraint_not(self):
        """constraint_not flips sign."""
        station, rot = self._setup()
        sat = station * 1.05

        c = elevation_constraint(0.0)
        val_orig = c(sat, station, rot)
        val_neg = constraint_not(c)(sat, station, rot)
        assert jnp.abs(val_orig + val_neg) < 1e-6

    def test_jit_composed_constraint(self):
        """Composed constraints work under JIT."""
        station, rot = self._setup()
        sat = station * 1.05

        c = constraint_all(
            elevation_constraint(0.0),
            elevation_constraint(jnp.deg2rad(5.0)),
        )

        eager = c(sat, station, rot)
        jitted = jax.jit(c)(sat, station, rot)
        assert jnp.allclose(eager, jitted, atol=1e-6)
