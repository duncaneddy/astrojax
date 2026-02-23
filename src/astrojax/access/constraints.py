"""Constraint functions for satellite access prediction.

A constraint function has the signature::

    constraint_fn(sat_ecef: Array[3], station_ecef: Array[3], rot_enz: Array[3,3]) -> float

- **Positive** means the constraint is satisfied.
- **Negative** means the constraint is violated.
- **Zero** is the boundary (where bisection finds roots).

Factory functions capture parameters inside closures, making them
JIT-compatible and composable.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from astrojax.coordinates.topocentric import position_enz_to_azel

# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------


def compute_off_nadir(sat_ecef: Array, station_ecef: Array) -> Array:
    """Off-nadir angle: angle between satellite nadir and line-of-sight to station.

    Args:
        sat_ecef: Satellite ECEF position ``[x, y, z]`` in metres.
        station_ecef: Station ECEF position ``[x, y, z]`` in metres.

    Returns:
        Off-nadir angle in radians.
    """
    nadir = -sat_ecef / jnp.linalg.norm(sat_ecef)
    los = station_ecef - sat_ecef
    los = los / jnp.linalg.norm(los)
    return jnp.arccos(jnp.clip(jnp.dot(nadir, los), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Constraint factory functions
# ---------------------------------------------------------------------------


def elevation_constraint(
    min_el: float, max_el: float | None = None
) -> Callable[[Array, Array, Array], Array]:
    """Create a constraint that requires elevation within a band.

    Args:
        min_el: Minimum elevation angle in radians.
        max_el: Optional maximum elevation angle in radians.
            If ``None``, only the lower bound is enforced.

    Returns:
        A constraint function matching the constraint protocol.
    """

    def constraint(sat_ecef: Array, station_ecef: Array, rot_enz: Array) -> Array:
        r_enz = rot_enz @ (sat_ecef - station_ecef)
        azel = position_enz_to_azel(r_enz)
        el = azel[1]
        if max_el is None:
            return el - min_el
        return jnp.minimum(el - min_el, max_el - el)

    return constraint


def elevation_mask_constraint(
    mask_az: Array, mask_el: Array
) -> Callable[[Array, Array, Array], Array]:
    """Create a constraint with azimuth-dependent elevation threshold.

    The mask defines minimum elevation as a function of azimuth.
    Azimuth values wrap around at ``2*pi``.

    Args:
        mask_az: 1-D array of azimuth breakpoints in radians,
            sorted ascending in ``[0, 2*pi)``.
        mask_el: 1-D array of corresponding minimum elevation
            thresholds in radians.

    Returns:
        A constraint function matching the constraint protocol.
    """
    mask_az = jnp.asarray(mask_az)
    mask_el = jnp.asarray(mask_el)

    def constraint(sat_ecef: Array, station_ecef: Array, rot_enz: Array) -> Array:
        r_enz = rot_enz @ (sat_ecef - station_ecef)
        azel = position_enz_to_azel(r_enz)
        az, el = azel[0], azel[1]
        threshold = jnp.interp(az, mask_az, mask_el, period=2.0 * jnp.pi)
        return el - threshold

    return constraint


def off_nadir_constraint(
    max_off_nadir: float, min_off_nadir: float = 0.0
) -> Callable[[Array, Array, Array], Array]:
    """Create a constraint on the satellite off-nadir angle.

    Args:
        max_off_nadir: Maximum off-nadir angle in radians.
        min_off_nadir: Minimum off-nadir angle in radians (default 0).

    Returns:
        A constraint function matching the constraint protocol.
    """

    def constraint(sat_ecef: Array, station_ecef: Array, rot_enz: Array) -> Array:
        angle = compute_off_nadir(sat_ecef, station_ecef)
        return jnp.minimum(angle - min_off_nadir, max_off_nadir - angle)

    return constraint


# ---------------------------------------------------------------------------
# Composition operators
# ---------------------------------------------------------------------------


def constraint_all(*constraints: Callable) -> Callable[[Array, Array, Array], Array]:
    """Compose constraints with AND logic.

    The combined value is ``jnp.min(values)`` — satisfied only when
    *all* sub-constraints are satisfied.

    Args:
        *constraints: Two or more constraint functions.

    Returns:
        A composed constraint function.
    """

    def combined(sat_ecef: Array, station_ecef: Array, rot_enz: Array) -> Array:
        values = jnp.array([c(sat_ecef, station_ecef, rot_enz) for c in constraints])
        return jnp.min(values)

    return combined


def constraint_any(*constraints: Callable) -> Callable[[Array, Array, Array], Array]:
    """Compose constraints with OR logic.

    The combined value is ``jnp.max(values)`` — satisfied when
    *any* sub-constraint is satisfied.

    Args:
        *constraints: Two or more constraint functions.

    Returns:
        A composed constraint function.
    """

    def combined(sat_ecef: Array, station_ecef: Array, rot_enz: Array) -> Array:
        values = jnp.array([c(sat_ecef, station_ecef, rot_enz) for c in constraints])
        return jnp.max(values)

    return combined


def constraint_not(constraint: Callable) -> Callable[[Array, Array, Array], Array]:
    """Negate a constraint.

    Args:
        constraint: A constraint function.

    Returns:
        A constraint function whose sign is flipped.
    """

    def negated(sat_ecef: Array, station_ecef: Array, rot_enz: Array) -> Array:
        return -constraint(sat_ecef, station_ecef, rot_enz)

    return negated
