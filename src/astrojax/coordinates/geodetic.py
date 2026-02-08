"""Geodetic (WGS84 ellipsoid) coordinate transformations.

Converts between geodetic coordinates ``[longitude, latitude, altitude]``
and Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates ``[x, y, z]``.

The geodetic model uses the WGS84 reference ellipsoid, which accounts for
Earth's oblateness.  The forward transformation is closed-form; the inverse
uses Bowring's iterative method implemented with ``jax.lax.while_loop``
for JAX traceability.

All inputs and outputs use SI base units (metres, radians).

References:
    1. NIMA Technical Report TR8350.2, *Department of Defense World Geodetic
       System 1984*.
    2. O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods
       and Applications*, Springer, 2012, Sec. 5.3.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.constants import WGS84_a, WGS84_f

# First eccentricity squared of the WGS84 ellipsoid
ECC2 = WGS84_f * (2.0 - WGS84_f)


def position_geodetic_to_ecef(
    x_geod: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Convert geodetic position to ECEF Cartesian coordinates.

    Uses the WGS84 prime vertical radius of curvature:

    .. math::

        N = \\frac{a}{\\sqrt{1 - e^2 \\sin^2 \\phi}}

    Args:
        x_geod: Geodetic coordinates ``[lon, lat, alt]``.
            Longitude and latitude in *rad* (or *deg* if ``use_degrees=True``),
            altitude in *m* above the WGS84 ellipsoid.
        use_degrees: If ``True``, interpret longitude and latitude as degrees.

    Returns:
        jax.Array: ECEF position ``[x, y, z]`` in *m*.

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax.coordinates import position_geodetic_to_ecef
        >>> x_geod = jnp.array([0.0, 0.0, 0.0])
        >>> x_ecef = position_geodetic_to_ecef(x_geod)
        >>> float(x_ecef[0])  # WGS84_a on the equator
        6378137.0
    """
    x_geod = jnp.asarray(x_geod, dtype=jnp.float32)

    lon = x_geod[0]
    lat = x_geod[1]
    alt = x_geod[2]

    if use_degrees:
        lon = jnp.deg2rad(lon)
        lat = jnp.deg2rad(lat)

    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)

    N = WGS84_a / jnp.sqrt(1.0 - ECC2 * sin_lat * sin_lat)

    x = (N + alt) * cos_lat * jnp.cos(lon)
    y = (N + alt) * cos_lat * jnp.sin(lon)
    z = ((1.0 - ECC2) * N + alt) * sin_lat

    return jnp.array([x, y, z])


def position_ecef_to_geodetic(
    x_ecef: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Convert ECEF Cartesian coordinates to geodetic position.

    Uses Bowring's iterative method with convergence controlled by
    ``jax.lax.while_loop`` (max 10 iterations).  The convergence
    threshold is scaled for float32 precision.

    Args:
        x_ecef: ECEF position ``[x, y, z]`` in *m*.
        use_degrees: If ``True``, return longitude and latitude in degrees.

    Returns:
        jax.Array: Geodetic coordinates ``[lon, lat, alt]``.
            Longitude and latitude in *rad* (or *deg*), altitude in *m*
            above the WGS84 ellipsoid.

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax.constants import WGS84_a
        >>> from astrojax.coordinates import position_ecef_to_geodetic
        >>> x_ecef = jnp.array([WGS84_a, 0.0, 0.0])
        >>> geod = position_ecef_to_geodetic(x_ecef)
        >>> float(geod[2])  # altitude ≈ 0
        0.0
    """
    x_ecef = jnp.asarray(x_ecef, dtype=jnp.float32)

    x = x_ecef[0]
    y = x_ecef[1]
    z = x_ecef[2]

    eps = jnp.float32(1.0e-3) * WGS84_a * jnp.finfo(jnp.float32).eps
    rho2 = x * x + y * y

    # State: (dz, dz_prev, iteration_count)
    dz0 = jnp.float32(ECC2) * z

    def cond(state):
        dz, dz_prev, i = state
        return (jnp.abs(dz - dz_prev) > eps) & (i < 10)

    def body(state):
        dz, _, i = state
        zdz = z + dz
        Nh = jnp.sqrt(rho2 + zdz * zdz)
        sinphi = zdz / Nh
        N = WGS84_a / jnp.sqrt(1.0 - ECC2 * sinphi * sinphi)
        dz_new = N * jnp.float32(ECC2) * sinphi
        return (dz_new, dz, i + 1)

    # Initial state: force first iteration by setting dz_prev far from dz0
    init_state = (dz0, dz0 + jnp.float32(1e10), jnp.int32(0))
    dz_final, _, _ = jax.lax.while_loop(cond, body, init_state)

    zdz = z + dz_final
    lon = jnp.arctan2(y, x)
    lat = jnp.arctan2(zdz, jnp.sqrt(rho2))

    sinphi = zdz / jnp.sqrt(rho2 + zdz * zdz)
    N = WGS84_a / jnp.sqrt(1.0 - ECC2 * sinphi * sinphi)
    alt = jnp.sqrt(rho2 + zdz * zdz) - N

    if use_degrees:
        lon = jnp.rad2deg(lon)
        lat = jnp.rad2deg(lat)

    return jnp.array([lon, lat, alt])
