"""Geocentric (spherical Earth) coordinate transformations.

Converts between geocentric coordinates ``[longitude, latitude, altitude]``
and Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates ``[x, y, z]``.

The geocentric model treats the Earth as a perfect sphere with radius
equal to the WGS84 semi-major axis.  For an ellipsoidal model, use the
geodetic functions in :mod:`astrojax.coordinates.geodetic`.

All inputs and outputs use SI base units (metres, radians).

References:
    1. O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods
       and Applications*, Springer, 2012, Sec. 5.3.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.constants import WGS84_a


def position_geocentric_to_ecef(
    x_geoc: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Convert geocentric position to ECEF Cartesian coordinates.

    Args:
        x_geoc: Geocentric coordinates ``[lon, lat, alt]``.
            Longitude and latitude in *rad* (or *deg* if ``use_degrees=True``),
            altitude in *m* above the spherical Earth surface.
        use_degrees: If ``True``, interpret longitude and latitude as degrees.

    Returns:
        jax.Array: ECEF position ``[x, y, z]`` in *m*.

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax.coordinates import position_geocentric_to_ecef
        >>> x_geoc = jnp.array([0.0, 0.0, 0.0])
        >>> x_ecef = position_geocentric_to_ecef(x_geoc)
        >>> float(x_ecef[0])  # WGS84_a on the equator
        6378137.0
    """
    x_geoc = jnp.asarray(x_geoc, dtype=get_dtype())

    lon = x_geoc[0]
    lat = x_geoc[1]
    alt = x_geoc[2]

    if use_degrees:
        lon = jnp.deg2rad(lon)
        lat = jnp.deg2rad(lat)

    r = WGS84_a + alt
    x = r * jnp.cos(lat) * jnp.cos(lon)
    y = r * jnp.cos(lat) * jnp.sin(lon)
    z = r * jnp.sin(lat)

    return jnp.array([x, y, z])


def position_ecef_to_geocentric(
    x_ecef: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Convert ECEF Cartesian coordinates to geocentric position.

    Args:
        x_ecef: ECEF position ``[x, y, z]`` in *m*.
        use_degrees: If ``True``, return longitude and latitude in degrees.

    Returns:
        jax.Array: Geocentric coordinates ``[lon, lat, alt]``.
            Longitude and latitude in *rad* (or *deg*), altitude in *m*.

    Example:
        >>> import jax.numpy as jnp
        >>> from astrojax.constants import WGS84_a
        >>> from astrojax.coordinates import position_ecef_to_geocentric
        >>> x_ecef = jnp.array([WGS84_a, 0.0, 0.0])
        >>> geoc = position_ecef_to_geocentric(x_ecef)
        >>> float(geoc[2])  # altitude ≈ 0
        0.0
    """
    x_ecef = jnp.asarray(x_ecef, dtype=get_dtype())

    x = x_ecef[0]
    y = x_ecef[1]
    z = x_ecef[2]

    lon = jnp.arctan2(y, x)
    lat = jnp.arctan2(z, jnp.sqrt(x * x + y * y))
    alt = jnp.sqrt(x * x + y * y + z * z) - WGS84_a

    if use_degrees:
        lon = jnp.rad2deg(lon)
        lat = jnp.rad2deg(lat)

    return jnp.array([lon, lat, alt])
