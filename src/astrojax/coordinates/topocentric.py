"""East-North-Zenith (ENZ) topocentric coordinate transformations.

Converts between Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates
and a local topocentric frame defined by East, North, and Zenith axes at
an observer location on the Earth's surface.  Also provides conversion
from ENZ to azimuth, elevation, and range.

The ENZ frame is a right-handed coordinate system:

- **East** (E): tangent to the surface, pointing geographic east
- **North** (N): tangent to the surface, pointing geographic north
- **Zenith** (Z): normal to the surface, pointing radially outward

All inputs and outputs use SI base units (metres, radians) unless
``use_degrees=True`` is specified.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.coordinates.geocentric import position_ecef_to_geocentric
from astrojax.coordinates.geodetic import position_ecef_to_geodetic


def rotation_ellipsoid_to_enz(
    x_ellipsoid: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Compute the rotation matrix from ECEF to East-North-Zenith (ENZ).

    The input ellipsoidal coordinates specify the observer location.  The
    returned 3x3 matrix transforms a vector in ECEF into the local ENZ
    frame at that location.

    Args:
        x_ellipsoid: Ellipsoidal coordinates ``[lon, lat, alt]``.
            Longitude and latitude in *rad* (or *deg* if ``use_degrees=True``),
            altitude in *m*.
        use_degrees: If ``True``, interpret longitude and latitude as degrees.

    Returns:
        3x3 rotation matrix (ECEF → ENZ).

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.coordinates import rotation_ellipsoid_to_enz
        x_geo = jnp.array([30.0, 60.0, 0.0])
        rot = rotation_ellipsoid_to_enz(x_geo, use_degrees=True)
        ```
    """
    x_ellipsoid = jnp.asarray(x_ellipsoid, dtype=get_dtype())

    lon = x_ellipsoid[0]
    lat = x_ellipsoid[1]

    if use_degrees:
        lon = jnp.deg2rad(lon)
        lat = jnp.deg2rad(lat)

    sin_lon = jnp.sin(lon)
    cos_lon = jnp.cos(lon)
    sin_lat = jnp.sin(lat)
    cos_lat = jnp.cos(lat)

    # Rows are E, N, Z basis vectors expressed in ECEF
    return jnp.array([
        [-sin_lon, cos_lon, 0.0],                             # East
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],    # North
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],      # Zenith
    ])


def rotation_enz_to_ellipsoid(
    x_ellipsoid: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Compute the rotation matrix from ENZ to ECEF.

    This is the transpose of :func:`rotation_ellipsoid_to_enz`.

    Args:
        x_ellipsoid: Ellipsoidal coordinates ``[lon, lat, alt]``.
            Longitude and latitude in *rad* (or *deg* if ``use_degrees=True``),
            altitude in *m*.
        use_degrees: If ``True``, interpret longitude and latitude as degrees.

    Returns:
        3x3 rotation matrix (ENZ → ECEF).

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.coordinates import rotation_enz_to_ellipsoid
        x_geo = jnp.array([30.0, 60.0, 0.0])
        rot_inv = rotation_enz_to_ellipsoid(x_geo, use_degrees=True)
        ```
    """
    return rotation_ellipsoid_to_enz(x_ellipsoid, use_degrees).T


def relative_position_ecef_to_enz(
    location_ecef: ArrayLike,
    r_ecef: ArrayLike,
    use_geodetic: bool = True,
) -> Array:
    """Convert an ECEF position to ENZ relative to an observer station.

    Computes the ENZ components of ``r_ecef - location_ecef`` by first
    determining the station's ellipsoidal coordinates (geodetic or
    geocentric) and then applying the ECEF→ENZ rotation.

    Args:
        location_ecef: ECEF position of the observing station ``[x, y, z]``
            in *m*.
        r_ecef: ECEF position of the target object ``[x, y, z]`` in *m*.
        use_geodetic: If ``True`` (default), use geodetic coordinates to
            define the local frame.  If ``False``, use geocentric.

    Returns:
        Relative position ``[east, north, zenith]`` in *m*.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH
        from astrojax.coordinates import relative_position_ecef_to_enz
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        x_sat = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
        r_enz = relative_position_ecef_to_enz(x_sta, x_sat)
        ```
    """
    location_ecef = jnp.asarray(location_ecef, dtype=get_dtype())
    r_ecef = jnp.asarray(r_ecef, dtype=get_dtype())

    if use_geodetic:
        x_ellipsoid = position_ecef_to_geodetic(location_ecef)
    else:
        x_ellipsoid = position_ecef_to_geocentric(location_ecef)

    rot = rotation_ellipsoid_to_enz(x_ellipsoid)
    return rot @ (r_ecef - location_ecef)


def relative_position_enz_to_ecef(
    location_ecef: ArrayLike,
    r_enz: ArrayLike,
    use_geodetic: bool = True,
) -> Array:
    """Convert an ENZ relative position back to absolute ECEF.

    Inverse of :func:`relative_position_ecef_to_enz`.

    Args:
        location_ecef: ECEF position of the observing station ``[x, y, z]``
            in *m*.
        r_enz: Relative position ``[east, north, zenith]`` in *m*.
        use_geodetic: If ``True`` (default), use geodetic coordinates to
            define the local frame.  If ``False``, use geocentric.

    Returns:
        Absolute ECEF position ``[x, y, z]`` in *m*.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH
        from astrojax.coordinates import relative_position_enz_to_ecef
        x_sta = jnp.array([R_EARTH, 0.0, 0.0])
        r_enz = jnp.array([0.0, 0.0, 500e3])
        r_ecef = relative_position_enz_to_ecef(x_sta, r_enz)
        ```
    """
    location_ecef = jnp.asarray(location_ecef, dtype=get_dtype())
    r_enz = jnp.asarray(r_enz, dtype=get_dtype())

    if use_geodetic:
        x_ellipsoid = position_ecef_to_geodetic(location_ecef)
    else:
        x_ellipsoid = position_ecef_to_geocentric(location_ecef)

    rot_t = rotation_enz_to_ellipsoid(x_ellipsoid)
    return location_ecef + rot_t @ r_enz


def position_enz_to_azel(
    x_enz: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Convert ENZ position to azimuth, elevation, and range.

    Azimuth is measured clockwise from North (0° = North, 90° = East).
    Elevation is measured from the local horizon (0°) to zenith (90°).

    At the zenith singularity (elevation = 90°), azimuth is defined as 0.

    Args:
        x_enz: ENZ position ``[east, north, zenith]`` in *m*.
        use_degrees: If ``True``, return azimuth and elevation in degrees.

    Returns:
        ``[azimuth, elevation, range]``.
            Azimuth in ``[0, 2pi)`` rad (or ``[0, 360)`` deg),
            elevation in ``[-pi/2, pi/2]`` rad, range in *m*.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.coordinates import position_enz_to_azel
        x_enz = jnp.array([100.0, 0.0, 0.0])
        azel = position_enz_to_azel(x_enz, use_degrees=True)
        # azel ≈ [90.0, 0.0, 100.0]
        ```
    """
    x_enz = jnp.asarray(x_enz, dtype=get_dtype())

    e = x_enz[0]
    n = x_enz[1]
    z = x_enz[2]

    # Range
    rho = jnp.sqrt(e * e + n * n + z * z)

    # Elevation
    horiz = jnp.sqrt(e * e + n * n)
    el = jnp.arctan2(z, horiz)

    # Azimuth (clockwise from north)
    # At zenith (el == pi/2), azimuth is ambiguous; define as 0
    az_raw = jnp.arctan2(e, n)
    az_wrapped = jnp.where(az_raw >= 0.0, az_raw, az_raw + 2.0 * jnp.pi)
    at_zenith = horiz == 0.0
    az = jnp.where(at_zenith, 0.0, az_wrapped)

    if use_degrees:
        az = jnp.rad2deg(az)
        el = jnp.rad2deg(el)

    return jnp.array([az, el, rho])
