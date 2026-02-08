# Coordinate Transformations

The `astrojax.coordinates` module provides functions for converting
between the coordinate representations commonly used in astrodynamics.

## Coordinate Systems

| System | Format | Description |
|--------|--------|-------------|
| **ECEF** | `[x, y, z]` metres | Earth-Centered Earth-Fixed Cartesian |
| **Geocentric** | `[lon, lat, alt]` | Spherical Earth model (radius = WGS84 semi-major axis) |
| **Geodetic** | `[lon, lat, alt]` | WGS84 ellipsoid model (accounts for Earth's oblateness) |
| **Keplerian** | `[a, e, i, Ω, ω, M]` | Classical orbital elements |
| **ENZ** | `[east, north, zenith]` metres | Local topocentric frame at an observer station |

## Geocentric Coordinates

Geocentric coordinates treat the Earth as a perfect sphere.  The
conversion between `[lon, lat, alt]` and ECEF is straightforward
trigonometry:

```python
import jax.numpy as jnp
from astrojax.coordinates import (
    position_geocentric_to_ecef,
    position_ecef_to_geocentric,
)

x_geoc = jnp.array([0.0, 0.0, 0.0])  # lon=0, lat=0, alt=0
x_ecef = position_geocentric_to_ecef(x_geoc)  # [WGS84_a, 0, 0]

# Round-trip
x_back = position_ecef_to_geocentric(x_ecef)
```

Use `use_degrees=True` to work in degrees instead of radians.

## Geodetic Coordinates

Geodetic coordinates account for the WGS84 ellipsoid, giving more
accurate surface positions.  The forward transform is closed-form;
the inverse uses Bowring's iterative method:

```python
from astrojax.coordinates import (
    position_geodetic_to_ecef,
    position_ecef_to_geodetic,
)

# Boulder, CO
x_geod = jnp.array([-105.0, 40.0, 1655.0])
x_ecef = position_geodetic_to_ecef(x_geod, use_degrees=True)
x_back = position_ecef_to_geodetic(x_ecef, use_degrees=True)
```

!!! note "Geocentric vs Geodetic"
    At the equator and poles, geocentric and geodetic latitudes are
    identical.  The maximum difference (~0.19°) occurs near 45° latitude,
    corresponding to ~21 km on the surface.

## Keplerian Orbital Elements

Convert between classical orbital elements and ECI Cartesian states:

```python
from astrojax.coordinates import state_koe_to_eci, state_eci_to_koe
from astrojax.constants import R_EARTH

# Sun-synchronous LEO
oe = jnp.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0])
state = state_koe_to_eci(oe, use_degrees=True)  # [x, y, z, vx, vy, vz]
oe_back = state_eci_to_koe(state, use_degrees=True)
```

## ENZ Topocentric Coordinates

The East-North-Zenith (ENZ) frame is a local horizontal coordinate
system centred at a ground station.  It is essential for satellite
tracking, converting a satellite's ECEF position into observer-relative
azimuth, elevation, and range.

### Rotation Matrices

Build the ECEF ↔ ENZ rotation matrix from the station's ellipsoidal
coordinates:

```python
from astrojax.coordinates import (
    rotation_ellipsoid_to_enz,
    rotation_enz_to_ellipsoid,
)

# Station at lon=30°, lat=60°
x_station = jnp.array([30.0, 60.0, 0.0])
R_ecef_to_enz = rotation_ellipsoid_to_enz(x_station, use_degrees=True)
R_enz_to_ecef = rotation_enz_to_ellipsoid(x_station, use_degrees=True)

# These are transposes of each other
assert jnp.allclose(R_ecef_to_enz @ R_enz_to_ecef, jnp.eye(3), atol=1e-6)
```

### Relative Positions

Compute the ENZ position of a satellite relative to a ground station,
both given in ECEF:

```python
from astrojax.constants import R_EARTH
from astrojax.coordinates import (
    relative_position_ecef_to_enz,
    relative_position_enz_to_ecef,
)

# Station on the equator at the prime meridian
x_sta = jnp.array([R_EARTH, 0.0, 0.0])

# Satellite 500 km overhead
x_sat = jnp.array([R_EARTH + 500e3, 0.0, 0.0])

r_enz = relative_position_ecef_to_enz(x_sta, x_sat)
# r_enz ≈ [0, 0, 500000] (directly above → pure zenith)

# Round-trip back to ECEF
x_sat_back = relative_position_enz_to_ecef(x_sta, r_enz)
```

The `use_geodetic` parameter controls whether the station's ellipsoidal
coordinates are computed using the geodetic (default) or geocentric
model.

### Azimuth, Elevation, and Range

Convert an ENZ position to observer-relative azimuth (clockwise from
North), elevation (from horizon), and slant range:

```python
from astrojax.coordinates import position_enz_to_azel

# Satellite to the east at the horizon
r_enz = jnp.array([100.0, 0.0, 0.0])
azel = position_enz_to_azel(r_enz, use_degrees=True)
# azel ≈ [90.0, 0.0, 100.0]  (az=90° East, el=0°, range=100m)
```

## JAX Compatibility

All coordinate functions are compatible with `jax.jit` and `jax.vmap`:

```python
import jax

# JIT-compiled ENZ conversion
jit_enz = jax.jit(
    lambda r: relative_position_ecef_to_enz(x_sta, r, use_geodetic=False)
)
r_enz = jit_enz(x_sat)

# Batch azimuth/elevation over multiple ENZ positions
enz_batch = jnp.array([
    [0.0, 0.0, 100.0],
    [100.0, 0.0, 0.0],
    [0.0, 100.0, 0.0],
])
azels = jax.vmap(position_enz_to_azel)(enz_batch)
```

!!! note "float32 precision"
    ENZ roundtrip transformations (ECEF → ENZ → ECEF) are accurate to
    ~1 m in position with the default float32 precision.
