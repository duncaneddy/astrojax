# ENZ Topocentric Coordinates

The East-North-Zenith (ENZ) frame is a local horizontal coordinate
system centred at a ground station.  It is essential for satellite
tracking, converting a satellite's ECEF position into observer-relative
azimuth, elevation, and range.

## Rotation Matrices

Build the ECEF ↔ ENZ rotation matrix from the station's ellipsoidal
coordinates:

```python
import jax.numpy as jnp
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

## Relative Positions

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

## Azimuth, Elevation, and Range

Convert an ENZ position to observer-relative azimuth (clockwise from
North), elevation (from horizon), and slant range:

```python
from astrojax.coordinates import position_enz_to_azel

# Satellite to the east at the horizon
r_enz = jnp.array([100.0, 0.0, 0.0])
azel = position_enz_to_azel(r_enz, use_degrees=True)
# azel ≈ [90.0, 0.0, 100.0]  (az=90° East, el=0°, range=100m)
```
