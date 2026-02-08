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

## JAX Compatibility

All coordinate functions are compatible with `jax.jit` and `jax.vmap`:

```python
import jax
import jax.numpy as jnp
from astrojax.constants import R_EARTH
from astrojax.coordinates import (
    relative_position_ecef_to_enz,
    position_enz_to_azel,
)

x_sta = jnp.array([R_EARTH, 0.0, 0.0])

# JIT-compiled ENZ conversion
jit_enz = jax.jit(
    lambda r: relative_position_ecef_to_enz(x_sta, r, use_geodetic=False)
)
x_sat = jnp.array([R_EARTH + 500e3, 0.0, 0.0])
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
