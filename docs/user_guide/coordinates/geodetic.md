# Geodetic Coordinates

Geodetic coordinates account for the WGS84 ellipsoid, giving more
accurate surface positions.  The forward transform is closed-form;
the inverse uses Bowring's iterative method:

```python
import jax.numpy as jnp
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
