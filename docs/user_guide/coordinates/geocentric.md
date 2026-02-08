# Geocentric Coordinates

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
