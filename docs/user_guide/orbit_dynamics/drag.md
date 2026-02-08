# Atmospheric Density and Drag

## Atmospheric Density

The Harris-Priester model computes atmospheric density accounting for
diurnal variations caused by solar heating.  It is valid for altitudes
between 100 km and 1000 km.

```python
import jax.numpy as jnp
from astrojax import Epoch
from astrojax.orbit_dynamics import density_harris_priester, sun_position

epc = Epoch(2024, 6, 15, 12, 0, 0)
r_ecef = jnp.array([0.0, 0.0, -(6378e3 + 400e3)])  # 400 km altitude
r_sun = sun_position(epc)
rho = density_harris_priester(r_ecef, r_sun)  # kg/m^3
```

## Atmospheric Drag

The drag model computes the non-conservative acceleration due to
atmospheric drag, accounting for the relative velocity between the
spacecraft and the co-rotating atmosphere:

```python
from astrojax.orbit_dynamics import accel_drag

x = jnp.array([6878e3, 0.0, 0.0, 0.0, 7500.0, 0.0])  # ECI state
T = jnp.eye(3)  # ECI to TOD rotation (identity for simplicity)
a = accel_drag(x, rho, mass=100.0, area=1.0, cd=2.3, T=T)
```

The rotation matrix $T$ transforms from ECI to the true-of-date (TOD)
frame.  For simplified models, the identity matrix can be used.
