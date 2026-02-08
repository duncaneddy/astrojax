# Third-Body Perturbations

Convenience functions combine the ephemeris lookup with the point-mass
gravity calculation:

```python
import jax.numpy as jnp
from astrojax import Epoch
from astrojax.orbit_dynamics import accel_third_body_sun, accel_third_body_moon

epc = Epoch(2024, 6, 15, 12, 0, 0)
r = jnp.array([6878e3, 0.0, 0.0])

a_sun = accel_third_body_sun(epc, r)    # ~1e-7 m/s^2 at LEO
a_moon = accel_third_body_moon(epc, r)  # ~1e-6 m/s^2 at LEO
```
