# Solar Radiation Pressure

SRP acceleration depends on the spacecraft's reflectivity coefficient,
cross-sectional area, and distance from the Sun:

```python
import jax.numpy as jnp
from astrojax.orbit_dynamics import accel_srp, eclipse_cylindrical
from astrojax.constants import P_SUN

r = jnp.array([6878e3, 0.0, 0.0])
r_sun = jnp.array([1.5e11, 0.0, 0.0])
a_srp = accel_srp(r, r_sun, mass=100.0, cr=1.8, area=1.0, p0=P_SUN)
nu = eclipse_cylindrical(r, r_sun)  # 0.0 (shadow) or 1.0 (illuminated)
a_effective = nu * a_srp  # Zero in shadow
```

Two shadow models are available:

| Model | Function | Output | Penumbra |
|-------|----------|--------|----------|
| Conical | `eclipse_conical` | 0.0 - 1.0 | Yes |
| Cylindrical | `eclipse_cylindrical` | 0.0 or 1.0 | No |
