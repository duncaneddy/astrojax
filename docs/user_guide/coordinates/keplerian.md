# Keplerian Orbital Elements

Convert between classical orbital elements and ECI Cartesian states:

```python
import jax.numpy as jnp
from astrojax.coordinates import state_koe_to_eci, state_eci_to_koe
from astrojax.constants import R_EARTH

# Sun-synchronous LEO
oe = jnp.array([R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0])
state = state_koe_to_eci(oe, use_degrees=True)  # [x, y, z, vx, vy, vz]
oe_back = state_eci_to_koe(state, use_degrees=True)
```
