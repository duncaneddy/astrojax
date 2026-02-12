# Gravity

## Point-Mass Gravity

Point-mass gravity computes the Newtonian gravitational acceleration.
For the central body at the origin (Earth two-body), use the
convenience function `accel_gravity`:

```python
import jax.numpy as jnp
from astrojax.orbit_dynamics import accel_gravity, accel_point_mass
from astrojax.constants import GM_SUN

r = jnp.array([6878e3, 0.0, 0.0])  # LEO position [m]
a = accel_gravity(r)  # Earth two-body acceleration [m/s^2]
```

For third-body perturbations, `accel_point_mass` handles the indirect
acceleration term when the attracting body is not at the origin:

$$
\mathbf{a} = -\mu \left(\frac{\mathbf{d}}{|\mathbf{d}|^3}
+ \frac{\mathbf{r}_{\text{body}}}{|\mathbf{r}_{\text{body}}|^3}\right)
$$

where $\mathbf{d} = \mathbf{r} - \mathbf{r}_{\text{body}}$.

## Spherical Harmonic Gravity

For accurate orbit propagation in LEO, point-mass gravity is
insufficient.  The `GravityModel` class loads spherical harmonic
gravity field models (Stokes coefficients $C_{nm}$, $S_{nm}$) from
standard ICGEM GFC format files:

```python
from astrojax.orbit_dynamics import GravityModel, accel_gravity_spherical_harmonics

# Load a packaged model
model = GravityModel.from_type("JGM3")  # 70x70, also: "EGM2008_360", "GGM05S"
```

The acceleration function uses the V/W matrix recursion from
Montenbruck & Gill (2012, p. 56-68) to evaluate the gravity field at a
given position.  The position must be provided in the ECI frame along
with the ECI-to-ECEF rotation matrix:

```python
import jax.numpy as jnp
from astrojax.frames import rotation_eci_to_ecef
from astrojax.eop import zero_eop
from astrojax import Epoch

eop = zero_eop()
epc = Epoch(2024, 6, 15, 12, 0, 0)
r_eci = jnp.array([6878e3, 0.0, 0.0])
R = rotation_eci_to_ecef(eop, epc)

# Evaluate to degree and order 20
a = accel_gravity_spherical_harmonics(r_eci, R, model, n_max=20, m_max=20)
```

Three packaged gravity models are available:

| Model | Degree/Order | Description |
|-------|-------------|-------------|
| `"EGM2008_360"` | 360 x 360 | Truncated EGM2008, high-resolution |
| `"GGM05S"` | 180 x 180 | GRACE satellite-only model |
| `"JGM3"` | 70 x 70 | Joint Gravity Model 3 |

Models can also be loaded from custom GFC files using
`GravityModel.from_file("path/to/model.gfc")`.

To reduce computation time, truncate a model to a lower degree/order:

```python
model = GravityModel.from_type("EGM2008_360")
model.set_max_degree_order(20, 20)  # Truncate to 20x20
```
