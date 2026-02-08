# Orbit Dynamics

The `astrojax.orbit_dynamics` module provides force models for orbit
propagation with perturbation modelling.  All functions are
JAX-traceable and compatible with `jax.jit`, `jax.vmap`, and
`jax.grad`.

## Overview

| Component | Functions | Description |
|-----------|-----------|-------------|
| **Ephemerides** | `sun_position`, `moon_position` | Low-precision analytical Sun/Moon positions |
| **Gravity** | `accel_point_mass`, `accel_gravity` | Point-mass gravitational acceleration |
| **Third body** | `accel_third_body_sun`, `accel_third_body_moon` | Sun/Moon gravitational perturbations |
| **Density** | `density_harris_priester` | Harris-Priester atmospheric density model |
| **Drag** | `accel_drag` | Atmospheric drag acceleration |
| **SRP** | `accel_srp` | Solar radiation pressure acceleration |
| **Eclipse** | `eclipse_conical`, `eclipse_cylindrical` | Shadow models for SRP modulation |

## Ephemerides

The Sun and Moon positions use the low-precision analytical model from
Montenbruck & Gill (2012, p. 70-73).  Positions are returned in the
EME2000 (ECI) inertial frame in metres.

```python
from astrojax import Epoch
from astrojax.orbit_dynamics import sun_position, moon_position

epc = Epoch(2024, 6, 15, 12, 0, 0)
r_sun = sun_position(epc)    # shape (3,), units: m
r_moon = moon_position(epc)  # shape (3,), units: m
```

The analytical model provides ~0.01 degree accuracy for the Sun and
~0.3 degree for the Moon, sufficient for perturbation force modelling.

## Gravity

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

## Third-Body Perturbations

Convenience functions combine the ephemeris lookup with the point-mass
gravity calculation:

```python
from astrojax import Epoch
from astrojax.orbit_dynamics import accel_third_body_sun, accel_third_body_moon

epc = Epoch(2024, 6, 15, 12, 0, 0)
r = jnp.array([6878e3, 0.0, 0.0])

a_sun = accel_third_body_sun(epc, r)    # ~1e-7 m/s^2 at LEO
a_moon = accel_third_body_moon(epc, r)  # ~1e-6 m/s^2 at LEO
```

## Atmospheric Density

The Harris-Priester model computes atmospheric density accounting for
diurnal variations caused by solar heating.  It is valid for altitudes
between 100 km and 1000 km.

```python
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

## Solar Radiation Pressure

SRP acceleration depends on the spacecraft's reflectivity coefficient,
cross-sectional area, and distance from the Sun:

```python
from astrojax.orbit_dynamics import accel_srp, eclipse_cylindrical
from astrojax.constants import P_SUN

a_srp = accel_srp(r, r_sun, mass=100.0, cr=1.8, area=1.0, p0=P_SUN)
nu = eclipse_cylindrical(r, r_sun)  # 0.0 (shadow) or 1.0 (illuminated)
a_effective = nu * a_srp  # Zero in shadow
```

Two shadow models are available:

| Model | Function | Output | Penumbra |
|-------|----------|--------|----------|
| Conical | `eclipse_conical` | 0.0 - 1.0 | Yes |
| Cylindrical | `eclipse_cylindrical` | 0.0 or 1.0 | No |

## JAX Compatibility

All orbit dynamics functions use JAX primitives and are fully compatible
with `jax.jit` for compilation and `jax.grad` for automatic
differentiation:

```python
import jax

# JIT-compiled force computation
jit_sun = jax.jit(accel_third_body_sun)
a = jit_sun(epc, r)

# Gradient of acceleration magnitude w.r.t. position
def accel_mag(r):
    return jnp.linalg.norm(accel_gravity(r))

grad_accel = jax.grad(accel_mag)(r)
```

!!! note "Float precision"
    At float32 (default), ephemeris positions are accurate to ~10 m and
    accelerations to ~1e-6 relative error.  For reference-quality
    comparisons, use `set_dtype(jnp.float64)` before any JIT compilation.
