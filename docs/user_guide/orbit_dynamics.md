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
| **Spherical Harmonics** | `GravityModel`, `accel_gravity_spherical_harmonics` | Spherical harmonic gravity field |
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
from astrojax import Epoch

epc = Epoch(2024, 6, 15, 12, 0, 0)
r_eci = jnp.array([6878e3, 0.0, 0.0])
R = rotation_eci_to_ecef(epc)

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

## Composing Force Models

The individual force model functions above can be composed into a single
dynamics function using `create_orbit_dynamics`.  This factory takes a
reference epoch and a configuration object, and returns a
`dynamics(t, state) -> derivative` closure compatible with all astrojax
integrators.

### Quick Start: Two-Body

```python
from astrojax import Epoch, create_orbit_dynamics
from astrojax.integrators import rk4_step
import jax.numpy as jnp

epoch_0 = Epoch(2024, 6, 15, 12, 0, 0)
dynamics = create_orbit_dynamics(epoch_0)  # default: point-mass gravity

x0 = jnp.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
result = rk4_step(dynamics, 0.0, x0, 60.0)
```

### LEO with Perturbations

Use `ForceModelConfig` to enable perturbation forces:

```python
from astrojax import ForceModelConfig, SpacecraftParams, GravityModel

config = ForceModelConfig(
    gravity_type="spherical_harmonics",
    gravity_model=GravityModel.from_type("JGM3"),
    gravity_degree=20,
    gravity_order=20,
    drag=True,
    srp=True,
    third_body_sun=True,
    third_body_moon=True,
    spacecraft=SpacecraftParams(mass=500.0, cd=2.2, cr=1.3),
)

dynamics = create_orbit_dynamics(epoch_0, config)
```

Or use a preset for common scenarios:

```python
config = ForceModelConfig.leo_default()   # 20x20 SH + drag + SRP + Sun/Moon
config = ForceModelConfig.geo_default()   # 8x8 SH + SRP + Sun/Moon (no drag)
config = ForceModelConfig.two_body()      # point-mass only
```

### Multi-Step Propagation

Use `jax.lax.scan` for efficient multi-step propagation:

```python
import jax

dt = 60.0
def scan_step(carry, _):
    t, state = carry
    result = rk4_step(dynamics, t, state, dt)
    return (t + dt, result.state), result.state

(_, _), trajectory = jax.lax.scan(scan_step, (0.0, x0), None, length=100)
```

### Adding Thrust

Use the integrator `control` parameter to add thrust or other external
forces:

```python
def thrust(t, x):
    v = x[3:]
    v_hat = v / jnp.linalg.norm(v)
    return jnp.concatenate([jnp.zeros(3), 1e-3 * v_hat])

result = rk4_step(dynamics, 0.0, x0, 60.0, control=thrust)
```

The effective derivative at each integrator stage is
`dynamics(t, x) + control(t, x)`.

### Spacecraft Parameters

`SpacecraftParams` holds physical properties used by drag and SRP:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mass` | 1000.0 | Spacecraft mass [kg] |
| `drag_area` | 10.0 | Wind-facing cross-sectional area [m^2] |
| `srp_area` | 10.0 | Sun-facing cross-sectional area [m^2] |
| `cd` | 2.2 | Coefficient of drag |
| `cr` | 1.3 | Coefficient of reflectivity |

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
