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
| **Density** | `density_harris_priester`, `density_nrlmsise00` | Atmospheric density models ([Harris-Priester](drag.md#harris-priester), [NRLMSISE-00](drag.md#nrlmsise-00)) |
| **Drag** | `accel_drag` | Atmospheric drag acceleration |
| **SRP** | `accel_srp` | Solar radiation pressure acceleration |
| **Eclipse** | `eclipse_conical`, `eclipse_cylindrical` | Shadow models for SRP modulation |

## JAX Compatibility

All orbit dynamics functions use JAX primitives and are fully compatible
with `jax.jit` for compilation and `jax.grad` for automatic
differentiation:

```python
import jax
import jax.numpy as jnp
from astrojax.orbit_dynamics import accel_third_body_sun, accel_gravity
from astrojax import Epoch

epc = Epoch(2024, 6, 15, 12, 0, 0)
r = jnp.array([6878e3, 0.0, 0.0])

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
