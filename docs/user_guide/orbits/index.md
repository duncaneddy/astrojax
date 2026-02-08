# Orbits

The `astrojax.orbits` module provides Keplerian orbital mechanics
functions for Earth-centric orbits. All functions use JAX primitives
and are compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.

## JAX Compatibility

All functions use JAX operations internally and work with `jax.jit`,
`jax.vmap`, and `jax.grad`:

```python
import jax
import jax.numpy as jnp
from astrojax.constants import R_EARTH
from astrojax.orbits import orbital_period

a = R_EARTH + 500e3

# JIT compilation
jit_period = jax.jit(orbital_period)
T = jit_period(a)

# Batch evaluation
smas = jnp.array([R_EARTH + 400e3, R_EARTH + 500e3, R_EARTH + 600e3])
periods = jax.vmap(orbital_period)(smas)

# Gradient (dT/da)
dT_da = jax.grad(orbital_period)(jnp.float32(a))
```

The Kepler solver (`anomaly_mean_to_eccentric`) is also differentiable,
enabling gradient-based optimization through anomaly conversions.

!!! note "float32 precision"
    All computations use float32 for GPU/TPU compatibility. Orbital
    periods are accurate to ~1 second, velocities to ~0.1 m/s, and
    anomaly conversions to ~0.01 degrees. Roundtrip anomaly conversions
    (e.g., $\nu \to M \to \nu$) are accurate to ~0.1 degrees for
    eccentricities up to 0.7.
