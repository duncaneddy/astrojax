# Orbits

The `astrojax.orbits` module provides Keplerian orbital mechanics
functions for Earth-centric orbits. All functions use JAX primitives
and are compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.

## Orbital Period and Semi-Major Axis

The orbital period $T$ is related to the semi-major axis $a$ by:

$$
T = 2\pi \sqrt{\frac{a^3}{\mu_\oplus}}
$$

```python
from astrojax.constants import R_EARTH
from astrojax.orbits import orbital_period, semimajor_axis_from_orbital_period

a = R_EARTH + 500e3          # 500 km LEO
T = orbital_period(a)         # ~5677 seconds (~94.6 minutes)

# Inverse: recover SMA from period
a_back = semimajor_axis_from_orbital_period(T)
```

You can also compute the period directly from an ECI state vector using
the vis-viva equation:

```python
import jax.numpy as jnp
from astrojax.constants import GM_EARTH
from astrojax.orbits import orbital_period_from_state

r = R_EARTH + 500e3
v = jnp.sqrt(GM_EARTH / r)
state = jnp.array([r, 0.0, 0.0, 0.0, v, 0.0])
T = orbital_period_from_state(state)
```

## Mean Motion

Mean motion $n$ is the angular rate of a Keplerian orbit:

$$
n = \sqrt{\frac{\mu_\oplus}{a^3}}
$$

```python
from astrojax.orbits import mean_motion, semimajor_axis

n = mean_motion(a)                         # rad/s
n_deg = mean_motion(a, use_degrees=True)   # deg/s

# Inverse: recover SMA from mean motion
a_back = semimajor_axis(n)
```

## Velocities at Apsides

For an elliptical orbit with semi-major axis $a$ and eccentricity $e$:

$$
v_{\text{perigee}} = \sqrt{\frac{\mu}{a}} \sqrt{\frac{1+e}{1-e}}, \quad
v_{\text{apogee}} = \sqrt{\frac{\mu}{a}} \sqrt{\frac{1-e}{1+e}}
$$

```python
from astrojax.orbits import perigee_velocity, apogee_velocity

vp = perigee_velocity(a, 0.01)  # m/s at perigee
va = apogee_velocity(a, 0.01)   # m/s at apogee
```

## Distances and Altitudes

```python
from astrojax.orbits import (
    periapsis_distance, apoapsis_distance,
    perigee_altitude, apogee_altitude,
)

rp = periapsis_distance(a, 0.01)   # a(1-e), metres
ra = apoapsis_distance(a, 0.01)    # a(1+e), metres
hp = perigee_altitude(a, 0.01)     # a(1-e) - R_EARTH, metres
ha = apogee_altitude(a, 0.01)      # a(1+e) - R_EARTH, metres
```

## Anomaly Conversions

The module provides six functions for converting between the three
anomaly types: mean ($M$), eccentric ($E$), and true ($\nu$).

The direct conversions are:

- **Eccentric to mean** (Kepler's equation): $M = E - e \sin E$
- **Mean to eccentric** (Newton-Raphson iteration): solves $M = E - e \sin E$ for $E$
- **True to eccentric**: $E = \text{atan2}\!\bigl(\sin\nu\,\sqrt{1-e^2},\; \cos\nu + e\bigr)$
- **Eccentric to true**: $\nu = \text{atan2}\!\bigl(\sin E\,\sqrt{1-e^2},\; \cos E - e\bigr)$

The composite conversions chain through eccentric anomaly:

- **True to mean**: $\nu \to E \to M$
- **Mean to true**: $M \to E \to \nu$

```python
from astrojax.orbits import (
    anomaly_eccentric_to_mean,
    anomaly_mean_to_eccentric,
    anomaly_true_to_eccentric,
    anomaly_eccentric_to_true,
    anomaly_true_to_mean,
    anomaly_mean_to_true,
)

e = 0.1

# Degrees
M = anomaly_eccentric_to_mean(90.0, e, use_degrees=True)   # ~84.27 deg
E = anomaly_mean_to_eccentric(M, e, use_degrees=True)      # ~90.0 deg (roundtrip)

# Radians (default)
import jax.numpy as jnp
nu = anomaly_mean_to_true(jnp.pi / 2.0, e)
```

### Kepler Equation Solver

The mean-to-eccentric anomaly conversion solves Kepler's equation using
Newton-Raphson iteration. The solver uses `jax.lax.fori_loop` with 10
fixed iterations, making it fully traceable by JAX's compiler. The
initial guess is $E_0 = M$ for $e < 0.8$ and $E_0 = \pi$ for higher
eccentricities.

## Special Orbits

### Sun-Synchronous Orbit

The inclination required for a Sun-synchronous orbit is determined by
the J2 perturbation on the RAAN precession rate:

```python
from astrojax.orbits import sun_synchronous_inclination

inc = sun_synchronous_inclination(R_EARTH + 500e3, 0.001, use_degrees=True)
# ~97.4 degrees
```

### Geostationary Orbit

```python
from astrojax.orbits import geo_sma

a_geo = geo_sma()  # ~42164 km
```

## JAX Compatibility

All functions use JAX operations internally and work with `jax.jit`,
`jax.vmap`, and `jax.grad`:

```python
import jax

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
