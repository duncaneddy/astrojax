# Orbit Measurements

The `astrojax.orbit_measurements` module provides measurement functions
and noise covariance constructors for orbit determination sensors. These
are designed to plug directly into the estimation filters from
`astrojax.estimation`.

## Purpose

In state estimation, a **measurement function** `h(x)` maps the full
state vector to an observable quantity (e.g., extracting position from a
position-velocity state). A **measurement noise covariance** `R`
describes the sensor's uncertainty. Together, they define how a filter
incorporates sensor data.

This module provides both components for common orbit determination
sensors so you can pass them directly to `ekf_update` or `ukf_update`.

## Available Models

### GNSS

| Function | Description |
|----------|-------------|
| `gnss_position_measurement` | Extracts position `[x, y, z]` from the state vector |
| `gnss_measurement_noise` | Builds a `(3, 3)` diagonal noise covariance from a position sigma |
| `gnss_position_velocity_measurement` | Extracts position and velocity `[x, y, z, vx, vy, vz]` from the state vector |
| `gnss_position_velocity_noise` | Builds a `(6, 6)` diagonal noise covariance from position and velocity sigmas |

## Usage

### Position-Only GNSS

```python
import jax.numpy as jnp
from astrojax.orbit_measurements import (
    gnss_position_measurement,
    gnss_measurement_noise,
)
from astrojax.estimation import FilterState, ekf_update

# Build the noise covariance: 10 m 1-sigma position noise
R = gnss_measurement_noise(10.0)  # (3, 3) diagonal, 100 on diagonal

# Simulated noisy position observation
z = jnp.array([6878e3 + 5.0, 3.0, -2.0])

# Update a filter state with the observation
state = FilterState(
    x=jnp.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0]),
    P=jnp.diag(jnp.array([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])),
)
result = ekf_update(state, z, gnss_position_measurement, R)
```

### Position-Velocity GNSS

```python
from astrojax.orbit_measurements import (
    gnss_position_velocity_measurement,
    gnss_position_velocity_noise,
)

# 10 m position noise, 0.1 m/s velocity noise
R_pv = gnss_position_velocity_noise(10.0, 0.1)  # (6, 6) diagonal

z_pv = jnp.array([6878e3 + 5.0, 3.0, -2.0, 0.01, 7612.0 + 0.05, -0.01])
result = ekf_update(state, z_pv, gnss_position_velocity_measurement, R_pv)
```

## Custom Measurement Functions

You can write your own measurement functions following the same
pattern: a pure function `h(x) -> z` that maps a state vector to an
observation vector.

```python
def range_measurement(state):
    """Range from the origin to the spacecraft."""
    r = state[:3]
    return jnp.array([jnp.linalg.norm(r)])

R_range = jnp.array([[25.0]])  # 5 m 1-sigma range noise
result = ekf_update(state, z_range, range_measurement, R_range)
```

The EKF will compute the measurement Jacobian automatically via
`jax.jacfwd`. The UKF evaluates the function at sigma points, so no
Jacobian is needed. See the [Estimation](estimation.md) user guide for
full filter usage patterns.

!!! note "Configurable precision"
    All measurement functions and noise constructors respect
    `astrojax.set_dtype()`. Call `set_dtype()` before JIT compilation
    to control float32 vs float64 precision.
