# Estimation

The `astrojax.estimation` module provides building-block functions for
sequential state estimation using Kalman filters. It includes an
Extended Kalman Filter (EKF) and an Unscented Kalman Filter (UKF).
Measurement models for orbit determination are in the separate
`astrojax.orbit_measurements` module.

## Purpose and When to Use

Use this module when you need to estimate a spacecraft's state (position,
velocity) from noisy sensor measurements. The filters combine a dynamics
model (prediction) with sensor data (correction) to produce optimal state
estimates.

| Scenario | Recommended Filter |
|----------|-------------------|
| Differentiable dynamics, moderate nonlinearity | EKF |
| Highly nonlinear dynamics or measurement models | UKF |
| Need gradients through the filter | EKF |
| No analytical Jacobians available | Either (both use autodiff/sigma points) |

## Available Components

### Estimation (`astrojax.estimation`)

| Component | Description |
|-----------|-------------|
| `FilterState` | State estimate and covariance matrix |
| `UKFConfig` | UKF sigma point tuning parameters |
| `FilterResult` | Update result with innovation diagnostics |
| `ekf_predict` | EKF state propagation (autodiff STM) |
| `ekf_update` | EKF measurement incorporation (Joseph form) |
| `ukf_predict` | UKF state propagation (sigma point transform) |
| `ukf_update` | UKF measurement incorporation (sigma point transform) |

### Orbit Measurements (`astrojax.orbit_measurements`)

| Component | Description |
|-----------|-------------|
| `gnss_position_measurement` | Position-only GNSS measurement model |
| `gnss_measurement_noise` | Position-only noise covariance constructor |
| `gnss_position_velocity_measurement` | Position-velocity GNSS measurement model |
| `gnss_position_velocity_noise` | Position-velocity noise covariance constructor |

## Design Philosophy

The filters are **building blocks**, not monolithic runners. You compose
`predict` and `update` calls yourself, typically inside `jax.lax.scan`
for sequential processing. This gives you full control over:

- When to predict vs. update
- How to handle missing measurements
- Custom logging or divergence detection
- Mixing different measurement types

## Basic Usage: EKF

```python
import jax.numpy as jnp
from astrojax.estimation import FilterState, ekf_predict, ekf_update
from astrojax.orbit_measurements import gnss_position_measurement, gnss_measurement_noise

# Initial state and covariance
x0 = jnp.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
P0 = jnp.diag(jnp.array([1e6, 1e6, 1e6, 1e2, 1e2, 1e2]))
fs = FilterState(x=x0, P=P0)

# Process and measurement noise
Q = jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]))
R = gnss_measurement_noise(10.0)  # 10 m 1-sigma

# Define propagation (user closes over dynamics, integrator, timestep)
from astrojax import create_orbit_dynamics, Epoch
from astrojax.eop import zero_eop
from astrojax.integrators import rk4_step

epoch_0 = Epoch(2024, 6, 15, 12, 0, 0)
dynamics = create_orbit_dynamics(zero_eop(), epoch_0)

def propagate(x):
    return rk4_step(dynamics, 0.0, x, 10.0).state

# Predict
fs = ekf_predict(fs, propagate, Q)

# Update with GNSS measurement
z = jnp.array([6878e3 + 5.0, 3.0, -2.0])  # noisy position
result = ekf_update(fs, z, gnss_position_measurement, R)
fs = result.state  # updated filter state
```

## Basic Usage: UKF

The UKF has the same interface, replacing `ekf_` with `ukf_`:

```python
from astrojax.estimation import ukf_predict, ukf_update, UKFConfig

# Optional: customize sigma point spread
config = UKFConfig(alpha=1.0, beta=2.0, kappa=0.0)

fs = ukf_predict(fs, propagate, Q, config=config)
result = ukf_update(fs, z, gnss_position_measurement, R, config=config)
```

## Sequential Filtering with `jax.lax.scan`

The canonical pattern for processing a sequence of measurements:

```python
import jax

def filter_step(fs, z):
    fs = ekf_predict(fs, propagate, Q)
    result = ekf_update(fs, z, gnss_position_measurement, R)
    return result.state, result.innovation

# measurements: (n_steps, 3) array of GNSS position observations
fs0 = FilterState(x=x0, P=P0)
final_state, innovations = jax.lax.scan(filter_step, fs0, measurements)
```

This compiles the entire filtering loop into a single XLA program,
giving significant speedups over Python-level loops.

## Custom Measurement Functions

Any function `h(x) -> z` that maps the state to an observation can be
used as a measurement function. The EKF will autodiff through it; the
UKF will evaluate it at sigma points:

```python
def range_measurement(state):
    """Range from origin to spacecraft."""
    r = state[:3]
    return jnp.array([jnp.linalg.norm(r)])

# Use with either filter
result = ekf_update(fs, z_range, range_measurement, R_range)
```

## Filter Diagnostics

The `FilterResult` returned by update functions includes diagnostic
fields for monitoring filter health:

- **`innovation`**: Should be zero-mean if the filter is consistent
- **`innovation_covariance`**: The innovation normalized by this matrix
  should follow a chi-squared distribution
- **`kalman_gain`**: Useful for analyzing filter sensitivity

```python
result = ekf_update(fs, z, gnss_position_measurement, R)

# Check innovation magnitude
innov_norm = jnp.linalg.norm(result.innovation)

# Normalized Innovation Squared (NIS) - should be ~chi2(m)
S_inv = jnp.linalg.inv(result.innovation_covariance)
nis = result.innovation @ S_inv @ result.innovation
```

## Key Differences: EKF vs UKF

| Feature | EKF | UKF |
|---------|-----|-----|
| Jacobian computation | `jax.jacfwd` (autodiff) | Not needed |
| Nonlinearity handling | First-order linearization | Sigma point sampling |
| Covariance accuracy | Second-order for linear | Third-order for Gaussian |
| Computational cost | Lower (one propagation + Jacobian) | Higher (2n+1 propagations) |
| Gradient support | Full `jax.grad` support | Limited by Cholesky |

!!! note "Configurable precision"
    All estimation functions respect `astrojax.set_dtype()`. Call
    `set_dtype()` before JIT compilation to control float32 vs float64
    precision. The UKF's Cholesky decomposition includes dtype-adaptive
    regularization for float32 stability.

!!! warning "Propagation function requirements"
    The `propagate_fn` passed to `ekf_predict` must be differentiable
    by JAX (composed of JAX operations). The `propagate_fn` for
    `ukf_predict` only needs to be vmappable. Both should map a state
    vector to a propagated state vector with the same shape.
