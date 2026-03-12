# Covariance Propagation

The `astrojax.covariance` module provides generic covariance propagation
via variational equations. It augments any differentiable ODE dynamics
function with automatic State Transition Matrix (STM) propagation using
`jax.jacfwd` — no hand-derived Jacobians needed.

## Purpose and When to Use

Use covariance propagation when you need to track how state uncertainty
evolves over time. Common scenarios:

| Scenario | Why Covariance Propagation Helps |
|----------|----------------------------------|
| Orbit uncertainty growth | Predict how position/velocity errors grow over time |
| Sensor scheduling | Determine when uncertainty exceeds a threshold |
| Initial orbit determination | Assess solution quality via covariance realism |
| Filter initialization | Provide an initial P₀ for the EKF/UKF |

## Workflow Overview

Covariance propagation follows a four-step workflow:

1. **Create** — `create_variational_dynamics(dynamics, n)` augments your dynamics with STM equations
2. **Initialize** — `augmented_initial_state(x0, n)` builds `[x₀, vec(I)]`
3. **Integrate** — Use any integrator (RK4, RKF45, etc.) on the augmented state
4. **Extract and map** — `extract_state_and_stm` splits the result, `propagate_covariance` computes `P = Φ P₀ Φᵀ + Q`

## Available Functions

| Function | Description |
|----------|-------------|
| `create_variational_dynamics` | Augment dynamics with STM propagation (auto-Jacobian) |
| `augmented_initial_state` | Build initial `[x₀, vec(I_n)]` augmented state |
| `extract_state_and_stm` | Split augmented state into `(x, Φ)` |
| `propagate_covariance` | Compute `P = Φ P₀ Φᵀ + Q` |

## Basic Example: Harmonic Oscillator

A simple 2-D system to illustrate the workflow:

```python
import jax.numpy as jnp
from astrojax.covariance import (
    create_variational_dynamics,
    augmented_initial_state,
    extract_state_and_stm,
    propagate_covariance,
)
from astrojax.integrators import rk4_step

# 1. Define dynamics: x'' = -x
def harmonic(t, x):
    return jnp.array([x[1], -x[0]])

# 2. Augment with variational equations
n = 2
aug_dynamics = create_variational_dynamics(harmonic, n)

# 3. Build initial augmented state
x0 = jnp.array([1.0, 0.0])
aug_x0 = augmented_initial_state(x0, n)
# aug_x0 has shape (6,): [x0, x1, Phi_00, Phi_01, Phi_10, Phi_11]

# 4. Integrate (single step for illustration)
result = rk4_step(aug_dynamics, 0.0, aug_x0, 0.1)

# 5. Extract state and STM
x, Phi = extract_state_and_stm(result.state, n)
print(f"State: {x}")  # propagated [position, velocity]
print(f"STM:\n{Phi}")  # 2x2 state transition matrix

# 6. Map a covariance forward
P0 = jnp.diag(jnp.array([0.01, 0.001]))
P = propagate_covariance(Phi, P0)
print(f"Propagated covariance:\n{P}")
```

## Orbit Dynamics Example: LEO Two-Body

A more realistic example propagating a 500 km circular LEO orbit
covariance over one orbital period:

```python
import jax
import jax.numpy as jnp
from astrojax import set_dtype, Epoch
from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.covariance import (
    create_variational_dynamics,
    augmented_initial_state,
    extract_state_and_stm,
    propagate_covariance,
)
from astrojax.eop import zero_eop
from astrojax.integrators import rk4_step
from astrojax.orbit_dynamics.factory import create_orbit_dynamics
from astrojax.orbits import orbital_period

set_dtype(jnp.float64)

# Circular LEO at 500 km
a = R_EARTH + 500e3
v_circ = jnp.sqrt(GM_EARTH / a)
x0 = jnp.array([a, 0.0, 0.0, 0.0, v_circ, 0.0])

# Initial covariance: 100 m position, 0.1 m/s velocity (1-sigma)
P0 = jnp.diag(jnp.array([100.0**2, 100.0**2, 100.0**2,
                           0.1**2, 0.1**2, 0.1**2]))

# Create dynamics and augment with variational equations
epoch_0 = Epoch(2026, 3, 11, 0, 0, 0.0)
dynamics = create_orbit_dynamics(zero_eop(), epoch_0)
n = 6
aug_dynamics = create_variational_dynamics(dynamics, n)

# Build augmented initial state
aug_x0 = augmented_initial_state(x0, n)  # shape (42,): 6 state + 36 STM

# Propagate one full orbit with lax.scan
T_orbit = orbital_period(a)
dt = 10.0
n_steps = int(float(T_orbit) / float(dt))

def scan_step(carry, _):
    t, aug_state = carry
    result = rk4_step(aug_dynamics, t, aug_state, dt)
    return (t + dt, result.state), result.state

init = (jnp.float64(0.0), aug_x0)
(t_final, aug_final), aug_history = jax.lax.scan(
    scan_step, init, None, length=n_steps
)

# Extract final STM and propagate covariance
x_final, Phi = extract_state_and_stm(aug_final, n)
P_final = propagate_covariance(Phi, P0)

# 1-sigma uncertainties
sigma = jnp.sqrt(jnp.diag(P_final))
print(f"Final 1-sigma position: {sigma[:3]} m")
print(f"Final 1-sigma velocity: {sigma[3:]} m/s")
print(f"STM determinant: {jnp.linalg.det(Phi):.6f}")  # ≈ 1.0 for Hamiltonian
```

## Multi-Step Covariance with `jax.lax.scan`

You can record the covariance at every timestep by extracting the STM
at each step inside the scan body:

```python
def scan_step_with_cov(carry, _):
    t, aug_state = carry
    result = rk4_step(aug_dynamics, t, aug_state, dt)

    # Extract STM and map covariance at this timestep
    _, Phi_i = extract_state_and_stm(result.state, n)
    P_i = propagate_covariance(Phi_i, P0)
    sigma_i = jnp.sqrt(jnp.diag(P_i))

    return (t + dt, result.state), sigma_i

_, sigma_history = jax.lax.scan(
    scan_step_with_cov, init, None, length=n_steps
)
# sigma_history: (n_steps, 6) — 1-sigma at each timestep
```

## JAX Compatibility

All covariance functions are pure JAX and work with:

- **`jax.jit`** — The augmented dynamics compile to a single XLA program
- **`jax.vmap`** — Vectorize over batches of initial states or covariances
- **`jax.lax.scan`** — Efficient multi-step propagation (see examples above)

```python
# Batch covariance propagation over multiple initial covariances
batch_propagate = jax.vmap(propagate_covariance, in_axes=(None, 0))
P0_batch = jnp.stack([P0 * scale for scale in [0.5, 1.0, 2.0]])
P_batch = batch_propagate(Phi, P0_batch)  # (3, 6, 6)
```

## Relationship to the Estimation Module

The covariance module is a **building block** that the estimation module
uses internally:

- `ekf_predict` uses `jax.jacfwd` to compute the STM and then applies
  the same `Φ P Φᵀ + Q` update
- The covariance module exposes this machinery directly so you can use
  it for open-loop uncertainty analysis without measurements

Use the covariance module when you want to study uncertainty growth
without sensor updates. Use the estimation module (EKF/UKF) when you
have measurements to incorporate.

!!! note "Configurable precision"
    All covariance functions respect `astrojax.set_dtype()`. Call
    `set_dtype()` before JIT compilation to control float32 vs float64
    precision. Float64 is recommended for long-duration propagation where
    STM elements can grow large.

!!! warning "Dynamics must be JAX-differentiable"
    The dynamics function passed to `create_variational_dynamics` must
    be composed of JAX operations, since `jax.jacfwd` is used internally
    to compute the Jacobian `A = ∂f/∂x`. This is the same requirement
    as `ekf_predict`.
