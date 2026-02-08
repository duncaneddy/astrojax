# Integrators

The `astrojax.integrators` module provides numerical ODE integrators for
propagating orbital dynamics forward (or backward) in time. All
integrators are implemented in pure JAX and are compatible with
`jax.jit` and `jax.vmap`.

## Available Integrators

| Integrator | Order | Step Control | Stages | Use Case |
|------------|-------|-------------|--------|----------|
| `rk4_step` | 4 | Fixed | 4 | Simple propagation, differentiable control |
| `rkf45_step` | 4(5) | Adaptive | 6 | General-purpose adaptive integration |
| `dp54_step` | 5(4) | Adaptive | 7 | High-accuracy adaptive integration |

## Common Interface

All step functions share the same calling convention:

```python
result = step_fn(dynamics, t, state, dt)
```

- **`dynamics(t, x) -> dx`**: The ODE right-hand side.
- **`t`**: Current time (scalar).
- **`state`**: Current state vector (1-D array).
- **`dt`**: Timestep to take (may be negative for backward integration).

The result is a `StepResult` named tuple with fields `state`, `dt_used`,
`error_estimate`, and `dt_next`.

## RK4: Fixed-Step Integration

The classic 4th-order Runge-Kutta method takes exactly the step size
you request. It is the simplest integrator and supports reverse-mode
differentiation (`jax.grad`), making it suitable for differentiable
control and optimization:

```python
import jax
import jax.numpy as jnp
from astrojax.integrators import rk4_step

def harmonic(t, x):
    return jnp.array([x[1], -x[0]])

# Single step
result = rk4_step(harmonic, 0.0, jnp.array([1.0, 0.0]), 0.01)
print(result.state)  # ~[cos(0.01), -sin(0.01)]

# Multi-step propagation with lax.scan
def scan_step(state, _):
    result = rk4_step(harmonic, 0.0, state, 0.01)
    return result.state, result.state

final, trajectory = jax.lax.scan(scan_step, jnp.array([1.0, 0.0]), None, length=100)
```

## Adaptive Methods: RKF45 and DP54

The adaptive integrators automatically adjust the step size to keep the
local error within configurable tolerances. If a step produces an error
above the tolerance, it is rejected and retried with a smaller step.

```python
from astrojax.integrators import rkf45_step, dp54_step, AdaptiveConfig

def two_body(t, state):
    r = state[:3]
    v = state[3:]
    r_norm = jnp.linalg.norm(r)
    a = -398600.4418 * r / r_norm**3
    return jnp.concatenate([v, a])

# Use default tolerances
result = rkf45_step(two_body, 0.0, state0, 60.0)

# Custom tolerances for higher accuracy
config = AdaptiveConfig(abs_tol=1e-10, rel_tol=1e-8)
result = dp54_step(two_body, 0.0, state0, 60.0, config=config)

# The result tells you what happened
print(f"Step taken: {result.dt_used}")
print(f"Error: {result.error_estimate}")
print(f"Suggested next dt: {result.dt_next}")
```

## Adaptive Configuration

The `AdaptiveConfig` named tuple controls step-size adaptation:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `abs_tol` | `1e-6` | Absolute error tolerance per component |
| `rel_tol` | `1e-3` | Relative error tolerance per component |
| `safety_factor` | `0.9` | Conservative scaling of step predictions |
| `min_scale_factor` | `0.2` | Maximum step shrinkage ratio |
| `max_scale_factor` | `10.0` | Maximum step growth ratio |
| `min_step` | `1e-12` | Absolute minimum step size |
| `max_step` | `900.0` | Absolute maximum step size (seconds) |
| `max_step_attempts` | `10` | Maximum retries per step |

The per-component error tolerance is:

$$
\text{tol}_i = \text{abs\_tol} + \text{rel\_tol} \cdot \max(|y_i^{\text{new}}|, |y_i^{\text{old}}|)
$$

A step is accepted when the normalized error (infinity norm) is $\leq 1.0$.

## Control Inputs

All integrators support an optional additive control function. This is
useful for thrust manoeuvres, perturbation forces, or feedback control:

```python
def gravity(t, x):
    r = x[:3]
    r_norm = jnp.linalg.norm(r)
    a = -398600.4418 * r / r_norm**3
    return jnp.concatenate([x[3:], a])

def thrust(t, x):
    # Constant along-track thrust of 1 mm/s^2
    v = x[3:]
    v_hat = v / jnp.linalg.norm(v)
    return jnp.concatenate([jnp.zeros(3), 1e-3 * v_hat])

result = rk4_step(gravity, 0.0, state0, 60.0, control=thrust)
```

The effective derivative at each stage is `dynamics(t, x) + control(t, x)`.

## Backward Integration

All integrators support backward integration by passing a negative `dt`:

```python
# Propagate backward 60 seconds
result = rk4_step(dynamics, t_final, state_final, -60.0)
```

## JAX Compatibility

All integrators work with `jax.jit` for compilation:

```python
jit_step = jax.jit(lambda t, x, dt: rk4_step(dynamics, t, x, dt))
result = jit_step(0.0, state0, 60.0)
```

RK4 also supports `jax.grad` for differentiable simulation. The adaptive
methods (RKF45, DP54) use `jax.lax.while_loop` internally, which supports
forward-mode differentiation but **not** reverse-mode `jax.grad`.

!!! note "Configurable precision"
    All integrators respect `astrojax.set_dtype()`. Call `set_dtype()`
    before JIT compilation to control whether computations use float32
    or float64 precision.
