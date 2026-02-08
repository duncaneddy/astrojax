# Attitude Dynamics

The `astrojax.attitude_dynamics` module provides rigid-body attitude
dynamics for spacecraft attitude propagation.  All functions are
JAX-traceable and compatible with `jax.jit`, `jax.vmap`, and
`jax.grad`.

## Overview

| Component | Functions / Classes | Description |
|-----------|---------------------|-------------|
| **Euler Dynamics** | `quaternion_derivative`, `euler_equation` | Quaternion kinematics and Euler's rotational equation |
| **Gravity Gradient** | `torque_gravity_gradient` | Gravity gradient torque model |
| **Configuration** | `SpacecraftInertia`, `AttitudeDynamicsConfig` | Inertia tensor and dynamics presets |
| **Factory** | `create_attitude_dynamics` | Compose torque models into an integrator-compatible closure |
| **Utilities** | `normalize_attitude_state` | Quaternion renormalization after integration |

## State Vector

The attitude state is a 7-element vector combining the unit quaternion
(scalar-first) with the body-frame angular velocity:

$$
\mathbf{x} = \begin{bmatrix} q_w & q_x & q_y & q_z & \omega_x & \omega_y & \omega_z \end{bmatrix}^T
$$

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | $q_w$ | Quaternion scalar part | — |
| 1–3 | $q_x, q_y, q_z$ | Quaternion vector part | — |
| 4–6 | $\omega_x, \omega_y, \omega_z$ | Body-frame angular velocity | rad/s |

The quaternion follows the scalar-first `[w, x, y, z]` convention used
throughout astrojax.

## Core Dynamics

### Quaternion Kinematics

The quaternion time-derivative is computed from the angular velocity
using the 4×4 Omega matrix form:

$$
\dot{q} = \frac{1}{2}\,\Omega(\boldsymbol{\omega})\,q
$$

where

$$
\Omega(\boldsymbol{\omega}) = \begin{bmatrix}
0 & -\omega_x & -\omega_y & -\omega_z \\
\omega_x & 0 & \omega_z & -\omega_y \\
\omega_y & -\omega_z & 0 & \omega_x \\
\omega_z & \omega_y & -\omega_x & 0
\end{bmatrix}
$$

```python
import jax.numpy as jnp
from astrojax.attitude_dynamics import quaternion_derivative

q = jnp.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
omega = jnp.array([0.0, 0.0, 0.1])    # yaw rate [rad/s]
q_dot = quaternion_derivative(q, omega)  # shape (4,)
```

### Euler's Rotational Equation

Angular acceleration is computed from Euler's equation for a rigid body:

$$
\mathbf{I}\,\dot{\boldsymbol{\omega}}
= -\boldsymbol{\omega} \times (\mathbf{I}\,\boldsymbol{\omega})
+ \boldsymbol{\tau}
$$

The implementation uses `jnp.linalg.solve` instead of an explicit
matrix inverse for numerical stability:

```python
from astrojax.attitude_dynamics import euler_equation

omega = jnp.array([0.1, 0.0, 0.0])             # rad/s
I = jnp.diag(jnp.array([10.0, 20.0, 30.0]))    # kg m^2
tau = jnp.zeros(3)                               # no torque
omega_dot = euler_equation(omega, I, tau)         # rad/s^2
```

## Gravity Gradient Torque

The gravity gradient torque arises from the differential gravitational
acceleration across an extended rigid body.  It tends to align the axis
of minimum inertia with the nadir direction:

$$
\boldsymbol{\tau}_{gg}
= \frac{3\mu}{r^3}\left(\hat{\mathbf{r}}_b \times
(\mathbf{I}\,\hat{\mathbf{r}}_b)\right)
$$

where $\hat{\mathbf{r}}_b$ is the unit position vector expressed in the
body frame.  The quaternion `q` defines the body-to-inertial rotation.

```python
from astrojax.attitude_dynamics import torque_gravity_gradient

q = jnp.array([1.0, 0.0, 0.0, 0.0])          # body-to-ECI quaternion
r_eci = jnp.array([7000e3, 0.0, 0.0])         # ECI position [m]
I = jnp.diag(jnp.array([10.0, 20.0, 30.0]))   # kg m^2
tau_gg = torque_gravity_gradient(q, r_eci, I)   # body-frame torque [N m]
```

The gravity gradient torque vanishes for a spherically symmetric body
($I_{xx} = I_{yy} = I_{zz}$) and is strongest when the inertia
principal values differ significantly.

## Configuration

### SpacecraftInertia

Stores the 3×3 inertia tensor.  For most spacecraft, the tensor is
diagonal (principal axes aligned with body axes) and can be constructed
with the `from_principal` factory:

```python
from astrojax.attitude_dynamics import SpacecraftInertia

inertia = SpacecraftInertia.from_principal(100.0, 200.0, 300.0)
# inertia.I is a (3, 3) diagonal JAX array
```

A full 3×3 tensor can be passed directly for bodies with off-diagonal
products of inertia:

```python
I_full = jnp.array([[100.0, -5.0, 0.0],
                     [-5.0, 200.0, 0.0],
                     [0.0,   0.0, 300.0]])
inertia = SpacecraftInertia(I=I_full)
```

### AttitudeDynamicsConfig

Selects which torque models to include.  Boolean toggles are resolved at
JAX trace time, producing an optimized computation graph with no runtime
branching.

| Preset | Method | Gravity Gradient |
|--------|--------|:----------------:|
| Torque-free | `AttitudeDynamicsConfig.torque_free(inertia)` | No |
| Gravity gradient | `AttitudeDynamicsConfig.with_gravity_gradient(inertia)` | Yes |

```python
from astrojax.attitude_dynamics import AttitudeDynamicsConfig, SpacecraftInertia

inertia = SpacecraftInertia.from_principal(10.0, 20.0, 30.0)

# Torque-free rigid body
config_free = AttitudeDynamicsConfig.torque_free(inertia)

# With gravity gradient torque
config_gg = AttitudeDynamicsConfig.with_gravity_gradient(inertia)
```

## Composing Dynamics

The individual functions above can be composed into a single dynamics
function using `create_attitude_dynamics`.  This factory takes a
configuration object and a position function, and returns a
`dynamics(t, state) -> derivative` closure compatible with all astrojax
integrators.

### Quick Start: Torque-Free Propagation

```python
import jax.numpy as jnp
from astrojax.attitude_dynamics import (
    AttitudeDynamicsConfig,
    SpacecraftInertia,
    create_attitude_dynamics,
)
from astrojax.integrators import rk4_step

inertia = SpacecraftInertia.from_principal(10.0, 20.0, 30.0)
config = AttitudeDynamicsConfig.torque_free(inertia)

# Position function (required argument, not called for torque-free)
pos_fn = lambda t: jnp.array([7000e3, 0.0, 0.0])

dynamics = create_attitude_dynamics(config, pos_fn)

# Initial state: identity quaternion, 0.1 rad/s roll rate
x0 = jnp.array([1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
result = rk4_step(dynamics, 0.0, x0, 1.0)
```

### With Gravity Gradient

When gravity gradient torque is enabled, the `position_fn` provides
the spacecraft ECI position at each time step:

```python
config = AttitudeDynamicsConfig.with_gravity_gradient(inertia)

# In practice, position_fn would interpolate an orbit propagation
pos_fn = lambda t: jnp.array([7000e3, 0.0, 0.0])

dynamics = create_attitude_dynamics(config, pos_fn)
result = rk4_step(dynamics, 0.0, x0, 1.0)
```

### Multi-Step Propagation

Use `jax.lax.scan` for efficient multi-step propagation.  Apply
`normalize_attitude_state` periodically to prevent quaternion norm
drift:

```python
import jax
from astrojax.attitude_dynamics import normalize_attitude_state

dt = 1.0  # seconds

def scan_step(carry, _):
    t, state = carry
    result = rk4_step(dynamics, t, state, dt)
    # Renormalize quaternion to prevent drift
    state_normed = normalize_attitude_state(result.state)
    return (t + dt, state_normed), state_normed

(_, _), trajectory = jax.lax.scan(scan_step, (0.0, x0), None, length=1000)
# trajectory shape: (1000, 7)
```

## JAX Compatibility

All attitude dynamics functions use JAX primitives and are fully
compatible with `jax.jit` for compilation and `jax.grad` for automatic
differentiation:

```python
import jax

# JIT-compiled dynamics
jit_dynamics = jax.jit(dynamics)
x_dot = jit_dynamics(0.0, x0)

# JIT-compiled gravity gradient torque
jit_gg = jax.jit(torque_gravity_gradient)
tau = jit_gg(q, r_eci, I)
```

!!! note "Float precision"
    At float32 (default), quaternion normalization and torque
    computations are accurate to ~1e-6 relative error.  For
    reference-quality comparisons, use `set_dtype(jnp.float64)` before
    any JIT compilation.
