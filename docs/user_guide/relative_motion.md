# Relative Motion

The `astrojax.relative_motion` module provides functions for analysing
satellite proximity operations: transforming between inertial and
rotating local frames, and evaluating linearised relative dynamics.

## The RTN Frame

When studying two satellites flying in formation, it is convenient to
describe the deputy's position and velocity relative to the chief in
a co-moving frame attached to the chief.  The **RTN** (Radial,
Along-Track, Normal) frame — also called the **LVLH** (Local Vertical
Local Horizontal) frame — is defined as:

| Axis | Direction |
|------|-----------|
| **R** (Radial) | From Earth's centre toward the chief's position |
| **T** (Along-track) | In the orbital plane, perpendicular to R, in the direction of motion |
| **N** (Normal / Cross-track) | Along the orbital angular-momentum vector, completing the right-handed triad |

The rotation matrix $R_{\text{RTN} \to \text{ECI}}$ has columns
$[\hat{r} \;\; \hat{t} \;\; \hat{n}]$, where

$$
\hat{r} = \frac{\mathbf{r}}{|\mathbf{r}|}, \quad
\hat{n} = \frac{\mathbf{r} \times \mathbf{v}}{|\mathbf{r} \times \mathbf{v}|}, \quad
\hat{t} = \hat{n} \times \hat{r}.
$$

## Rotation Matrices

`rotation_rtn_to_eci` and `rotation_eci_to_rtn` compute the 3x3
direction-cosine matrix for a given chief ECI state:

```python
import jax.numpy as jnp
from astrojax.constants import R_EARTH, GM_EARTH
from astrojax.relative_motion import rotation_rtn_to_eci, rotation_eci_to_rtn

sma = R_EARTH + 500e3
v_circ = jnp.sqrt(GM_EARTH / sma)
chief = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])

R_rtn2eci = rotation_rtn_to_eci(chief)  # 3x3
R_eci2rtn = rotation_eci_to_rtn(chief)  # transpose of the above
```

## ECI to RTN State Transformation

`state_eci_to_rtn` transforms the absolute ECI states of a chief and
deputy into the deputy's relative state in the chief's RTN frame.  It
accounts for the Coriolis effect caused by the rotating frame:

$$
\dot{\boldsymbol{\rho}}_{\text{RTN}}
= R_{\text{ECI} \to \text{RTN}} \,
  (\mathbf{v}_{\text{dep}} - \mathbf{v}_{\text{chief}})
  - \boldsymbol{\omega} \times \boldsymbol{\rho}_{\text{RTN}}
$$

where $\boldsymbol{\omega} = [0, 0, \dot{f}]^T$ and the true-anomaly
rate is $\dot{f} = |\mathbf{r} \times \mathbf{v}| \;/\; |\mathbf{r}|^2$.

```python
from astrojax.relative_motion import state_eci_to_rtn, state_rtn_to_eci

deputy = chief + jnp.array([100.0, 200.0, 0.0, 0.0, 0.0, 0.0])
rel_rtn = state_eci_to_rtn(chief, deputy)

# Reconstruct the deputy's absolute ECI state
deputy_back = state_rtn_to_eci(chief, rel_rtn)
```

The inverse, `state_rtn_to_eci`, recovers the deputy's absolute ECI
state from the chief state and the relative RTN state.

## Hill-Clohessy-Wiltshire Dynamics

For a chief satellite on a *circular* orbit with mean motion
$n = \sqrt{\mu / a^3}$, the linearised equations of relative motion
(HCW model) are:

$$
\ddot{x} = 3n^2 x + 2n\dot{y}, \quad
\ddot{y} = -2n\dot{x}, \quad
\ddot{z} = -n^2 z.
$$

`hcw_derivative` returns the 6-element state derivative
$[\dot{x}, \dot{y}, \dot{z}, \ddot{x}, \ddot{y}, \ddot{z}]$ and is
designed to plug directly into a numerical integrator:

```python
from astrojax.relative_motion import hcw_derivative

sma = R_EARTH + 500e3
n = jnp.sqrt(GM_EARTH / sma**3)
state = jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])

deriv = hcw_derivative(state, n)
```

Because it is a pure JAX function, `hcw_derivative` is compatible with
`jax.jit`, `jax.vmap`, and `jax.grad`, making it suitable for batched
simulation and differentiable control.

## JAX Compatibility

All five functions use JAX primitives internally and work with
`jax.jit`:

```python
import jax

jit_rtn = jax.jit(state_eci_to_rtn)
rel = jit_rtn(chief, deputy)
```

`hcw_derivative` and `rotation_rtn_to_eci` also support `jax.vmap`
for batched evaluation and `jax.grad` for gradient computation.

!!! note "float32 precision"
    All computations use float32 for GPU/TPU compatibility.  Rotation
    matrix operations on position magnitudes of ~7000 km introduce
    rounding at the metre level.  Roundtrip transformations
    (ECI -> RTN -> ECI) are accurate to ~0.1 mm.
