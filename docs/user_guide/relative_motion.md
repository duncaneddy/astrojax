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

## Relative Orbital Elements (ROE)

An alternative to describing relative motion in the RTN frame is to
use **Quasi-Nonsingular Relative Orbital Elements** (ROE).  While RTN
coordinates oscillate rapidly with orbital position, ROE components
vary slowly (secular and long-period only), making them well-suited
for formation design and station-keeping.

The ROE vector $\delta\boldsymbol{\alpha} = [da,\; d\lambda,\; de_x,\; de_y,\; di_x,\; di_y]$
is defined as:

| Component | Definition | Units |
|-----------|-----------|-------|
| $da$ | $(a_d - a_c) / a_c$ | dimensionless |
| $d\lambda$ | $(u_d - u_c) + (\Omega_d - \Omega_c) \cos i_c$ | rad (or deg) |
| $de_x$ | $e_d \cos\omega_d - e_c \cos\omega_c$ | dimensionless |
| $de_y$ | $e_d \sin\omega_d - e_c \sin\omega_c$ | dimensionless |
| $di_x$ | $i_d - i_c$ | rad (or deg) |
| $di_y$ | $(\Omega_d - \Omega_c) \sin i_c$ | rad (or deg) |

where $u = M + \omega$ is the argument of latitude.

### OE to ROE

`state_oe_to_roe` computes the ROE from chief and deputy Keplerian
elements:

```python
import jax.numpy as jnp
from astrojax.constants import R_EARTH
from astrojax.relative_motion import state_oe_to_roe

oe_chief = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
oe_deputy = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

roe = state_oe_to_roe(oe_chief, oe_deputy, use_degrees=True)
```

The inverse, `state_roe_to_oe`, recovers the deputy's orbital elements
from the chief elements and the ROE:

```python
from astrojax.relative_motion import state_roe_to_oe

oe_deputy_recovered = state_roe_to_oe(oe_chief, roe, use_degrees=True)
```

### ECI to ROE

For convenience, `state_eci_to_roe` and `state_roe_to_eci` compose the
ECI$\leftrightarrow$KOE and OE$\leftrightarrow$ROE transformations,
allowing direct conversion between ECI state vectors and ROE:

```python
from astrojax.coordinates import state_koe_to_eci
from astrojax.relative_motion import state_eci_to_roe, state_roe_to_eci

x_chief = state_koe_to_eci(oe_chief, use_degrees=True)
x_deputy = state_koe_to_eci(oe_deputy, use_degrees=True)

roe = state_eci_to_roe(x_chief, x_deputy, use_degrees=True)
x_deputy_recovered = state_roe_to_eci(x_chief, roe, use_degrees=True)
```

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

All functions use JAX primitives internally and work with
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
