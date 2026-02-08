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

## JAX Compatibility

All functions use JAX primitives internally and work with
`jax.jit`:

```python
import jax
from astrojax.relative_motion import state_eci_to_rtn

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
