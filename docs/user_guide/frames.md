# Frame Transformations

The `astrojax.frames` module provides functions for transforming
satellite state vectors between common coordinate frames.

## ECI and ECEF Frames

Two fundamental reference frames in astrodynamics are:

| Frame | Full Name | Rotation |
|-------|-----------|----------|
| **ECI** | Earth-Centered Inertial | Fixed with respect to the stars |
| **ECEF** | Earth-Centered Earth-Fixed | Rotates with the Earth |

Both share the same origin (Earth's centre of mass) and z-axis
(celestial/geographic north pole), but their x and y axes differ by
the Earth's rotation angle at a given time.

## The Simplified Rotation Model

The full IAU 2006/2000A transformation between ECI and ECEF involves
three components:

$$
T = W(t) \; R(t) \; Q(t)
$$

| Component | Effect | Magnitude |
|-----------|--------|-----------|
| $Q(t)$ | Bias-precession-nutation | ~arcseconds |
| $R(t)$ | Earth rotation | 0–360° |
| $W(t)$ | Polar motion | ~arcseconds |

astrojax implements only the **Earth rotation** component — a single
$R_z(\theta_{\text{GMST}})$ rotation.  The precession-nutation and
polar motion terms contribute arcsecond-level corrections that are
below the float32 precision floor (~8 ms in time, ~100 m at LEO).

## Rotation Matrices

`rotation_eci_to_ecef` and `rotation_ecef_to_eci` compute the 3×3
direction-cosine matrix at a given epoch:

```python
from astrojax import Epoch
from astrojax.frames import rotation_eci_to_ecef, rotation_ecef_to_eci

epc = Epoch(2024, 1, 1, 12, 0, 0)

R_eci2ecef = rotation_eci_to_ecef(epc)  # 3x3
R_ecef2eci = rotation_ecef_to_eci(epc)  # transpose of the above
```

The underlying function `earth_rotation` returns the same matrix and
can be used directly if preferred.

## State Vector Transformations

For satellite state vectors $[\mathbf{r}, \mathbf{v}]$, position and
velocity must be transformed differently.  The velocity includes a
correction for the rotating frame:

$$
\mathbf{v}_{\text{ECEF}} = R \, \mathbf{v}_{\text{ECI}}
    - \boldsymbol{\omega} \times \mathbf{r}_{\text{ECEF}}
$$

where $\boldsymbol{\omega} = [0, 0, \omega_\oplus]^T$ is Earth's
angular velocity vector (7.292×10⁻⁵ rad/s).  The
$\boldsymbol{\omega} \times \mathbf{r}$ term subtracts the velocity
that an Earth-fixed observer acquires from the rotating frame — about
465 m/s at the equator.

```python
import jax.numpy as jnp
from astrojax import Epoch
from astrojax.constants import R_EARTH, GM_EARTH
from astrojax.frames import state_eci_to_ecef, state_ecef_to_eci

epc = Epoch(2024, 1, 1, 12, 0, 0)
sma = R_EARTH + 500e3
v_circ = jnp.sqrt(GM_EARTH / sma)
x_eci = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])

# Transform to ECEF
x_ecef = state_eci_to_ecef(epc, x_eci)

# Transform back to ECI (roundtrip)
x_back = state_ecef_to_eci(epc, x_ecef)
```

The inverse `state_ecef_to_eci` adds the $\boldsymbol{\omega} \times
\mathbf{r}$ term back before rotating into the inertial frame.

## JAX Compatibility

All functions use JAX primitives and work with `jax.jit` and
`jax.vmap`:

```python
import jax

# JIT compilation
jit_transform = jax.jit(state_eci_to_ecef)
x_ecef = jit_transform(epc, x_eci)

# Batch transformation over multiple states
states = jnp.stack([x_eci, x_eci + 100.0])
batched = jax.vmap(state_eci_to_ecef, in_axes=(None, 0))(epc, states)
```

!!! note "float32 precision"
    All computations use float32 for GPU/TPU compatibility.  Roundtrip
    transformations (ECI → ECEF → ECI) are accurate to ~1 m in position
    and ~1 mm/s in velocity.
