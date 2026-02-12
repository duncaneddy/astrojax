# Frame Transformations

The `astrojax.frames` module provides functions for transforming
satellite state vectors between common coordinate frames using the
full IAU 2006/2000A CIO-based model.

## Reference Frames

Two fundamental reference frames in astrodynamics are:

| Frame | Full Name | Rotation |
|-------|-----------|----------|
| **GCRF** (ECI) | Geocentric Celestial Reference Frame | Fixed with respect to the stars |
| **ITRF** (ECEF) | International Terrestrial Reference Frame | Rotates with the Earth |

Both share the same origin (Earth's centre of mass), but their axes
differ due to Earth's rotation, precession, nutation, and polar motion.

## The IAU 2006/2000A Model

The full transformation between GCRF and ITRF involves three components:

$$
R_{\text{GCRF} \to \text{ITRF}} = W(t) \; R(t) \; Q(t)
$$

| Component | Effect | Model |
|-----------|--------|-------|
| $Q(t)$ | Bias-precession-nutation (GCRF -> CIRS) | IAU 2006/2000A |
| $R(t)$ | Earth rotation (CIRS -> TIRS) | IAU 2000 ERA |
| $W(t)$ | Polar motion (TIRS -> ITRF) | IERS conventions |

All functions require an `eop: EOPData` parameter providing Earth
Orientation Parameters (polar motion, UT1-UTC, celestial pole offsets).
Use `zero_eop()` for a quick approximation or `static_eop(...)` for
known EOP values.

## Rotation Matrices

`rotation_gcrf_to_itrf` and `rotation_itrf_to_gcrf` compute the 3x3
direction-cosine matrix at a given epoch:

```python
from astrojax import Epoch
from astrojax.eop import zero_eop
from astrojax.frames import rotation_gcrf_to_itrf, rotation_itrf_to_gcrf

eop = zero_eop()
epc = Epoch(2024, 1, 1, 12, 0, 0)

R_gcrf2itrf = rotation_gcrf_to_itrf(eop, epc)  # 3x3
R_itrf2gcrf = rotation_itrf_to_gcrf(eop, epc)  # transpose of the above
```

ECI/ECEF aliases (`rotation_eci_to_ecef`, `rotation_ecef_to_eci`) are
provided for backward compatibility.

## State Vector Transformations

For satellite state vectors $[\mathbf{r}, \mathbf{v}]$, position and
velocity must be transformed differently.  The velocity includes a
correction for the rotating frame:

$$
\mathbf{v}_{\text{ITRF}} = W \left(
    R \, Q \, \mathbf{v}_{\text{GCRF}}
    - \boldsymbol{\omega} \times (R \, Q \, \mathbf{r}_{\text{GCRF}})
\right)
$$

where $\boldsymbol{\omega} = [0, 0, \omega_\oplus]^T$ is Earth's
angular velocity vector (7.292x10^-5 rad/s).

```python
import jax.numpy as jnp
from astrojax import Epoch
from astrojax.eop import zero_eop
from astrojax.constants import R_EARTH, GM_EARTH
from astrojax.frames import state_gcrf_to_itrf, state_itrf_to_gcrf

eop = zero_eop()
epc = Epoch(2024, 1, 1, 12, 0, 0)
sma = R_EARTH + 500e3
v_circ = jnp.sqrt(GM_EARTH / sma)
x_gcrf = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])

# Transform to ITRF
x_itrf = state_gcrf_to_itrf(eop, epc, x_gcrf)

# Transform back to GCRF (roundtrip)
x_back = state_itrf_to_gcrf(eop, epc, x_itrf)
```

## JAX Compatibility

All functions use JAX primitives and work with `jax.jit` and
`jax.vmap`:

```python
import jax

# JIT compilation
jit_transform = jax.jit(state_gcrf_to_itrf)
x_itrf = jit_transform(eop, epc, x_gcrf)

# Batch transformation over multiple states
states = jnp.stack([x_gcrf, x_gcrf + 100.0])
batched = jax.vmap(state_gcrf_to_itrf, in_axes=(None, None, 0))(eop, epc, states)
```
