# Earth Orientation Parameters (EOP)

The `astrojax.eop` module provides Earth Orientation Parameters for
precise reference frame transformations. EOP data corrects for
irregularities in Earth's rotation that cannot be predicted from theory
alone: polar motion, UT1-UTC offset, length-of-day variations, and
celestial pole offsets.

## Why EOP Matters

Earth's rotation axis wobbles (polar motion) and its spin rate fluctuates
(UT1-UTC, LOD). These effects are small but significant for high-fidelity
ECI/ECEF frame transformations. Without EOP corrections, frame rotation
errors grow to tens of metres for LEO spacecraft.

| Parameter | Physical Meaning | Typical Magnitude |
|-----------|-----------------|-------------------|
| PM_X, PM_Y | Pole position offset | ~0.1-0.3 arcsec |
| UT1-UTC | Earth rotation angle correction | -0.9 to +0.9 s |
| LOD | Length of day excess | ~0.1-4 ms |
| dX, dY | Celestial pole offsets (IAU 2000) | ~0.1-0.3 mas |

## JIT-Compatible Design

Unlike traditional EOP implementations that use Python dictionaries,
astrojax stores all data as sorted JAX arrays. Lookups use
`jnp.searchsorted` for O(log n) binary search with linear interpolation,
making everything compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.

## Loading EOP Data

### Bundled Data (Recommended)

The simplest approach loads the bundled `finals.all.iau2000.txt` file:

```python
from astrojax.eop import load_default_eop, get_ut1_utc

eop = load_default_eop()
ut1_utc = get_ut1_utc(eop, 59569.0)
```

### Custom Data File

Load from any IERS standard format file:

```python
from astrojax.eop import load_eop_from_file

eop = load_eop_from_file("/path/to/finals.all.iau2000.txt")
```

### Constant / Zero EOP

For testing or when EOP corrections are not needed:

```python
from astrojax.eop import zero_eop, static_eop

# No corrections
eop = zero_eop()

# Fixed known values
eop = static_eop(ut1_utc=0.1, pm_x=1e-6, pm_y=2e-6)
```

## Querying EOP Values

All query functions take an `EOPData` instance and an MJD, returning
interpolated values:

```python
from astrojax.eop import get_ut1_utc, get_pm, get_lod, get_dxdy, get_eop

# Individual queries
ut1_utc = get_ut1_utc(eop, mjd)        # UT1-UTC [seconds]
pm_x, pm_y = get_pm(eop, mjd)          # Polar motion [rad]
lod = get_lod(eop, mjd)                # LOD excess [seconds]
dx, dy = get_dxdy(eop, mjd)            # Celestial pole offsets [rad]

# All at once
pm_x, pm_y, ut1_utc, lod, dx, dy = get_eop(eop, mjd)
```

## Extrapolation Modes

When querying outside the data range, two modes are available:

```python
from astrojax.eop import EOPExtrapolation

# Hold: clamp to boundary values (default)
val = get_ut1_utc(eop, far_future_mjd, EOPExtrapolation.HOLD)

# Zero: return zero outside range
val = get_ut1_utc(eop, far_future_mjd, EOPExtrapolation.ZERO)
```

The extrapolation mode is a Python enum resolved at trace time (not a
JAX array), so changing it triggers JIT recompilation.

## Using with JIT and vmap

All query functions work inside JAX transformations:

```python
import jax

# JIT compilation
jit_query = jax.jit(get_ut1_utc)
val = jit_query(eop, 59569.0)

# Vectorized over MJD batch
import jax.numpy as jnp
mjd_batch = jnp.array([59569.0, 59570.0, 59571.0])
vals = jax.vmap(lambda m: get_ut1_utc(eop, m))(mjd_batch)
```

## Data Format

The module parses the IERS Standard (Bulletin A/B) format
(`finals.all.iau2000.txt`). This fixed-width format provides daily
EOP values with observed data for past dates and predictions for
future dates. Optional fields (LOD, dX, dY) that are unavailable in
prediction regions are stored as NaN.

!!! note "Configurable precision"
    EOP data is stored internally as float64 arrays for interpolation
    precision. If `jax_enable_x64` is not enabled, JAX silently truncates
    to float32, which is adequate for most applications. For full
    precision, call `astrojax.set_dtype(jnp.float64)` before loading
    EOP data.
