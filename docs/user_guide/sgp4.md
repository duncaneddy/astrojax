# SGP4/SDP4 Propagation

The `astrojax.sgp4` module provides a JAX-native implementation of the
SGP4 (Simplified General Perturbations 4) and SDP4 (Simplified Deep-space
Perturbations 4) orbit propagators for propagating Two-Line Element (TLE)
sets. All functions are compatible with `jax.jit`, `jax.vmap`, and
autodifferentiation.

## Quick Start

```python
import jax.numpy as jnp
from astrojax.sgp4 import parse_tle, sgp4_init, sgp4_propagate

line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

elements = parse_tle(line1, line2)
params, method = sgp4_init(elements)

# Propagate 60 minutes from epoch
r, v = sgp4_propagate(params, jnp.float64(60.0), method)
# r: position [km] in TEME frame, shape (3,)
# v: velocity [km/s] in TEME frame, shape (3,)
```

## Near-Earth vs Deep-Space

SGP4 automatically classifies satellites into two regimes based on
their orbital period:

| Regime | Period | Method flag | Propagation model |
|--------|--------|-------------|-------------------|
| Near-Earth | < 225 min | `'n'` | SGP4 (atmospheric drag, secular/periodic perturbations) |
| Deep-Space | ≥ 225 min | `'d'` | SDP4 (lunar-solar perturbations, resonance effects) |

The `method` string returned by `sgp4_init` tells you which regime
the satellite falls into. You pass it to `sgp4_propagate` so that JAX
traces only the relevant code path (zero overhead from the unused branch).

## Propagation Variants

The module provides two propagation variants that differ in how
deep-space resonance integration is performed. **For near-earth
satellites, both variants behave identically.**

### Default: `sgp4_propagate`

Uses `jax.lax.scan` internally for deep-space resonance integration.
This supports **both** forward-mode and reverse-mode autodifferentiation,
but imposes a configurable upper bound on propagation time.

```python
from astrojax.sgp4 import sgp4_propagate

# Default: up to ~100 days from epoch (200 iterations × 720 min)
r, v = sgp4_propagate(params, jnp.float64(tsince), method)

# Extend to ~200 days by increasing max_dspace_iters
r, v = sgp4_propagate(params, jnp.float64(tsince), method,
                       max_dspace_iters=400)
```

### Unbounded: `sgp4_propagate_unbounded`

Uses `jax.lax.while_loop` internally for deep-space resonance
integration. This imposes **no upper bound** on propagation time,
but only supports forward-mode autodifferentiation.

```python
from astrojax.sgp4 import sgp4_propagate_unbounded

# No time limit — works for any tsince value
r, v = sgp4_propagate_unbounded(params, jnp.float64(tsince), method)
```

### Choosing a Variant

| | `sgp4_propagate` | `sgp4_propagate_unbounded` |
|---|---|---|
| **Deep-space loop** | `jax.lax.scan` (fixed iterations) | `jax.lax.while_loop` (dynamic) |
| **Reverse-mode AD** (`jax.grad`, `jax.vjp`) | Yes | **No** |
| **Forward-mode AD** (`jax.jacfwd`, `jax.jvp`) | Yes | Yes |
| **Time limit (deep-space)** | `max_dspace_iters × 720` min (~100 days default) | Unlimited |
| **Near-earth behavior** | Identical | Identical |

**Use `sgp4_propagate` (default) when:**

- You need reverse-mode gradients (e.g., gradient-based orbit
  determination, optimization, training)
- Your propagation horizon is within the iteration budget
  (adjustable via `max_dspace_iters`)

**Use `sgp4_propagate_unbounded` when:**

- You need to propagate deep-space satellites over long time spans
  (months to years)
- You only need forward-mode differentiation, or no differentiation
- You don't want to worry about tuning `max_dspace_iters`

!!! warning "Reverse-mode AD and `sgp4_propagate_unbounded`"
    The unbounded variant uses `jax.lax.while_loop`, which does not
    support reverse-mode differentiation in JAX. Calling `jax.grad`
    or `jax.vjp` through `sgp4_propagate_unbounded` with a deep-space
    satellite will raise an error.

### Unified Variants

Both propagation functions have "unified" counterparts that auto-detect
the satellite regime from the parameter array (no `method` flag needed).
These use `jax.lax.cond` internally, enabling `jax.vmap` over mixed
batches of near-earth and deep-space satellites:

```python
from astrojax.sgp4 import (
    sgp4_init_jax,
    sgp4_propagate_unified,
    sgp4_propagate_unified_unbounded,
    elements_to_array,
)

arr = elements_to_array(elements)
params = sgp4_init_jax(arr)

# Auto-detects near-earth vs deep-space from params
r, v = sgp4_propagate_unified(params, jnp.float64(60.0))
r, v = sgp4_propagate_unified_unbounded(params, jnp.float64(60.0))
```

## TLE Input Formats

The module accepts TLEs in several formats:

### From TLE Strings

```python
from astrojax.sgp4 import create_sgp4_propagator

params, propagate_fn = create_sgp4_propagator(line1, line2)
r, v = propagate_fn(jnp.float64(60.0))
```

### From OMM Fields

```python
from astrojax.sgp4 import create_sgp4_propagator_from_omm

fields = {
    "OBJECT_ID": "1998-067A",
    "EPOCH": "2008-09-20T12:25:40.104192",
    "MEAN_MOTION": "15.72125391",
    "ECCENTRICITY": "0.0006703",
    "INCLINATION": "51.6416",
    "RA_OF_ASC_NODE": "247.4627",
    "ARG_OF_PERICENTER": "130.5360",
    "MEAN_ANOMALY": "325.0288",
    "BSTAR": "-0.11606e-4",
    "MEAN_MOTION_DOT": "-0.00002182",
    "MEAN_MOTION_DDOT": "0",
}
params, propagate_fn = create_sgp4_propagator_from_omm(fields)
```

### From GPRecord

```python
from astrojax.sgp4 import create_sgp4_propagator_from_gp_record

params, propagate_fn = create_sgp4_propagator_from_gp_record(record)
```

## JIT-Compilable Initialization

For workflows that need to differentiate or `vmap` through initialization
(e.g., optimizing orbital elements), use `sgp4_init_jax`:

```python
import jax
from astrojax.sgp4 import sgp4_init_jax, sgp4_propagate_unified, elements_to_array

arr = elements_to_array(elements)
params = sgp4_init_jax(arr, gravity=WGS72, opsmode="i")

# Differentiate through init + propagation
def loss(elems):
    p = sgp4_init_jax(elems, gravity=WGS72, opsmode="i")
    r, v = sgp4_propagate_unified(p, jnp.float64(60.0))
    return jnp.sum(r**2)

grads = jax.grad(loss)(arr)
```

## Batch Propagation with vmap

### Over Time

```python
times = jnp.array([0.0, 60.0, 360.0, 1440.0])
r_batch, v_batch = jax.vmap(
    lambda t: sgp4_propagate(params, t, method)
)(times)
# r_batch: shape (4, 3), v_batch: shape (4, 3)
```

### Over Satellites

```python
from astrojax.sgp4 import sgp4_init_jax, sgp4_propagate_unified, elements_to_array

# Stack element arrays for multiple satellites
batch = jnp.stack([elements_to_array(e) for e in element_list])
init_vmap = jax.vmap(lambda e: sgp4_init_jax(e, gravity=WGS72, opsmode="i"))
params_batch = init_vmap(batch)

# Propagate all at once
prop_vmap = jax.vmap(lambda p: sgp4_propagate_unified(p, jnp.float64(60.0)))
r_batch, v_batch = prop_vmap(params_batch)
```

## Gravity Models

Three standard gravity models are available:

| Model | Constant | Description |
|-------|----------|-------------|
| WGS72 | `WGS72` | Standard SGP4 gravity model (default) |
| WGS84 | `WGS84` | Modern WGS84 model |
| WGS72 (legacy) | `WGS72OLD` | Original WGS72 constants |

```python
from astrojax.sgp4 import WGS72, WGS84, sgp4_init

params, method = sgp4_init(elements, WGS84)
```

## High-Level TLE Class

For convenience, the `TLE` class wraps initialization and propagation
with frame transformations:

```python
from astrojax.sgp4 import TLE

tle = TLE(line1, line2)
print(tle.epoch)       # Epoch as astrojax.Epoch
print(tle.method)      # 'n' or 'd'

r, v = tle.state_teme(jnp.float64(60.0))   # TEME frame
r, v = tle.state_gcrf(jnp.float64(60.0))   # GCRF frame (requires EOP)
r, v = tle.state_itrf(jnp.float64(60.0))   # ITRF frame (requires EOP)
```

## Output Format

All propagation functions return the same output:

```
(r, v) where:
    r: jnp.array, shape (3,) — position in km (TEME frame)
    v: jnp.array, shape (3,) — velocity in km/s (TEME frame)
```

If the orbit becomes invalid (e.g., reentry, numerical divergence),
both arrays are filled with `NaN`.
