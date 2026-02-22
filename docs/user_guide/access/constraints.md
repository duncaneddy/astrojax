# Constraints

The `astrojax.access.constraints` module provides factory functions for
building visibility constraints.  Constraints plug into
`find_access_windows` and `find_access_windows_jit` to define custom
visibility conditions beyond a simple elevation threshold.

## Constraint Protocol

A constraint function has the signature:

```python
def constraint_fn(sat_ecef, station_ecef, rot_enz) -> float:
    ...
```

The return value follows a sign convention:

| Value | Meaning |
|-------|---------|
| Positive | Constraint **satisfied** (satellite is visible) |
| Negative | Constraint **violated** (satellite is not visible) |
| Zero | **Boundary** (where bisection finds rise/set times) |

This convention allows the access window finder to detect zero-crossings
and refine boundaries using bisection search.

## Elevation Constraints

The most common constraint requires the satellite to be above a minimum
elevation angle:

```python
from astrojax.access.constraints import elevation_constraint

# Minimum 10 degrees (in radians)
import jax.numpy as jnp
constraint = elevation_constraint(min_el=jnp.deg2rad(10.0))
```

You can also specify an elevation band with both minimum and maximum
bounds.  This is useful for excluding high-elevation passes (e.g.
geostationary interference avoidance):

```python
# Only accept passes between 10 and 70 degrees
constraint = elevation_constraint(
    min_el=jnp.deg2rad(10.0),
    max_el=jnp.deg2rad(70.0),
)
```

## Elevation Mask

An elevation mask defines azimuth-dependent minimum elevation thresholds.
This models terrain obstructions or antenna field-of-view limits:

```python
from astrojax.access.constraints import elevation_mask_constraint
import jax.numpy as jnp

# Define mask: higher threshold to the north (mountains)
mask_az = jnp.array([0.0, jnp.pi/2, jnp.pi, 3*jnp.pi/2])  # N, E, S, W
mask_el = jnp.deg2rad(jnp.array([15.0, 5.0, 5.0, 10.0]))

constraint = elevation_mask_constraint(mask_az, mask_el)
```

The mask interpolates linearly between breakpoints and wraps at
$2\pi$ azimuth.

## Off-Nadir Constraints

Off-nadir constraints bound the angle between the satellite's nadir
direction and the line of sight to the station.  This is useful for
modelling sensor field-of-view from the spacecraft perspective:

```python
from astrojax.access.constraints import off_nadir_constraint
import jax.numpy as jnp

# Sensor can only point up to 60 degrees off nadir
constraint = off_nadir_constraint(max_off_nadir=jnp.deg2rad(60.0))

# Exclude near-nadir (e.g. for SAR imaging geometry)
constraint = off_nadir_constraint(
    max_off_nadir=jnp.deg2rad(60.0),
    min_off_nadir=jnp.deg2rad(20.0),
)
```

The helper `compute_off_nadir(sat_ecef, station_ecef)` returns the
off-nadir angle directly if you need it outside the constraint framework.

## Composing Constraints

Constraint functions compose with boolean-like operators:

| Operator | Meaning | Implementation |
|----------|---------|----------------|
| `constraint_all(*constraints)` | AND -- all must be satisfied | `min(values)` |
| `constraint_any(*constraints)` | OR -- at least one satisfied | `max(values)` |
| `constraint_not(constraint)` | NOT -- flip sign | `-value` |

### Example: Elevation Band with Off-Nadir Limit

```python
from astrojax.access.constraints import (
    constraint_all,
    elevation_constraint,
    off_nadir_constraint,
)
import jax.numpy as jnp

# Satellite must be above 5 degrees AND within 70 degrees off-nadir
combined = constraint_all(
    elevation_constraint(jnp.deg2rad(5.0)),
    off_nadir_constraint(jnp.deg2rad(70.0)),
)
```

### Example: Exclude a Region

```python
from astrojax.access.constraints import (
    constraint_all,
    constraint_not,
    elevation_constraint,
)
import jax.numpy as jnp

# Above 5 degrees but NOT above 80 degrees
combined = constraint_all(
    elevation_constraint(jnp.deg2rad(5.0)),
    constraint_not(elevation_constraint(jnp.deg2rad(80.0))),
)
```

## Using Constraints with Access Functions

Pass a constraint to `find_access_windows` or `find_access_windows_jit`
via the `constraint_fn` parameter.  When provided, it takes precedence
over the `min_elevation` parameter:

```python
from astrojax.access import find_access_windows, ground_location
from astrojax.access.constraints import (
    constraint_all,
    elevation_constraint,
    off_nadir_constraint,
)
import jax.numpy as jnp

loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

combined = constraint_all(
    elevation_constraint(jnp.deg2rad(5.0)),
    off_nadir_constraint(jnp.deg2rad(70.0)),
)

# Hybrid API
windows = find_access_windows(
    position_ecef_fn, loc,
    t_start=0.0, t_end=5400.0,
    dt=30.0,
    constraint_fn=combined,
)

# JIT-compilable API
from astrojax.access import find_access_windows_jit

result = find_access_windows_jit(
    position_ecef_fn,
    loc.ecef, loc.rot_enz,
    t_start=0.0, t_end=5400.0,
    max_windows=10, n_steps=181,
    constraint_fn=combined,
)
```

!!! note "Constraint functions are JIT-compatible"
    All constraint factories return closures that capture their
    parameters, making them fully compatible with `jax.jit` and
    `jax.vmap`.  The factory call itself happens in Python; only the
    returned closure runs inside JAX traces.
