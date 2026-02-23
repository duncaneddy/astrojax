# Access Computation

The `astrojax.access` module predicts when a satellite is visible from a
ground station.  It computes rise/set times, azimuth/elevation/range at
key moments, and supports custom visibility constraints.  All heavy
computation uses JAX for hardware acceleration.

## Purpose and When to Use

Use this module for pass prediction, contact scheduling, and sensor
coverage analysis.  Given a satellite position function and a ground
station, the module finds all time windows where the satellite satisfies
a visibility constraint (e.g. above a minimum elevation angle).

| Scenario | Recommended Function |
|----------|---------------------|
| General pass prediction with Python list output | `find_access_windows` |
| Inside a `jax.jit` / `jax.vmap` pipeline | `find_access_windows_jit` |
| Many windows over a long time span | `find_all_access_windows` |
| Starting from GCRF ephemeris arrays | `find_access_windows_from_ephemeris` |
| Elevation or az/el/range at a single instant | `compute_elevation` / `compute_azel` |

## Available Components

### Data Types

| Type | Description |
|------|-------------|
| `GroundLocation` | Ground station with pre-computed ECEF position and ENZ rotation matrix |
| `AccessWindow` | A single visibility window (rise/set/max times, az/el/range, duration) |
| `AccessResult` | Fixed-shape JIT-compatible result with a `valid` mask |

### Functions

| Function | JIT-compatible | Returns |
|----------|---------------|---------|
| `ground_location` | No | `GroundLocation` |
| `compute_elevation` | Yes | Scalar elevation (rad) |
| `compute_azel` | Yes | `[az, el, range]` |
| `find_access_windows` | No (hybrid) | `list[AccessWindow]` |
| `find_access_windows_jit` | Yes | `AccessResult` |
| `find_all_access_windows` | No (paging wrapper) | `list[AccessWindow]` |
| `find_access_windows_from_ephemeris` | No (hybrid) | `list[AccessWindow]` |

## Ground Locations

A `GroundLocation` bundles geodetic coordinates with pre-computed ECEF
position and ECEF-to-ENZ rotation matrix.  Pre-computing these avoids
redundant geodetic-to-ECEF conversions when the same station is used
across many evaluations.

```python
from astrojax.access import ground_location

# Coordinates in degrees (default)
loc = ground_location(lon=-122.0, lat=37.0, alt=100.0)

# Or in radians
import jax.numpy as jnp
loc = ground_location(
    lon=jnp.deg2rad(-122.0),
    lat=jnp.deg2rad(37.0),
    alt=100.0,
    use_degrees=False,
)

# Pre-computed arrays ready for use
print(loc.ecef.shape)     # (3,)
print(loc.rot_enz.shape)  # (3, 3)
```

## Computing Elevation and Az/El/Range

The low-level building blocks compute topocentric angles for a single
satellite position:

```python
from astrojax.access import ground_location, compute_elevation, compute_azel
import jax.numpy as jnp

loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

# Satellite ECEF position (metres)
sat_ecef = jnp.array([7000e3, 0.0, 0.0])

# Elevation only
el = compute_elevation(sat_ecef, loc.ecef, loc.rot_enz)

# Full azimuth, elevation, range
azel = compute_azel(sat_ecef, loc.ecef, loc.rot_enz)
az, el, rng = azel[0], azel[1], azel[2]
```

Both functions accept an optional `rot_enz` argument.  Passing the
pre-computed matrix from `GroundLocation` avoids recomputing it on
every call.

## Finding Access Windows (Hybrid)

`find_access_windows` is the easiest entry point.  It uses a three-stage
hybrid approach:

1. **JIT-accelerated constraint grid** -- `vmap` over a coarse time array
2. **Python-level window detection** -- sign-change detection with NumPy
3. **JIT-accelerated refinement** -- bisection search for precise
   rise/set times, golden-section search for max elevation

```python
from astrojax.access import ground_location, find_access_windows
from astrojax.config import get_dtype
from astrojax.constants import R_EARTH
import jax.numpy as jnp

dtype = get_dtype()
loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

# Define a satellite position function: f(t) -> ECEF [x,y,z]
r = dtype(R_EARTH + 500e3)
omega = dtype(2.0 * jnp.pi / 5400.0)

def position_ecef(t):
    t = jnp.asarray(t, dtype=dtype)
    theta = omega * t
    return jnp.array([r * jnp.cos(theta), dtype(0.0), r * jnp.sin(theta)])

# Find all windows over one orbit
windows = find_access_windows(
    position_ecef, loc,
    t_start=0.0, t_end=5400.0,
    min_elevation=5.0, use_degrees=True,
    dt=30.0,
)

for w in windows:
    print(f"Rise: {w.t_rise:.1f}s  Set: {w.t_set:.1f}s  "
          f"Max El: {jnp.rad2deg(w.el_max):.1f}deg  "
          f"Duration: {w.duration:.1f}s")
```

The `dt` parameter controls the coarse grid step size.  Smaller values
catch short passes but increase computation.  The default (60 s) works
well for LEO orbits.

## JIT-Compilable Access Windows

`find_access_windows_jit` replaces the Python window-detection stage
with a `lax.scan`, producing fixed-shape output arrays.  This makes it
composable inside `jax.jit` and `jax.vmap` pipelines.

```python
from astrojax.access import ground_location, find_access_windows_jit

loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

result = find_access_windows_jit(
    position_ecef,
    loc.ecef,
    loc.rot_enz,
    t_start=0.0,
    t_end=5400.0,
    max_windows=10,   # static: determines output shape
    n_steps=181,       # static: determines grid resolution
)

# result.valid is a (max_windows,) bool mask
n = int(result.n_windows)
for i in range(n):
    print(f"Window {i}: rise={float(result.t_rise[i]):.1f}s "
          f"set={float(result.t_set[i]):.1f}s")
```

!!! note "`max_windows` and `n_steps` are static"
    Both `max_windows` and `n_steps` determine array shapes and must be
    passed via `static_argnums` when wrapping with `jax.jit`.  Changing
    them triggers recompilation.

### Paging Over Long Time Spans

`find_all_access_windows` repeatedly calls `find_access_windows_jit`
in batches, advancing the start time past each batch.  It returns a
Python list of `AccessWindow` instances, combining the JIT-compiled
performance with flexible output:

```python
from astrojax.access import ground_location, find_all_access_windows

loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

windows = find_all_access_windows(
    position_ecef,
    loc.ecef,
    loc.rot_enz,
    t_start=0.0,
    t_end=86400.0,     # 24 hours
    max_windows=100,
    n_steps=181,
    batch_size=10,
)
```

## Working with Ephemeris Data

When you have pre-computed GCRF position arrays (e.g. from orbit
propagation), use `find_access_windows_from_ephemeris`.  It handles the
GCRF-to-ITRF frame transformation and builds a linear interpolation
function internally:

```python
from astrojax.access import ground_location, find_access_windows_from_ephemeris
from astrojax.eop import zero_eop
from astrojax.epoch import Epoch
import jax.numpy as jnp

eop = zero_eop()
epoch0 = Epoch(2024, 1, 1, 12, 0, 0.0)
loc = ground_location(lon=0.0, lat=0.0, alt=0.0)

# positions_gcrf: (N, 3) array from orbit propagation
# times: (N,) array of seconds since epoch0
# epochs: list of Epoch instances for each time step
windows = find_access_windows_from_ephemeris(
    positions_gcrf, times, loc, eop, epochs,
    min_elevation=5.0, use_degrees=True,
)
```

## Data Type Reference

### `GroundLocation` Fields

| Field | Type | Description |
|-------|------|-------------|
| `lon` | `float` | Longitude in radians |
| `lat` | `float` | Latitude in radians |
| `alt` | `float` | Altitude above WGS84 in metres |
| `ecef` | `Array[3]` | Pre-computed ECEF position in metres |
| `rot_enz` | `Array[3,3]` | Pre-computed ECEF-to-ENZ rotation matrix |

### `AccessWindow` Fields

| Field | Type | Description |
|-------|------|-------------|
| `t_rise` | `float` | Window open time (seconds) |
| `t_set` | `float` | Window close time (seconds) |
| `t_max_el` | `float` | Time of maximum elevation (seconds) |
| `az_rise` / `el_rise` / `rng_rise` | `float` | Az/el/range at rise |
| `az_set` / `el_set` / `rng_set` | `float` | Az/el/range at set |
| `az_max` / `el_max` / `rng_max` | `float` | Az/el/range at max elevation |
| `duration` | `float` | Window duration in seconds |

### `AccessResult` Fields

Same fields as `AccessWindow` but as arrays of shape `(max_windows,)`,
plus:

| Field | Type | Description |
|-------|------|-------------|
| `valid` | `Array[max_windows]` (bool) | `True` for slots with real data |
| `n_windows` | `Array` (int32 scalar) | Number of valid windows found |

## JAX Compatibility

### JIT Compilation

```python
import jax

jitted = jax.jit(
    find_access_windows_jit,
    static_argnums=(0, 5, 6),  # position_fn, max_windows, n_steps
)
result = jitted(
    position_ecef, loc.ecef, loc.rot_enz,
    0.0, 5400.0, 10, 181,
)
```

### Batch Over Multiple Stations

```python
import jax

loc_a = ground_location(lon=0.0, lat=0.0, alt=0.0)
loc_b = ground_location(lon=90.0, lat=0.0, alt=0.0)

stations = jnp.stack([loc_a.ecef, loc_b.ecef])
rots = jnp.stack([loc_a.rot_enz, loc_b.rot_enz])

def access_for_station(station, rot):
    return find_access_windows_jit(
        position_ecef, station, rot,
        0.0, 5400.0,
        max_windows=10, n_steps=181,
    )

results = jax.vmap(access_for_station)(stations, rots)
# results.t_rise.shape == (2, 10)
```

!!! note "Configurable precision"
    All access functions respect `astrojax.set_dtype()`.  Call
    `set_dtype()` before JIT compilation to control float32 vs float64
    precision.
