# Space Weather

The `astrojax.space_weather` module provides solar and geomagnetic
activity data for atmospheric density models.  Space weather indices
drive the NRLMSISE-00 model's predictions of thermospheric temperature
and density, which directly affect atmospheric drag on LEO satellites.

## Why Space Weather Matters

Solar EUV radiation heats the thermosphere, causing it to expand and
increasing drag on satellites. Geomagnetic storms driven by solar wind
further modulate upper-atmosphere density. Two families of indices
capture these effects:

| Index | Physical Meaning | Typical Range |
|-------|-----------------|---------------|
| F10.7 | 10.7 cm solar radio flux [sfu] | 60-300 sfu |
| F10.7a | 81-day average F10.7 | 60-250 sfu |
| Kp | 3-hourly geomagnetic index | 0-9 |
| Ap | 3-hourly geomagnetic index (linear) | 0-400 |
| Ap daily | Daily average Ap | 0-400 |

The NRLMSISE-00 model requires F10.7, the 81-day average F10.7a, and a
structured 7-element Ap array describing recent geomagnetic history.

## JIT-Compatible Design

Like the [EOP module](eop.md), astrojax stores space weather data as
sorted JAX arrays in a `NamedTuple`.  Lookups use `jnp.searchsorted`
for O(log n) binary search, making everything compatible with
`jax.jit`, `jax.vmap`, and `jax.grad`.

The `SpaceWeatherData` type is a JAX pytree automatically, so it can be
passed through JIT boundaries and used with `jax.lax` control flow.

## Loading Space Weather Data

### Bundled Data (Recommended)

The simplest approach loads the bundled `sw19571001.txt` file:

```python
from astrojax.space_weather import load_default_sw, get_sw_f107_obs

sw = load_default_sw()
f107 = get_sw_f107_obs(sw, 59569.0)
```

### Custom Data File

Load from any CSSI-format space weather file:

```python
from astrojax.space_weather import load_sw_from_file

sw = load_sw_from_file("/path/to/sw19571001.txt")
```

### Cached Data with Auto-Refresh

For production use, `load_cached_sw` keeps a local copy and
automatically downloads a fresh version from CelesTrak when the
cached copy is older than a configurable threshold (default: 7 days):

```python
from astrojax.space_weather import load_cached_sw, get_sw_f107_obs

# Uses default cache location (~/.cache/astrojax/space_weather/) and 7-day refresh
sw = load_cached_sw()
val = get_sw_f107_obs(sw, 59569.0)

# Custom cache path and 1-day refresh
sw = load_cached_sw("/tmp/my_sw/sw19571001.txt", max_age_days=1.0)
```

If the download fails (network unavailable, CelesTrak server down,
etc.), the function falls back to the bundled data so it never raises
on network issues.

| Scenario | Behaviour |
|----------|-----------|
| File missing, download succeeds | Load from fresh download |
| File missing, download fails | Fall back to bundled data |
| File stale, download succeeds | Load from fresh download |
| File stale, download fails | Fall back to bundled data |
| File fresh | Load from cached file |
| Cached file corrupt | Fall back to bundled data |

The cache location defaults to `~/.cache/astrojax/space_weather/` and
can be overridden with the `ASTROJAX_CACHE` environment variable.

!!! note "No auto-refresh inside JIT"
    File I/O is incompatible with `jax.jit`, so call `load_cached_sw()`
    once at program startup and pass the resulting `SpaceWeatherData`
    into your JIT-compiled functions.

### Constant / Zero Space Weather

For testing or when specific constant values are known:

```python
from astrojax.space_weather import static_space_weather, zero_space_weather

# Fixed known values
sw = static_space_weather(f107=150.0, ap=4.0, f107a=150.0)

# All-zero values
sw = zero_space_weather()
```

## Querying Values

All query functions take a `SpaceWeatherData` instance and an MJD:

```python
from astrojax.space_weather import (
    get_sw_f107_obs,
    get_sw_f107_adj,
    get_sw_f107_obs_ctr81,
    get_sw_f107_obs_lst81,
    get_sw_ap,
    get_sw_ap_daily,
    get_sw_kp,
    get_sw_ap_array,
)

# Solar flux
f107 = get_sw_f107_obs(sw, mjd)          # Observed F10.7 [sfu]
f107_adj = get_sw_f107_adj(sw, mjd)      # Adjusted F10.7 [sfu]
f107a = get_sw_f107_obs_ctr81(sw, mjd)   # 81-day centered average
f107a_lst = get_sw_f107_obs_lst81(sw, mjd)  # 81-day last average

# Geomagnetic indices
kp = get_sw_kp(sw, mjd)                  # 3-hourly Kp (0-9)
ap = get_sw_ap(sw, mjd)                  # 3-hourly Ap
ap_daily = get_sw_ap_daily(sw, mjd)      # Daily average Ap

# NRLMSISE-00 structured Ap array (7 elements)
ap_array = get_sw_ap_array(sw, mjd)
```

### The NRLMSISE-00 Ap Array

The `get_sw_ap_array` function builds the 7-element magnetic activity
array required by NRLMSISE-00:

| Index | Value |
|-------|-------|
| `[0]` | Daily Ap |
| `[1]` | Current 3-hour Ap |
| `[2]` | 3-hour Ap at -3h |
| `[3]` | 3-hour Ap at -6h |
| `[4]` | 3-hour Ap at -9h |
| `[5]` | Average of eight 3-hour Ap from 12-33h prior |
| `[6]` | Average of eight 3-hour Ap from 36-57h prior |

This structured array captures the time history of geomagnetic
activity that the NRLMSISE-00 model uses for thermospheric heating
response.

## Using with JIT and vmap

All query functions work inside JAX transformations:

```python
import jax
import jax.numpy as jnp

# JIT compilation
jit_query = jax.jit(get_sw_f107_obs)
val = jit_query(sw, 59569.0)

# Vectorized over MJD batch
mjd_batch = jnp.array([59569.0, 59570.0, 59571.0])
vals = jax.vmap(lambda m: get_sw_f107_obs(sw, m))(mjd_batch)
```

!!! note "Configurable precision"
    Space weather data is stored using the dtype set by
    `astrojax.set_dtype()`.  If `jax_enable_x64` is not enabled, JAX
    silently truncates to float32, which is adequate for most
    applications.
