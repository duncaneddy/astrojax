# MPC Asteroids

The `astrojax.datasets` module provides access to the Minor Planet Center
(MPC) asteroid orbit catalog.  It handles downloading, caching, parsing,
and querying orbital elements for over 1.4 million numbered asteroids,
and can propagate heliocentric ecliptic state vectors at arbitrary epochs
using JAX-compatible computations.

## Loading MPC Data

### Cached Data with Auto-Refresh

The simplest approach uses `load_mpc_asteroids()`, which maintains a
local cache and automatically downloads a fresh copy from the MPC when
the cached file is older than a configurable threshold (default: 7 days):

```python
from astrojax.datasets import load_mpc_asteroids

# Uses default cache location and 7-day refresh
df = load_mpc_asteroids()
print(df.shape)  # (number_of_asteroids, 13)

# Custom cache path and 1-day refresh
df = load_mpc_asteroids("/tmp/my_mpc/mpcorb_extended.json.gz", max_age_days=1.0)
```

If the download fails (network unavailable, MPC server down, etc.), the
function falls back to the existing cached file.  A `RuntimeError` is
raised only when no cached file exists and the download also fails.

| Scenario | Behaviour |
|----------|-----------|
| File missing, download succeeds | Load from fresh download |
| File missing, download fails | Raise `RuntimeError` |
| File stale, download succeeds | Load from fresh download |
| File stale, download fails | Fall back to cached file |
| File fresh | Load from cached file |

The cache location defaults to `~/.cache/astrojax/datasets/mpc/` and
can be overridden with the `ASTROJAX_CACHE` environment variable.

### Custom Data File

Load from any local `mpcorb_extended.json.gz` file:

```python
from astrojax.datasets import load_mpc_from_file

df = load_mpc_from_file("/path/to/mpcorb_extended.json.gz")
```

## Querying Asteroids

Use `get_asteroid_ephemeris()` to look up a single asteroid's orbital
elements from the loaded DataFrame.  You can search by number or by
name:

```python
from astrojax.datasets import load_mpc_asteroids, get_asteroid_ephemeris

df = load_mpc_asteroids()

# By number
ceres = get_asteroid_ephemeris(df, 1)
print(ceres["name"])  # "Ceres"

# By name
vesta = get_asteroid_ephemeris(df, "Vesta")
print(vesta["number"])  # "4"
```

The returned dictionary contains all orbital element columns plus
metadata:

```python
{
    "name", "number", "principal_desig", "epoch_jd",
    "a", "e", "i", "node", "peri", "M", "n", "H",
}
```

A `KeyError` is raised if the asteroid is not found in the catalog.

## Computing State Vectors

`asteroid_state_ecliptic()` propagates Keplerian elements to an
arbitrary Julian Date and returns a heliocentric ecliptic J2000
Cartesian state vector.  The function uses two-body propagation:
advance the mean anomaly, solve Kepler's equation, then convert to
Cartesian coordinates.

```python
import jax.numpy as jnp
from astrojax.datasets import (
    load_mpc_asteroids,
    get_asteroid_ephemeris,
    asteroid_state_ecliptic,
)

df = load_mpc_asteroids()
ceres = get_asteroid_ephemeris(df, 1)

oe = jnp.array([ceres["a"], ceres["e"], ceres["i"],
                 ceres["node"], ceres["peri"], ceres["M"]])

# SI units (metres, m/s) — default
state_si = asteroid_state_ecliptic(ceres["epoch_jd"], oe, 2460100.5)

# AU and AU/day
state_au = asteroid_state_ecliptic(ceres["epoch_jd"], oe, 2460100.5, use_au=True)
```

The orbital element array has the format
`[a_AU, e, i_deg, node_deg, peri_deg, M_deg]` — semi-major axis in AU
and all angles in degrees.

### JIT and vmap Compatibility

`asteroid_state_ecliptic()` uses JAX primitives throughout and is
compatible with `jax.jit`, `jax.vmap`, and `jax.grad`:

```python
import jax

# JIT compilation
jit_state = jax.jit(asteroid_state_ecliptic, static_argnames=("use_au",))
state = jit_state(ceres["epoch_jd"], oe, 2460100.5)

# Vectorized over target dates
jd_batch = jnp.array([2460100.5, 2460200.5, 2460300.5])
states = jax.vmap(lambda jd: asteroid_state_ecliptic(ceres["epoch_jd"], oe, jd))(jd_batch)
```

## Packed Epoch Decoding

The MPC encodes osculation epochs as 5-character packed strings.  Two
utility functions decode them:

```python
from astrojax.datasets import unpack_mpc_epoch, packed_mpc_epoch_to_jd

# Decode to (year, month, day)
year, month, day = unpack_mpc_epoch("K24BN")  # (2024, 11, 23)

# Decode directly to Julian Date (TT)
jd = packed_mpc_epoch_to_jd("J9611")  # 2450083.5
```

The encoding uses one character for century (`I`=18xx, `J`=19xx,
`K`=20xx), two digits for the year, one character for the month
(`1`–`9` for Jan–Sep, `A`=Oct, `B`=Nov, `C`=Dec), and one character
for the day (`1`–`9`, `A`=10, ..., `V`=31).

## DataFrame Columns

The Polars DataFrame returned by `load_mpc_asteroids()` contains the
following columns:

| Column | Type | Description |
|--------|------|-------------|
| `number` | Utf8 | Asteroid number |
| `name` | Utf8 | Asteroid name (may be null) |
| `principal_desig` | Utf8 | Principal designation |
| `epoch_packed` | Utf8 | MPC packed epoch string |
| `epoch_jd` | Float64 | Epoch as Julian Date (TT) |
| `a` | Float64 | Semi-major axis [AU] |
| `e` | Float64 | Eccentricity |
| `i` | Float64 | Inclination [deg] |
| `node` | Float64 | Longitude of ascending node [deg] |
| `peri` | Float64 | Argument of perihelion [deg] |
| `M` | Float64 | Mean anomaly [deg] |
| `n` | Float64 | Mean motion [deg/day] |
| `H` | Float64 | Absolute magnitude |

## End-to-End Example

```python
import jax
import jax.numpy as jnp
from astrojax.datasets import (
    load_mpc_asteroids,
    get_asteroid_ephemeris,
    asteroid_state_ecliptic,
)

# 1. Load the catalog (downloads if needed)
df = load_mpc_asteroids()
print(f"Loaded {df.shape[0]} asteroids")

# 2. Look up Ceres by number
ceres = get_asteroid_ephemeris(df, 1)
print(f"Ceres epoch JD: {ceres['epoch_jd']}")

# 3. Build the orbital element vector
oe = jnp.array([ceres["a"], ceres["e"], ceres["i"],
                 ceres["node"], ceres["peri"], ceres["M"]])

# 4. Compute heliocentric ecliptic state at a target date
target_jd = 2460400.5
state = asteroid_state_ecliptic(ceres["epoch_jd"], oe, target_jd)
print(f"Position [m]:  {state[:3]}")
print(f"Velocity [m/s]: {state[3:]}")

# 5. Batch propagation over multiple dates with vmap
jd_batch = jnp.linspace(2460000.5, 2460365.5, 12)
states = jax.vmap(
    lambda jd: asteroid_state_ecliptic(ceres["epoch_jd"], oe, jd)
)(jd_batch)
print(f"Batch shape: {states.shape}")  # (12, 6)
```
