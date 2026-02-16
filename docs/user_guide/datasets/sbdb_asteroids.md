# SBDB Asteroids

The `astrojax.datasets` module provides access to the JPL Small Body
Database (SBDB) asteroid catalog.  It handles downloading, caching,
parsing, and querying orbital elements and physical properties
(diameter, GM) for approximately 875,000 asteroids, and can propagate
heliocentric ecliptic state vectors at arbitrary epochs using
JAX-compatible computations.

## Loading SBDB Data

### Cached Data with Auto-Refresh

The simplest approach uses `load_sbdb_asteroids()`, which maintains a
local cache and automatically downloads a fresh copy when the cached
file is older than a configurable threshold (default: 365 days):

```python
from astrojax.datasets import load_sbdb_asteroids

# Uses default cache location and 365-day refresh
df = load_sbdb_asteroids()
print(df.shape)

# Custom cache path and 30-day refresh
df = load_sbdb_asteroids("/tmp/my_sbdb/sbdb_asteroid_data.csv", max_age_days=30.0)
```

If the download fails (network unavailable, server down, etc.), the
function falls back to the existing cached file.  A `RuntimeError` is
raised only when no cached file exists and the download also fails.

| Scenario | Behaviour |
|----------|-----------|
| File missing, download succeeds | Load from fresh download |
| File missing, download fails | Raise `RuntimeError` |
| File stale, download succeeds | Load from fresh download |
| File stale, download fails | Fall back to cached file |
| File fresh | Load from cached file |

The cache location defaults to `~/.cache/astrojax/datasets/sbdb/` and
can be overridden with the `ASTROJAX_CACHE` environment variable.

### Custom Data File

Load from any local SBDB CSV file:

```python
from astrojax.datasets import load_sbdb_from_file

df = load_sbdb_from_file("/path/to/sbdb_asteroid_data.csv")
```

## Querying Asteroids

Use `get_sbdb_asteroid_ephemeris()` to look up a single asteroid's
orbital elements and physical properties from the loaded DataFrame.
You can search by number or by name:

```python
from astrojax.datasets import load_sbdb_asteroids, get_sbdb_asteroid_ephemeris

df = load_sbdb_asteroids()

# By number
ceres = get_sbdb_asteroid_ephemeris(df, 1)
print(ceres["name"])       # "Ceres"
print(ceres["diameter"])   # 939.4  (km)
print(ceres["GM"])         # 62.6284 (km^3/s^2)

# By name
vesta = get_sbdb_asteroid_ephemeris(df, "Vesta")
print(vesta["number"])  # "4"
```

The returned dictionary contains orbital element columns, physical
properties, and metadata:

```python
{
    "name", "full_name", "number", "epoch_jd",
    "a", "e", "i", "node", "peri", "M", "n",
    "diameter", "GM",
}
```

A `KeyError` is raised if the asteroid is not found in the catalog.

## Computing State Vectors

`asteroid_state_ecliptic()` propagates Keplerian elements to an
arbitrary Julian Date and returns a heliocentric ecliptic J2000
Cartesian state vector.  Since SBDB orbital elements use the same
ecliptic J2000 frame and conventions as MPC elements, the same
function works directly:

```python
import jax.numpy as jnp
from astrojax.datasets import (
    load_sbdb_asteroids,
    get_sbdb_asteroid_ephemeris,
    asteroid_state_ecliptic,
)

df = load_sbdb_asteroids()
ceres = get_sbdb_asteroid_ephemeris(df, 1)

oe = jnp.array([ceres["a"], ceres["e"], ceres["i"],
                 ceres["node"], ceres["peri"], ceres["M"]])

# SI units (metres, m/s) — default
state_si = asteroid_state_ecliptic(ceres["epoch_jd"], oe, 2460100.5)

# AU and AU/day
state_au = asteroid_state_ecliptic(ceres["epoch_jd"], oe, 2460100.5, use_au=True)
```

## DataFrame Columns

The Polars DataFrame returned by `load_sbdb_asteroids()` contains the
following columns:

| Column | Type | Description |
|--------|------|-------------|
| `full_name` | Utf8 | Full asteroid name (e.g. "1 Ceres") |
| `number` | Utf8 | Asteroid number |
| `name` | Utf8 | Asteroid name (may be null) |
| `diameter` | Float64 | Diameter [km] (may be null) |
| `GM` | Float64 | Gravitational parameter [km^3/s^2] (may be null) |
| `epoch_jd` | Float64 | Epoch as Julian Date (TT) |
| `epoch_mjd` | Float64 | Epoch as Modified Julian Date |
| `epoch_cal` | Int64 | Epoch as calendar date (YYYYMMDD) |
| `e` | Float64 | Eccentricity |
| `a` | Float64 | Semi-major axis [AU] |
| `q` | Float64 | Perihelion distance [AU] |
| `i` | Float64 | Inclination [deg] |
| `node` | Float64 | Longitude of ascending node [deg] |
| `peri` | Float64 | Argument of perihelion [deg] |
| `M` | Float64 | Mean anomaly [deg] |
| `ad` | Float64 | Aphelion distance [AU] |
| `n` | Float64 | Mean motion [deg/day] |
| `tp` | Float64 | Time of perihelion passage [JD] |
| `tp_cal` | Int64 | Time of perihelion passage (YYYYMMDD) |
| `per` | Float64 | Orbital period [days] |
| `per_y` | Float64 | Orbital period [years] |

## SBDB vs MPC Comparison

| Feature | MPC | SBDB |
|---------|-----|------|
| Source | Minor Planet Center | JPL Horizons |
| Records | ~1.4M | ~875K |
| Physical data | None | `diameter`, `GM` |
| Extra orbital | `H` magnitude | `q`, `ad`, `tp`, `per`, `per_y` |
| Update frequency | Weekly | Static snapshot |
| Column naming | Native | Renamed to MPC convention |
| State computation | `asteroid_state_ecliptic` | Same function |

Both datasets use the same `asteroid_state_ecliptic()` function for
heliocentric ecliptic J2000 state vector computation.

## End-to-End Example

```python
import jax
import jax.numpy as jnp
from astrojax.datasets import (
    load_sbdb_asteroids,
    get_sbdb_asteroid_ephemeris,
    asteroid_state_ecliptic,
)

# 1. Load the catalog (downloads if needed)
df = load_sbdb_asteroids()
print(f"Loaded {df.shape[0]} asteroids")

# 2. Look up Ceres by number
ceres = get_sbdb_asteroid_ephemeris(df, 1)
print(f"Ceres diameter: {ceres['diameter']} km")
print(f"Ceres GM: {ceres['GM']} km^3/s^2")

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
