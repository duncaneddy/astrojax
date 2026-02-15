# Asteroid Masses

The `astrojax.datasets` module provides access to the SBN Archive
asteroid masses compilation dataset.  It handles downloading, caching,
and parsing mass, density, and shape data for 248 asteroids from a
fixed-width ASCII table distributed as a ZIP archive.

## Loading Asteroid Masses

### Cached Data with Auto-Refresh

The simplest approach uses `load_asteroid_masses()`, which maintains a
local cache and automatically downloads a fresh copy from the SBN Archive
when the cached file is older than a configurable threshold (default:
30 days):

```python
from astrojax.datasets import load_asteroid_masses

# Uses default cache location and 30-day refresh
df = load_asteroid_masses()
print(df.shape)  # (248, 15)

# Custom cache path and 7-day refresh
df = load_asteroid_masses("/tmp/my_astmass/compil.ast.masses.zip", max_age_days=7.0)
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

The cache location defaults to `~/.cache/astrojax/datasets/astmass/` and
can be overridden with the `ASTROJAX_CACHE` environment variable.

### Custom Data File

Load from any local `compil.ast.masses.zip` file:

```python
from astrojax.datasets import load_astmass_from_file

df = load_astmass_from_file("/path/to/compil.ast.masses.zip")
```

## DataFrame Columns

The Polars DataFrame returned by `load_asteroid_masses()` contains the
following columns:

| Column | Type | Description |
|--------|------|-------------|
| `ast_number` | Int64 | Asteroid catalog number |
| `ast_name` | Utf8 | Asteroid name |
| `prov_desig` | Utf8 | Provisional designation |
| `satellite_name` | Utf8 | Satellite name (if applicable) |
| `mass_sol` | Float64 | Mass in solar masses |
| `mass_sol_unc` | Float64 | Mass uncertainty in solar masses |
| `mass_kg` | Float64 | Mass [kg] |
| `mass_kg_unc` | Float64 | Mass uncertainty [kg] |
| `bulk_density` | Float64 | Bulk density [g/cm^3] |
| `bulk_density_unc` | Float64 | Bulk density uncertainty [g/cm^3] |
| `axis_a` | Float64 | Ellipsoid diameter a [km] |
| `axis_b` | Float64 | Ellipsoid diameter b [km] |
| `axis_c` | Float64 | Ellipsoid diameter c [km] |
| `equiv_radius_unc` | Float64 | Equivalent radius uncertainty [km] |
| `mass_ref` | Utf8 | Mass determination reference |

## Example

```python
from astrojax.datasets import load_asteroid_masses

# 1. Load the dataset (downloads if needed)
df = load_asteroid_masses()
print(f"Loaded {df.shape[0]} asteroid mass records")

# 2. Filter to asteroids with known bulk density
dense = df.filter(df["bulk_density"].is_not_null())
print(f"{dense.shape[0]} asteroids have bulk density data")

# 3. Find the most massive asteroids
top = df.sort("mass_kg", descending=True, nulls_last=True).head(5)
print(top.select(["ast_number", "ast_name", "mass_kg", "bulk_density"]))
```
