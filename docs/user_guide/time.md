# Time Conversions

The `astrojax.time` module provides functions for converting between calendar
dates, Julian Dates (JD), and Modified Julian Dates (MJD). All functions accept
JAX `ArrayLike` inputs and return `jax.Array` values, so they work seamlessly
with `jax.jit`.

## Calendar Date to Julian Date / MJD

Convert a calendar date to Julian Date or Modified Julian Date using
`caldate_to_jd` and `caldate_to_mjd`. The date components (year, month, day)
are required; hour, minute, and second default to zero.

```python
from astrojax.time import caldate_to_jd, caldate_to_mjd

# J2000.0 epoch: 2000-01-01 12:00:00
jd = caldate_to_jd(2000, 1, 1, 12, 0, 0.0)
print(jd)  # 2451545.0

mjd = caldate_to_mjd(2000, 1, 1, 12, 0, 0.0)
print(mjd)  # 51544.5
```

## Julian Date / MJD to Calendar Date

Convert back to calendar date components using `jd_to_caldate` or
`mjd_to_caldate`. Both return a tuple of six values:
`(year, month, day, hour, minute, second)`.

```python
from astrojax.time import jd_to_caldate, mjd_to_caldate

year, month, day, hour, minute, second = jd_to_caldate(2451545.0)
print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f}")
# 2000-01-01 12:00:00.000

year, month, day, hour, minute, second = mjd_to_caldate(51544.5)
print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f}")
# 2000-01-01 12:00:00.000
```

## JD and MJD Conversion

Convert directly between Julian Date and Modified Julian Date:

```python
from astrojax.time import jd_to_mjd, mjd_to_jd

mjd = jd_to_mjd(2451545.0)  # 51544.5
jd = mjd_to_jd(51544.5)     # 2451545.0
```

The offset between JD and MJD is the constant `JD_MJD_OFFSET = 2400000.5`.

## JAX Compatibility

All time functions use JAX primitives internally and are compatible with
`jax.jit`:

```python
import jax
from astrojax.time import caldate_to_jd

jd_jit = jax.jit(caldate_to_jd)
jd = jd_jit(2024, 6, 15, 0, 0, 0.0)
```

Inputs are `ArrayLike`, so you can pass Python ints/floats or JAX arrays.

!!! note "float32 precision"
    All outputs are float32. A single float32 Julian Date near typical values
    (~2,451,545) has an effective precision of about 0.25 days. For sub-day
    precision, use the [`Epoch`](epoch.md) class, which splits the
    representation into an int32 day number and float32 seconds.
