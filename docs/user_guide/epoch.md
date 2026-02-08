# Epoch

The `Epoch` class represents a single instant in time with higher precision
than a raw float32 Julian Date. Internally it stores an **int32 Julian Day
number** and **float32 seconds within the day**, giving ~8 ms time precision
while remaining GPU/TPU friendly (no float64 required).

## Creating Epochs

### From Calendar Date Components

Provide year, month, and day. Hour, minute, and second are optional:

```python
from astrojax import Epoch

epc = Epoch(2024, 6, 15)                    # midnight
epc = Epoch(2024, 6, 15, 14, 30, 0.0)       # 14:30:00 UTC
epc = Epoch(2024, 6, 15, 14, 30, 15.123)    # with fractional seconds
```

### From ISO 8601 String

```python
epc = Epoch("2024-06-15")
epc = Epoch("2024-06-15T14:30:00Z")
epc = Epoch("2024-06-15T14:30:15.123Z")
```

### Copy Constructor

```python
epc2 = Epoch(epc)  # independent copy
```

## Accessing Time Properties

### Julian Date and Modified Julian Date

```python
epc = Epoch(2000, 1, 1, 12, 0, 0.0)

print(epc.jd())   # ~2451545.0 (float32, lossy for sub-day)
print(epc.mjd())   # ~51544.5
```

!!! warning "Precision of `jd()` and `mjd()`"
    These return a single float32. Near typical JD values, float32 has ~0.25 day
    precision. For precise time-of-day information use `caldate()` or Epoch
    subtraction instead.

### Calendar Date

```python
year, month, day, hour, minute, second = epc.caldate()
print(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:06.3f}Z")
```

`caldate()` computes hours/minutes/seconds directly from the internal float32
seconds field, preserving the full ~8 ms precision of the split representation.

### Greenwich Mean Sidereal Time

```python
gmst_rad = epc.gmst()                  # radians (default)
gmst_deg = epc.gmst(use_degrees=True)  # degrees
```

Uses the IAU 1982 (GMST82) polynomial. Assumes UTC approximates UT1.

## Arithmetic

### Adding and Subtracting Seconds

```python
epc = Epoch(2024, 1, 1)

epc2 = epc + 3600.0      # one hour later
epc3 = epc - 86400.0     # one day earlier
```

The `+=` and `-=` operators also work. They return a new `Epoch` (functional
style, required for JAX traceability):

```python
epc = Epoch(2024, 1, 1)
epc += 3600.0   # rebinds `epc` to a new Epoch
```

### Kahan Compensated Summation

Repeated small additions (e.g., time-stepping in numerical integration) use
Kahan compensated summation to prevent floating-point error accumulation. The
compensator is tracked automatically — no user action needed.

### Differencing Two Epochs

Subtracting one Epoch from another returns the time difference in seconds:

```python
epc1 = Epoch(2024, 1, 1)
epc2 = Epoch(2024, 1, 2)

dt = epc2 - epc1  # 86400.0 seconds
```

This uses the full split representation, so it is more precise than computing
`epc2.jd() - epc1.jd()`.

## Comparisons

All six comparison operators are supported:

```python
epc1 = Epoch(2024, 1, 1)
epc2 = Epoch(2024, 1, 2)

epc1 < epc2   # True
epc1 == epc1  # True
epc1 != epc2  # True
epc1 >= epc2  # False
```

Equality uses a tolerance of 1 ms, matching the float32 precision of the
seconds field.

## JAX Pytree Integration

`Epoch` is registered as a JAX pytree, so it works with JAX transformations.

### jit

`gmst()` and other JAX-based methods can be called inside JIT-compiled
functions:

```python
import jax

@jax.jit
def compute_gmst(epc):
    return epc.gmst()

epc = Epoch(2024, 6, 15, 12, 0, 0.0)
gmst = compute_gmst(epc)
```

### vmap

Vectorize over a batch of Epochs:

```python
import jax
import jax.numpy as jnp

base = Epoch(2024, 1, 1)
epochs = jax.tree.map(lambda *xs: jnp.stack(xs), *[base + i * 3600.0 for i in range(24)])

gmst_batch = jax.vmap(lambda e: Epoch._from_internal(e._jd, e._seconds, e._kahan_c).gmst())(epochs)
```

### scan

Use `jax.lax.scan` for sequential time stepping:

```python
import jax
import jax.numpy as jnp

def step(carry, _):
    epc = carry
    new_epc = epc + 60.0  # advance 60 seconds
    return new_epc, epc.gmst()

epc0 = Epoch(2024, 1, 1)
final_epc, gmst_history = jax.lax.scan(step, epc0, None, length=10)
```

## String Representation

Epochs print as ISO 8601 strings:

```python
epc = Epoch(2024, 6, 15, 14, 30, 15.123)
print(epc)  # 2024-06-15T14:30:15.123Z
```

## Precision Summary

| Representation | Precision | Use case |
|---|---|---|
| Internal (int32 + float32) | ~8 ms | Arithmetic, comparisons, GMST |
| `jd()` (single float32) | ~0.25 days | Rough JD value, not sub-day |
| `mjd()` (single float32) | ~6 min | Better than JD, still lossy |
| `caldate()` | ~8 ms | Display, logging |
| Epoch subtraction | ~8 ms | Time differences |
