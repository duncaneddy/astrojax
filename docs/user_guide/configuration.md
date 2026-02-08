# Configuration

AstroJAX uses a module-wide configuration to control the floating-point
precision of all computations.

## Float Dtype

By default, all float computations use `jnp.float32` for GPU/TPU
compatibility. You can switch to `jnp.float64` for higher precision,
or to `jnp.float16`/`jnp.bfloat16` for faster low-precision workloads.

```python
import astrojax
import jax.numpy as jnp

# Switch to float64 for high-precision work
astrojax.set_dtype(jnp.float64)

# Check the current dtype
print(astrojax.get_dtype())  # <class 'jax.numpy.float64'>
```

### Supported Dtypes

| Dtype | Epoch Precision | Use Case |
|-------|----------------|----------|
| `jnp.float16` | ~seconds | Fast training, low-fidelity |
| `jnp.bfloat16` | ~seconds | TPU training |
| `jnp.float32` (default) | ~8 ms | GPU simulation, most applications |
| `jnp.float64` | sub-nanosecond | High-fidelity analysis, validation |

### When to Call `set_dtype`

Call `set_dtype` **before** any `jax.jit` compilation, just like JAX's
own `jax.config.update("jax_enable_x64", True)`. Under JIT, `get_dtype()`
runs during tracing and its result is baked into the compiled program.

```python
import astrojax
import jax
import jax.numpy as jnp

# Set dtype BEFORE defining JIT functions
astrojax.set_dtype(jnp.float64)

@jax.jit
def propagate(state, dt):
    # All internal computations use float64
    ...
```

### Float64 and JAX x64 Mode

Setting `jnp.float64` automatically enables JAX's 64-bit mode via
`jax.config.update("jax_enable_x64", True)`. This is a process-level
setting that cannot be reverted.

### Integer Components

The `Epoch` class uses `jnp.int32` for its Julian Day number (`_jd`)
regardless of the configured float dtype. Only the float components
(`_seconds`, `_kahan_c`) follow the configured dtype.

## Epoch Equality Tolerance

The `Epoch.__eq__` comparison uses a dtype-adaptive tolerance that
scales with precision:

| Dtype | Tolerance |
|-------|-----------|
| `float16` / `bfloat16` | 0.1 s |
| `float32` | 1e-3 s |
| `float64` | 1e-9 s |
