# Anomaly Conversions

The module provides six functions for converting between the three
anomaly types: mean ($M$), eccentric ($E$), and true ($\nu$).

The direct conversions are:

- **Eccentric to mean** (Kepler's equation): $M = E - e \sin E$
- **Mean to eccentric** (Newton-Raphson iteration): solves $M = E - e \sin E$ for $E$
- **True to eccentric**: $E = \text{atan2}\!\bigl(\sin\nu\,\sqrt{1-e^2},\; \cos\nu + e\bigr)$
- **Eccentric to true**: $\nu = \text{atan2}\!\bigl(\sin E\,\sqrt{1-e^2},\; \cos E - e\bigr)$

The composite conversions chain through eccentric anomaly:

- **True to mean**: $\nu \to E \to M$
- **Mean to true**: $M \to E \to \nu$

```python
from astrojax.orbits import (
    anomaly_eccentric_to_mean,
    anomaly_mean_to_eccentric,
    anomaly_true_to_eccentric,
    anomaly_eccentric_to_true,
    anomaly_true_to_mean,
    anomaly_mean_to_true,
)

e = 0.1

# Degrees
M = anomaly_eccentric_to_mean(90.0, e, use_degrees=True)   # ~84.27 deg
E = anomaly_mean_to_eccentric(M, e, use_degrees=True)      # ~90.0 deg (roundtrip)

# Radians (default)
import jax.numpy as jnp
nu = anomaly_mean_to_true(jnp.pi / 2.0, e)
```

## Kepler Equation Solver

The mean-to-eccentric anomaly conversion solves Kepler's equation using
Newton-Raphson iteration. The solver uses `jax.lax.fori_loop` with 10
fixed iterations, making it fully traceable by JAX's compiler. The
initial guess is $E_0 = M$ for $e < 0.8$ and $E_0 = \pi$ for higher
eccentricities.
