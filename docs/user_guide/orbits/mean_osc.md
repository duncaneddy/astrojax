# Mean and Osculating Elements

Real satellite orbits are perturbed by Earth's oblateness ($J_2$). The
**osculating** elements describe the instantaneous orbit at a given
moment, while **mean** elements average out the short-period $J_2$
oscillations. Mean elements are useful for mission design, constellation
maintenance, and long-term orbit characterisation because they vary
smoothly over time.

The conversion between mean and osculating elements uses first-order
Brouwer-Lyddane theory. The perturbation parameter is:

$$
\gamma_2 = \frac{J_2}{2} \left(\frac{R_\oplus}{a}\right)^2
$$

For a typical LEO orbit ($a \approx 6878$ km), $\gamma_2 \approx 5 \times 10^{-4}$,
so the first-order corrections are on the order of hundreds of metres
in semi-major axis and milliradians in angular elements.

```python
import jax.numpy as jnp
from astrojax.constants import R_EARTH
from astrojax.orbits import state_koe_mean_to_osc, state_koe_osc_to_mean

# Define mean elements: [a, e, i, Omega, omega, M]
mean = jnp.array([R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])

# Mean -> Osculating
osc = state_koe_mean_to_osc(mean, use_degrees=True)

# Osculating -> Mean (roundtrip)
mean_back = state_koe_osc_to_mean(osc, use_degrees=True)
```

The forward and inverse transformations are not exact inverses — the
roundtrip error is of order $J_2^2$, which is small but nonzero.

!!! note "float32 precision"
    At float32 precision, the roundtrip SMA error is typically
    under 100 metres and angular errors under 0.01 radians
    (~0.6 degrees). Use `set_dtype(jnp.float64)` for higher
    fidelity if needed.

!!! warning "Critical inclination"
    The first-order theory contains a $(1 - 5\cos^2 i)$ term in
    several denominators, which goes to zero at the critical
    inclination ($i \approx 63.4°$ or $i \approx 116.6°$). The
    transformation should not be used near these values.
