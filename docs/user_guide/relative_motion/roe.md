# Relative Orbital Elements (ROE)

An alternative to describing relative motion in the RTN frame is to
use **Quasi-Nonsingular Relative Orbital Elements** (ROE).  While RTN
coordinates oscillate rapidly with orbital position, ROE components
vary slowly (secular and long-period only), making them well-suited
for formation design and station-keeping.

The ROE vector $\delta\boldsymbol{\alpha} = [da,\; d\lambda,\; de_x,\; de_y,\; di_x,\; di_y]$
is defined as:

| Component | Definition | Units |
|-----------|-----------|-------|
| $da$ | $(a_d - a_c) / a_c$ | dimensionless |
| $d\lambda$ | $(u_d - u_c) + (\Omega_d - \Omega_c) \cos i_c$ | rad (or deg) |
| $de_x$ | $e_d \cos\omega_d - e_c \cos\omega_c$ | dimensionless |
| $de_y$ | $e_d \sin\omega_d - e_c \sin\omega_c$ | dimensionless |
| $di_x$ | $i_d - i_c$ | rad (or deg) |
| $di_y$ | $(\Omega_d - \Omega_c) \sin i_c$ | rad (or deg) |

where $u = M + \omega$ is the argument of latitude.

## OE to ROE

`state_oe_to_roe` computes the ROE from chief and deputy Keplerian
elements:

```python
import jax.numpy as jnp
from astrojax.constants import R_EARTH
from astrojax.relative_motion import state_oe_to_roe

oe_chief = jnp.array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
oe_deputy = jnp.array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

roe = state_oe_to_roe(oe_chief, oe_deputy, use_degrees=True)
```

The inverse, `state_roe_to_oe`, recovers the deputy's orbital elements
from the chief elements and the ROE:

```python
from astrojax.relative_motion import state_roe_to_oe

oe_deputy_recovered = state_roe_to_oe(oe_chief, roe, use_degrees=True)
```

## ECI to ROE

For convenience, `state_eci_to_roe` and `state_roe_to_eci` compose the
ECI$\leftrightarrow$KOE and OE$\leftrightarrow$ROE transformations,
allowing direct conversion between ECI state vectors and ROE:

```python
from astrojax.coordinates import state_koe_to_eci
from astrojax.relative_motion import state_eci_to_roe, state_roe_to_eci

x_chief = state_koe_to_eci(oe_chief, use_degrees=True)
x_deputy = state_koe_to_eci(oe_deputy, use_degrees=True)

roe = state_eci_to_roe(x_chief, x_deputy, use_degrees=True)
x_deputy_recovered = state_roe_to_eci(x_chief, roe, use_degrees=True)
```
