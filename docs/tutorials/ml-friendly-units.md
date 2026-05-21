# ML-Friendly Units: Nondimensional Astrodynamics

Astrojax can propagate orbits in nondimensional ("nondim") units so
that positions and velocities sit in an $O(1)$ range. That makes
`float32` propagation strictly more accurate than the SI default, and
unlocks `bfloat16` for ML training loops where it would otherwise
diverge.

This tutorial walks through the three workflows the feature was
designed for: HCW relative motion, Keplerian propagation, and a full
perturbed force model.

## Why nondim?

A position of $7{,}100{,}000\,\mathrm{m}$ cannot be represented in
`bfloat16` with better than ~4 km precision &mdash; bfloat16 has only ~3
decimal digits of mantissa. The trick is to choose a *unit system* in
which positions are $O(1)$ and the gravitational parameter $\mu$ is
$1$. The physics is unchanged &mdash; it's just a change of units.

`astrojax.nondim.UnitSystem` is the value object that carries those
unit choices.

## Workflow 1: HCW relative motion

```python
import jax.numpy as jnp
from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.nondim import (
    UnitSystem, to_nondim_state, from_nondim_state,
)
from astrojax.relative_motion import hcw_stm

a_chief = R_EARTH + 500e3
units = UnitSystem.from_orbit_relative(
    a_chief, GM_EARTH, LU_rel=100.0,  # 100 m characteristic separation
)

state_si = jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 100 m radial
state_nd = to_nondim_state(state_si, units)             # |r_nd| ~ 1

# Mean motion is exactly 1.0 in these units.
n_nd = 1.0
t_nd = 3000.0 / units.TU

state_nd_final = hcw_stm(t_nd, n_nd) @ state_nd
state_si_final = from_nondim_state(state_nd_final, units)
```

## Workflow 2: Keplerian propagation

```python
import jax.numpy as jnp
from astrojax.constants import GM_EARTH
from astrojax.integrators import rk4_step
from astrojax.nondim import UnitSystem, to_nondim_state, from_nondim_state
from astrojax.orbit_dynamics import accel_point_mass

a = 7000e3
state_si = jnp.array([a, 0.0, 0.0, 0.0, jnp.sqrt(GM_EARTH / a), 0.0])

units = UnitSystem.from_orbit(a, GM_EARTH)   # mu_nd = 1
state_nd = to_nondim_state(state_si, units)
mu_nd = 1.0
dt_nd = 60.0 / units.TU

def deriv(t, s):
    r, v = s[:3], s[3:]
    a = accel_point_mass(r, jnp.zeros(3, dtype=r.dtype), mu_nd)
    return jnp.concatenate([v, a])

result = rk4_step(deriv, 0.0, state_nd, dt_nd)
state_nd_next = result.state
state_si_next = from_nondim_state(state_nd_next, units)
```

## Workflow 3: Full perturbed force model

```python
import jax.numpy as jnp
from astrojax.constants import GM_EARTH, R_EARTH
from astrojax.eop import zero_eop
from astrojax.epoch import Epoch
from astrojax.nondim import UnitSystem, nondim_orbit_dynamics, to_nondim_state
from astrojax.orbit_dynamics import ForceModelConfig
from astrojax.orbit_dynamics.gravity import GravityModel

epoch_0 = Epoch(2024, 1, 1, 12, 0, 0.0)
units = UnitSystem.from_orbit(R_EARTH + 500e3, GM_EARTH)
config = ForceModelConfig(
    gravity_type="spherical_harmonics",
    gravity_model=GravityModel.from_type("JGM3"),
    gravity_degree=5, gravity_order=5,
    drag=True, density_model="harris_priester",
    srp=True, third_body_sun=True, third_body_moon=True,
)

dynamics_nd = nondim_orbit_dynamics(
    eop=zero_eop(), epoch_0=epoch_0,
    units=units, config=config,
)

# dynamics_nd(t_nd, state_nd) -> deriv_nd, suitable for any astrojax integrator.
```

## Choosing a UnitSystem

| Constructor | When to use |
|---|---|
| `UnitSystem.from_orbit(a, mu)` | Absolute Earth orbit; `mu_nd = 1`, `|r_nd| ~ 1`. |
| `UnitSystem.from_orbit_relative(a, mu, LU_rel)` | HCW / relative motion; chief defines time scale, `LU_rel` defines length scale. |
| `UnitSystem.earth_canonical()` | Generic Earth-centric problem; `LU = R_earth`. |
| `UnitSystem.lunar_canonical()` | Lunar orbits. |
| `UnitSystem.solar_canonical()` | Heliocentric problems. |
| `UnitSystem.from_scales(LU, TU, MU)` | Anything else. |

## Composing with `set_dtype`

`UnitSystem` is dtype-agnostic. Compose it with `set_dtype` to choose
your precision:

```python
from astrojax.config import set_dtype
import jax.numpy as jnp

set_dtype(jnp.bfloat16)
units = UnitSystem.from_orbit(7000e3, GM_EARTH)
# All subsequent astrojax calls run in bfloat16.
```

For ML training, the recommended combo is `set_dtype(jnp.bfloat16)`
plus a `UnitSystem.from_orbit(...)`. The validation suites in
`tests/nondim/` document the actual numerical error you can expect for
your chosen dtype and propagation duration.
