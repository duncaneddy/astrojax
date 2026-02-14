# Composing Force Models

The individual force model functions can be composed into a single
dynamics function using `create_orbit_dynamics`.  This factory takes
EOP data, a reference epoch, and a configuration object, and returns a
`dynamics(t, state) -> derivative` closure compatible with all astrojax
integrators.

## Quick Start: Two-Body

```python
from astrojax import Epoch, create_orbit_dynamics
from astrojax.eop import zero_eop
from astrojax.integrators import rk4_step
import jax.numpy as jnp

epoch_0 = Epoch(2024, 6, 15, 12, 0, 0)
dynamics = create_orbit_dynamics(zero_eop(), epoch_0)  # default: point-mass gravity

x0 = jnp.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
result = rk4_step(dynamics, 0.0, x0, 60.0)
```

## LEO with Perturbations

Use `ForceModelConfig` to enable perturbation forces:

```python
from astrojax import ForceModelConfig, SpacecraftParams, GravityModel

config = ForceModelConfig(
    gravity_type="spherical_harmonics",
    gravity_model=GravityModel.from_type("JGM3"),
    gravity_degree=20,
    gravity_order=20,
    drag=True,
    srp=True,
    third_body_sun=True,
    third_body_moon=True,
    spacecraft=SpacecraftParams(mass=500.0, cd=2.2, cr=1.3),
)

dynamics = create_orbit_dynamics(zero_eop(), epoch_0, config)
```

Or use a preset for common scenarios:

```python
config = ForceModelConfig.leo_default()   # 20x20 SH + drag + SRP + Sun/Moon
config = ForceModelConfig.geo_default()   # 8x8 SH + SRP + Sun/Moon (no drag)
config = ForceModelConfig.two_body()      # point-mass only
```

## Using NRLMSISE-00 Density Model

To use the NRLMSISE-00 atmospheric density model instead of the
default Harris-Priester, set `density_model="nrlmsise00"` and pass
[space weather data](../space_weather.md) to `create_orbit_dynamics`:

```python
from astrojax import Epoch, ForceModelConfig, create_orbit_dynamics
from astrojax.eop import zero_eop
from astrojax.space_weather import load_default_sw
import jax.numpy as jnp

epoch_0 = Epoch(2024, 6, 15, 12, 0, 0)
sw = load_default_sw()

config = ForceModelConfig.leo_default(density_model="nrlmsise00")
dynamics = create_orbit_dynamics(zero_eop(), epoch_0, config, space_weather=sw)

x0 = jnp.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
```

The `leo_default()` preset accepts a `density_model` parameter that
can be either `"harris_priester"` (default) or `"nrlmsise00"`.

!!! note "Space weather is required"
    When using `density_model="nrlmsise00"`, the `space_weather`
    argument to `create_orbit_dynamics` must be provided.  A
    `ValueError` is raised if it is omitted.

## Multi-Step Propagation

Use `jax.lax.scan` for efficient multi-step propagation:

```python
import jax

dt = 60.0
def scan_step(carry, _):
    t, state = carry
    result = rk4_step(dynamics, t, state, dt)
    return (t + dt, result.state), result.state

(_, _), trajectory = jax.lax.scan(scan_step, (0.0, x0), None, length=100)
```

## Adding Thrust

Use the integrator `control` parameter to add thrust or other external
forces:

```python
def thrust(t, x):
    v = x[3:]
    v_hat = v / jnp.linalg.norm(v)
    return jnp.concatenate([jnp.zeros(3), 1e-3 * v_hat])

result = rk4_step(dynamics, 0.0, x0, 60.0, control=thrust)
```

The effective derivative at each integrator stage is
`dynamics(t, x) + control(t, x)`.

## Spacecraft Parameters

`SpacecraftParams` holds physical properties used by drag and SRP:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mass` | 1000.0 | Spacecraft mass [kg] |
| `drag_area` | 10.0 | Wind-facing cross-sectional area [m^2] |
| `srp_area` | 10.0 | Sun-facing cross-sectional area [m^2] |
| `cd` | 2.2 | Coefficient of drag |
| `cr` | 1.3 | Coefficient of reflectivity |
