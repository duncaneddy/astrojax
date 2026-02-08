# Hill-Clohessy-Wiltshire Dynamics

For a chief satellite on a *circular* orbit with mean motion
$n = \sqrt{\mu / a^3}$, the linearised equations of relative motion
(HCW model) are:

$$
\ddot{x} = 3n^2 x + 2n\dot{y}, \quad
\ddot{y} = -2n\dot{x}, \quad
\ddot{z} = -n^2 z.
$$

`hcw_derivative` returns the 6-element state derivative
$[\dot{x}, \dot{y}, \dot{z}, \ddot{x}, \ddot{y}, \ddot{z}]$ and is
designed to plug directly into a numerical integrator:

```python
import jax.numpy as jnp
from astrojax.constants import R_EARTH, GM_EARTH
from astrojax.relative_motion import hcw_derivative

sma = R_EARTH + 500e3
n = jnp.sqrt(GM_EARTH / sma**3)
state = jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])

deriv = hcw_derivative(state, n)
```

Because it is a pure JAX function, `hcw_derivative` is compatible with
`jax.jit`, `jax.vmap`, and `jax.grad`, making it suitable for batched
simulation and differentiable control.
