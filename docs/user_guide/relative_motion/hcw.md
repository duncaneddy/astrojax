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

## State Transition Matrix

Because the HCW equations are linear and time-invariant, the system
$\dot{\mathbf{x}} = A\mathbf{x}$ has a closed-form state transition
matrix $\Phi(t) = e^{At}$ that maps an initial state directly to the
state at elapsed time $t$:

$$
\mathbf{x}(t) = \Phi(t)\,\mathbf{x}_0
$$

Writing $c = \cos(nt)$ and $s = \sin(nt)$, the 6 $\times$ 6 matrix is:

$$
\Phi(t) = \begin{bmatrix}
4-3c        & 0 & 0 & s/n          & 2(1-c)/n      & 0   \\
6(s-nt)     & 1 & 0 & -2(1-c)/n    & (4s-3nt)/n    & 0   \\
0           & 0 & c & 0            & 0             & s/n \\
3ns         & 0 & 0 & c            & 2s            & 0   \\
-6n(1-c)    & 0 & 0 & -2s          & 4c-3          & 0   \\
0           & 0 &-ns & 0            & 0             & c
\end{bmatrix}
$$

`hcw_stm` returns this matrix and can be used for fast, exact
propagation without numerical integration:

```python
import jax.numpy as jnp
from astrojax.constants import R_EARTH, GM_EARTH
from astrojax.relative_motion import hcw_stm

sma = R_EARTH + 500e3
n = jnp.sqrt(GM_EARTH / sma**3)
state0 = jnp.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])

state_10min = hcw_stm(600.0, n) @ state0
```

The STM satisfies the **semigroup (composition) property**:
$\Phi(t_1 + t_2) = \Phi(t_2)\,\Phi(t_1)$. This means you can
chain shorter propagation steps without accumulating integration error.
It is also volume-preserving ($\det\Phi = 1$), as expected for a
Hamiltonian system.
