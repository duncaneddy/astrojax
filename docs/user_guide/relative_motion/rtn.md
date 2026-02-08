# Rotation Matrices and ECI-RTN Transforms

## Rotation Matrices

`rotation_rtn_to_eci` and `rotation_eci_to_rtn` compute the 3x3
direction-cosine matrix for a given chief ECI state:

```python
import jax.numpy as jnp
from astrojax.constants import R_EARTH, GM_EARTH
from astrojax.relative_motion import rotation_rtn_to_eci, rotation_eci_to_rtn

sma = R_EARTH + 500e3
v_circ = jnp.sqrt(GM_EARTH / sma)
chief = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])

R_rtn2eci = rotation_rtn_to_eci(chief)  # 3x3
R_eci2rtn = rotation_eci_to_rtn(chief)  # transpose of the above
```

## ECI to RTN State Transformation

`state_eci_to_rtn` transforms the absolute ECI states of a chief and
deputy into the deputy's relative state in the chief's RTN frame.  It
accounts for the Coriolis effect caused by the rotating frame:

$$
\dot{\boldsymbol{\rho}}_{\text{RTN}}
= R_{\text{ECI} \to \text{RTN}} \,
  (\mathbf{v}_{\text{dep}} - \mathbf{v}_{\text{chief}})
  - \boldsymbol{\omega} \times \boldsymbol{\rho}_{\text{RTN}}
$$

where $\boldsymbol{\omega} = [0, 0, \dot{f}]^T$ and the true-anomaly
rate is $\dot{f} = |\mathbf{r} \times \mathbf{v}| \;/\; |\mathbf{r}|^2$.

```python
from astrojax.relative_motion import state_eci_to_rtn, state_rtn_to_eci

deputy = chief + jnp.array([100.0, 200.0, 0.0, 0.0, 0.0, 0.0])
rel_rtn = state_eci_to_rtn(chief, deputy)

# Reconstruct the deputy's absolute ECI state
deputy_back = state_rtn_to_eci(chief, rel_rtn)
```

The inverse, `state_rtn_to_eci`, recovers the deputy's absolute ECI
state from the chief state and the relative RTN state.
