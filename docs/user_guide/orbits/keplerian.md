# Keplerian Orbit Functions

## Orbital Period and Semi-Major Axis

The orbital period $T$ is related to the semi-major axis $a$ by:

$$
T = 2\pi \sqrt{\frac{a^3}{\mu_\oplus}}
$$

```python
from astrojax.constants import R_EARTH
from astrojax.orbits import orbital_period, semimajor_axis_from_orbital_period

a = R_EARTH + 500e3          # 500 km LEO
T = orbital_period(a)         # ~5677 seconds (~94.6 minutes)

# Inverse: recover SMA from period
a_back = semimajor_axis_from_orbital_period(T)
```

You can also compute the period directly from an ECI state vector using
the vis-viva equation:

```python
import jax.numpy as jnp
from astrojax.constants import GM_EARTH
from astrojax.orbits import orbital_period_from_state

r = R_EARTH + 500e3
v = jnp.sqrt(GM_EARTH / r)
state = jnp.array([r, 0.0, 0.0, 0.0, v, 0.0])
T = orbital_period_from_state(state)
```

## Mean Motion

Mean motion $n$ is the angular rate of a Keplerian orbit:

$$
n = \sqrt{\frac{\mu_\oplus}{a^3}}
$$

```python
from astrojax.orbits import mean_motion, semimajor_axis

n = mean_motion(a)                         # rad/s
n_deg = mean_motion(a, use_degrees=True)   # deg/s

# Inverse: recover SMA from mean motion
a_back = semimajor_axis(n)
```

## Velocities at Apsides

For an elliptical orbit with semi-major axis $a$ and eccentricity $e$:

$$
v_{\text{perigee}} = \sqrt{\frac{\mu}{a}} \sqrt{\frac{1+e}{1-e}}, \quad
v_{\text{apogee}} = \sqrt{\frac{\mu}{a}} \sqrt{\frac{1-e}{1+e}}
$$

```python
from astrojax.orbits import perigee_velocity, apogee_velocity

vp = perigee_velocity(a, 0.01)  # m/s at perigee
va = apogee_velocity(a, 0.01)   # m/s at apogee
```

## Distances and Altitudes

```python
from astrojax.orbits import (
    periapsis_distance, apoapsis_distance,
    perigee_altitude, apogee_altitude,
)

rp = periapsis_distance(a, 0.01)   # a(1-e), metres
ra = apoapsis_distance(a, 0.01)    # a(1+e), metres
hp = perigee_altitude(a, 0.01)     # a(1-e) - R_EARTH, metres
ha = apogee_altitude(a, 0.01)      # a(1+e) - R_EARTH, metres
```
