# Ephemerides

The Sun and Moon positions use the low-precision analytical model from
Montenbruck & Gill (2012, p. 70-73).  Positions are returned in the
EME2000 (ECI) inertial frame in metres.

```python
from astrojax import Epoch
from astrojax.orbit_dynamics import sun_position, moon_position

epc = Epoch(2024, 6, 15, 12, 0, 0)
r_sun = sun_position(epc)    # shape (3,), units: m
r_moon = moon_position(epc)  # shape (3,), units: m
```

The analytical model provides ~0.01 degree accuracy for the Sun and
~0.3 degree for the Moon, sufficient for perturbation force modelling.
