# Special Orbits

## Sun-Synchronous Orbit

The inclination required for a Sun-synchronous orbit is determined by
the J2 perturbation on the RAAN precession rate:

```python
from astrojax.constants import R_EARTH
from astrojax.orbits import sun_synchronous_inclination

inc = sun_synchronous_inclination(R_EARTH + 500e3, 0.001, use_degrees=True)
# ~97.4 degrees
```

## Geostationary Orbit

```python
from astrojax.orbits import geo_sma

a_geo = geo_sma()  # ~42164 km
```
