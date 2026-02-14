# Atmospheric Density and Drag

## Atmospheric Density Models

Two atmospheric density models are available for computing drag
perturbations.

### Harris-Priester

The Harris-Priester model computes atmospheric density accounting for
diurnal variations caused by solar heating.  It is valid for altitudes
between 100 km and 1000 km and requires no external data files.

```python
import jax.numpy as jnp
from astrojax import Epoch
from astrojax.orbit_dynamics import density_harris_priester, sun_position

from astrojax.eop import zero_eop
from astrojax.frames.gcrf_itrf import bias_precession_nutation

epc = Epoch(2024, 6, 15, 12, 0, 0)
eop = zero_eop()

# Convert to true-of-date (TOD) frame for density computation
BPN = bias_precession_nutation(eop, epc)
r_eci = jnp.array([0.0, 0.0, -(6378e3 + 400e3)])  # 400 km altitude
r_tod = BPN @ r_eci
r_sun_tod = BPN @ sun_position(epc)
rho = density_harris_priester(r_tod, r_sun_tod)  # kg/m^3
```

### NRLMSISE-00

The NRLMSISE-00 model is an empirical atmospheric model covering
ground level through the thermosphere.  It computes number densities
of nine atmospheric species (He, O, N2, O2, Ar, H, N, anomalous O)
plus total mass density, based on solar activity and geomagnetic
conditions.

NRLMSISE-00 requires [space weather data](../space_weather.md)
(F10.7 solar flux and Ap geomagnetic indices):

```python
import jax.numpy as jnp
from astrojax import Epoch
from astrojax.space_weather import load_default_sw
from astrojax.orbit_dynamics import density_nrlmsise00
from astrojax.eop import zero_eop
from astrojax.frames import rotation_eci_to_ecef

epc = Epoch(2024, 6, 15, 12, 0, 0)
sw = load_default_sw()

# NRLMSISE-00 works in ECEF coordinates
eop = zero_eop()
R = rotation_eci_to_ecef(eop, epc)
r_eci = jnp.array([0.0, 0.0, -(6378e3 + 400e3)])
r_ecef = R @ r_eci

rho = density_nrlmsise00(sw, epc, r_ecef)  # kg/m^3
```

You can also pass geodetic coordinates directly using
`density_nrlmsise00_geod`:

```python
from astrojax.orbit_dynamics import density_nrlmsise00_geod

# [longitude_deg, latitude_deg, altitude_m]
geod = jnp.array([-74.0, 40.7, 400e3])
rho = density_nrlmsise00_geod(sw, epc, geod)  # kg/m^3
```

### Choosing a Density Model

| Feature | Harris-Priester | NRLMSISE-00 |
|---------|----------------|-------------|
| Altitude range | 100-1000 km | Ground to thermosphere |
| Species | Total density only | 9 species + total |
| Space weather data | Not required | Required (F10.7, Ap) |
| Coordinate frame | True-of-Date (TOD) | ECEF |
| Computational cost | Low | Higher |
| Use case | Quick LEO estimates | High-fidelity simulations |

## Atmospheric Drag

The drag model computes the non-conservative acceleration due to
atmospheric drag, accounting for the relative velocity between the
spacecraft and the co-rotating atmosphere:

```python
from astrojax.orbit_dynamics import accel_drag

x = jnp.array([6878e3, 0.0, 0.0, 0.0, 7500.0, 0.0])  # ECI state
T = jnp.eye(3)  # ECI to ECEF rotation (identity for simplicity)
a = accel_drag(x, rho, mass=100.0, area=1.0, cd=2.3, T=T)
```

The rotation matrix $T$ transforms from ECI to the ECEF (ITRF) frame.
For simplified models, the identity matrix can be used.
