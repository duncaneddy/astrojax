"""Orbit dynamics force models for perturbation modelling.

Provides gravitational, atmospheric, and radiation force models for
orbit propagation:

- **Ephemerides**: Low-precision Sun and Moon position (Montenbruck & Gill)
- **Planetary ephemerides**: Approximate heliocentric positions (JPL Table 1)
- **Gravity**: Point-mass gravitational acceleration
- **Third body**: Sun and Moon gravitational perturbations
- **Density**: Harris-Priester atmospheric density model
- **Drag**: Atmospheric drag acceleration
- **SRP**: Solar radiation pressure and eclipse shadow models
"""

from ._jpl_planetary_coefficients import (
    EMB_ID,
    JUPITER_ID,
    MARS_ID,
    MERCURY_ID,
    NEPTUNE_ID,
    SATURN_ID,
    URANUS_ID,
    VENUS_ID,
)
from .config import ForceModelConfig, SpacecraftParams
from .density import density_harris_priester
from .drag import accel_drag
from .ephemerides import moon_position, sun_position
from .factory import create_orbit_dynamics
from .gravity import (
    GravityModel,
    accel_gravity,
    accel_gravity_spherical_harmonics,
    accel_point_mass,
)
from .planetary_ephemerides import (
    emb_position_jpl_approx,
    jupiter_position_jpl_approx,
    mars_position_jpl_approx,
    mercury_position_jpl_approx,
    neptune_position_jpl_approx,
    planet_position_jpl_approx,
    saturn_position_jpl_approx,
    uranus_position_jpl_approx,
    venus_position_jpl_approx,
)
from .srp import accel_srp, eclipse_conical, eclipse_cylindrical
from .third_body import accel_third_body_moon, accel_third_body_sun

__all__ = [
    # Ephemerides
    "sun_position",
    "moon_position",
    # Planetary Ephemerides - Planet IDs
    "MERCURY_ID",
    "VENUS_ID",
    "EMB_ID",
    "MARS_ID",
    "JUPITER_ID",
    "SATURN_ID",
    "URANUS_ID",
    "NEPTUNE_ID",
    # Planetary Ephemerides - Position functions
    "mercury_position_jpl_approx",
    "venus_position_jpl_approx",
    "emb_position_jpl_approx",
    "mars_position_jpl_approx",
    "jupiter_position_jpl_approx",
    "saturn_position_jpl_approx",
    "uranus_position_jpl_approx",
    "neptune_position_jpl_approx",
    "planet_position_jpl_approx",
    # Gravity
    "accel_point_mass",
    "accel_gravity",
    "GravityModel",
    "accel_gravity_spherical_harmonics",
    # Third body
    "accel_third_body_sun",
    "accel_third_body_moon",
    # Density
    "density_harris_priester",
    # Drag
    "accel_drag",
    # SRP & Eclipse
    "accel_srp",
    "eclipse_conical",
    "eclipse_cylindrical",
    # Dynamics factory
    "ForceModelConfig",
    "SpacecraftParams",
    "create_orbit_dynamics",
]
