"""Coordinate transformations.

This sub-module provides functions for converting between common
coordinate representations used in astrodynamics:

- **Geocentric**: spherical Earth model ``[lon, lat, alt]`` ↔ ECEF
- **Geodetic**: WGS84 ellipsoid model ``[lon, lat, alt]`` ↔ ECEF
- **Keplerian**: orbital elements ``[a, e, i, Ω, ω, M]`` ↔ ECI Cartesian
"""

from .geocentric import (
    position_ecef_to_geocentric,
    position_geocentric_to_ecef,
)
from .geodetic import (
    position_ecef_to_geodetic,
    position_geodetic_to_ecef,
)
from .keplerian import (
    state_eci_to_koe,
    state_koe_to_eci,
)

__all__ = [
    "position_geocentric_to_ecef",
    "position_ecef_to_geocentric",
    "position_geodetic_to_ecef",
    "position_ecef_to_geodetic",
    "state_koe_to_eci",
    "state_eci_to_koe",
]
