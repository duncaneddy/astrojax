"""Frame transformations.

This sub-module provides functions for converting between common
coordinate frames used in astrodynamics:

- **ECI-ECEF transformations**: converting between the Earth-Centered
  Inertial frame and the Earth-Centered Earth-Fixed frame using Earth
  rotation.
"""

from .eci_ecef import (
    earth_rotation,
    rotation_eci_to_ecef,
    rotation_ecef_to_eci,
    state_eci_to_ecef,
    state_ecef_to_eci,
)

__all__ = [
    "earth_rotation",
    "rotation_eci_to_ecef",
    "rotation_ecef_to_eci",
    "state_eci_to_ecef",
    "state_ecef_to_eci",
]
