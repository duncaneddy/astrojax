"""Frame transformations.

This sub-module provides functions for converting between common
coordinate frames used in astrodynamics:

- **GCRF-ITRF transformations**: converting between the Geocentric Celestial
  Reference Frame and the International Terrestrial Reference Frame using the
  full IAU 2006/2000A CIO-based model (bias-precession-nutation, Earth rotation
  angle, and polar motion).
- **TEME transformations**: converting SGP4 output (TEME frame) to PEF, ITRF,
  and GCRF via GMST rotation and polar motion.
- **ECI-ECEF aliases**: backward-compatible names mapping to GCRF/ITRF functions.
"""

from .gcrf_itrf import (
    bias_precession_nutation,
    earth_rotation,
    earth_rotation_angle,
    polar_motion,
    rotation_ecef_to_eci,
    rotation_eci_to_ecef,
    rotation_gcrf_to_itrf,
    rotation_itrf_to_gcrf,
    state_ecef_to_eci,
    state_eci_to_ecef,
    state_gcrf_to_itrf,
    state_itrf_to_gcrf,
)
from .teme import (
    rotation_itrf_to_teme,
    rotation_pef_to_teme,
    rotation_teme_to_itrf,
    rotation_teme_to_pef,
    state_gcrf_to_teme,
    state_itrf_to_teme,
    state_pef_to_teme,
    state_teme_to_gcrf,
    state_teme_to_itrf,
    state_teme_to_pef,
)

__all__ = [
    # GCRF/ITRF (primary API)
    "bias_precession_nutation",
    "earth_rotation_angle",
    "polar_motion",
    "rotation_gcrf_to_itrf",
    "rotation_itrf_to_gcrf",
    "state_gcrf_to_itrf",
    "state_itrf_to_gcrf",
    # TEME transformations
    "rotation_teme_to_pef",
    "rotation_pef_to_teme",
    "rotation_teme_to_itrf",
    "rotation_itrf_to_teme",
    "state_teme_to_pef",
    "state_pef_to_teme",
    "state_teme_to_itrf",
    "state_itrf_to_teme",
    "state_teme_to_gcrf",
    "state_gcrf_to_teme",
    # ECI/ECEF aliases
    "earth_rotation",
    "rotation_eci_to_ecef",
    "rotation_ecef_to_eci",
    "state_eci_to_ecef",
    "state_ecef_to_eci",
]
