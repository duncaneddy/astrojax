"""
SGP4/SDP4 orbit propagator implemented in JAX.

This module provides a JAX-native implementation of the SGP4 (Simplified General
Perturbations 4) and SDP4 (Simplified Deep-space Perturbations 4) orbit propagators
for propagating Two-Line Element (TLE) sets. The implementation supports JIT
compilation and ``vmap`` over time arrays.
"""

from astrojax.sgp4._constants import WGS72, WGS72OLD, WGS84, EarthGravity
from astrojax.sgp4._propagation import create_sgp4_propagator, sgp4_init, sgp4_propagate
from astrojax.sgp4._satellite import TLE
from astrojax.sgp4._tle import compute_checksum, parse_omm, parse_tle, validate_tle_line
from astrojax.sgp4._types import SGP4Elements

__all__ = [
    # Types
    "SGP4Elements",
    "EarthGravity",
    "TLE",
    # Constants
    "WGS72OLD",
    "WGS72",
    "WGS84",
    # TLE Parsing
    "parse_tle",
    "parse_omm",
    "compute_checksum",
    "validate_tle_line",
    # Propagation
    "sgp4_init",
    "sgp4_propagate",
    "create_sgp4_propagator",
]
