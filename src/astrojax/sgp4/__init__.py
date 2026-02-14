"""
SGP4/SDP4 orbit propagator implemented in JAX.

This module provides a JAX-native implementation of the SGP4 (Simplified General
Perturbations 4) and SDP4 (Simplified Deep-space Perturbations 4) orbit propagators
for propagating Two-Line Element (TLE) sets. The implementation supports JIT
compilation and ``vmap`` over time arrays.
"""

from astrojax.sgp4._constants import WGS72, WGS72OLD, WGS84, EarthGravity
from astrojax.sgp4._propagation import (
    create_sgp4_propagator,
    create_sgp4_propagator_from_elements,
    create_sgp4_propagator_from_gp_record,
    create_sgp4_propagator_from_omm,
    gp_record_to_array,
    omm_to_array,
    sgp4_init,
    sgp4_init_jax,
    sgp4_propagate,
    sgp4_propagate_unified,
)
from astrojax.sgp4._satellite import TLE
from astrojax.sgp4._tle import compute_checksum, parse_omm, parse_tle, validate_tle_line
from astrojax.sgp4._types import SGP4Elements, elements_to_array

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
    # Propagation (Python-time init)
    "sgp4_init",
    "sgp4_propagate",
    "create_sgp4_propagator",
    "create_sgp4_propagator_from_elements",
    "create_sgp4_propagator_from_omm",
    "create_sgp4_propagator_from_gp_record",
    # JIT-compilable init and unified propagation
    "sgp4_init_jax",
    "sgp4_propagate_unified",
    "elements_to_array",
    "omm_to_array",
    "gp_record_to_array",
]
