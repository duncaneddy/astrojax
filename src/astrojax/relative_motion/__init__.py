"""Relative motion transformations and dynamics.

This sub-module provides functions for:

- **ECI-RTN frame transformations**: converting between inertial and
  rotating local-vertical/local-horizontal frames attached to a chief
  satellite.
- **OE-ROE transformations**: converting between Keplerian orbital
  elements and quasi-nonsingular Relative Orbital Elements.
- **ECI-ROE transformations**: direct conversion between ECI state
  vectors and Relative Orbital Elements.
- **Hill-Clohessy-Wiltshire dynamics**: linearised relative motion
  equations for proximity operations near a circular reference orbit.
"""

from .eci_rtn import (
    rotation_rtn_to_eci,
    rotation_eci_to_rtn,
    state_eci_to_rtn,
    state_rtn_to_eci,
)
from .oe_roe import (
    state_oe_to_roe,
    state_roe_to_oe,
)
from .eci_roe import (
    state_eci_to_roe,
    state_roe_to_eci,
)
from .hcw_dynamics import hcw_derivative

__all__ = [
    "rotation_rtn_to_eci",
    "rotation_eci_to_rtn",
    "state_eci_to_rtn",
    "state_rtn_to_eci",
    "state_oe_to_roe",
    "state_roe_to_oe",
    "state_eci_to_roe",
    "state_roe_to_eci",
    "hcw_derivative",
]
