"""Relative motion transformations and dynamics.

This sub-module provides functions for:

- **ECI-RTN frame transformations**: converting between inertial and
  rotating local-vertical/local-horizontal frames attached to a chief
  satellite.
- **Hill-Clohessy-Wiltshire dynamics**: linearised relative motion
  equations for proximity operations near a circular reference orbit.
"""

from .eci_rtn import (
    rotation_rtn_to_eci,
    rotation_eci_to_rtn,
    state_eci_to_rtn,
    state_rtn_to_eci,
)
from .hcw_dynamics import hcw_derivative

__all__ = [
    "rotation_rtn_to_eci",
    "rotation_eci_to_rtn",
    "state_eci_to_rtn",
    "state_rtn_to_eci",
    "hcw_derivative",
]
