"""
astrojax is a simple, minimal satellite orbit and attitude dynamics library implemented in JAX.
"""

from .constants import (
    DEG2RAD,
    RAD2DEG,
    AS2RAD,
    RAD2AS,
    JD_MJD_OFFSET,
    C_LIGHT,
    AU,
    R_EARTH,
    WGS84_a,
    WGS84_f,
    GM_EARTH,
    ECC_EARTH,
    J2_EARTH,
    OMEGA_EARTH,
    GM_SUN,
    R_SUN,
    P_SUN,
    GM_MOON
)

from .attitude_representations import (
    Rx,
    Ry,
    Rz
)

from .epoch import Epoch

from .relative_motion import (
    rotation_rtn_to_eci,
    rotation_eci_to_rtn,
    state_eci_to_rtn,
    state_rtn_to_eci,
    hcw_derivative,
)

__all__ = [
    # Constants
    "DEG2RAD",
    "RAD2DEG",
    "AS2RAD",
    "RAD2AS",
    "JD_MJD_OFFSET",
    "C_LIGHT",
    "AU",
    "R_EARTH",
    "WGS84_a",
    "WGS84_f",
    "GM_EARTH",
    "ECC_EARTH",
    "J2_EARTH",
    "OMEGA_EARTH",
    "GM_SUN",
    "R_SUN",
    "P_SUN",
    "GM_MOON",
    # Attitude Representations
    "Rx",
    "Ry",
    "Rz",
    # Epoch
    "Epoch",
    # Relative Motion
    "rotation_rtn_to_eci",
    "rotation_eci_to_rtn",
    "state_eci_to_rtn",
    "state_rtn_to_eci",
    "hcw_derivative",
]