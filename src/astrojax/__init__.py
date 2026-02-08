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

from .config import set_dtype, get_dtype
from .epoch import Epoch

from .frames import (
    earth_rotation,
    rotation_eci_to_ecef,
    rotation_ecef_to_eci,
    state_eci_to_ecef,
    state_ecef_to_eci,
)

from .relative_motion import (
    rotation_rtn_to_eci,
    rotation_eci_to_rtn,
    state_eci_to_rtn,
    state_rtn_to_eci,
    hcw_derivative,
)

from .coordinates import (
    position_geocentric_to_ecef,
    position_ecef_to_geocentric,
    position_geodetic_to_ecef,
    position_ecef_to_geodetic,
    state_koe_to_eci,
    state_eci_to_koe,
)

from .orbits import (
    orbital_period,
    orbital_period_from_state,
    semimajor_axis_from_orbital_period,
    semimajor_axis,
    mean_motion,
    perigee_velocity,
    apogee_velocity,
    periapsis_distance,
    apoapsis_distance,
    perigee_altitude,
    apogee_altitude,
    sun_synchronous_inclination,
    geo_sma,
    anomaly_eccentric_to_mean,
    anomaly_mean_to_eccentric,
    anomaly_true_to_eccentric,
    anomaly_eccentric_to_true,
    anomaly_true_to_mean,
    anomaly_mean_to_true,
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
    # Config
    "set_dtype",
    "get_dtype",
    # Epoch
    "Epoch",
    # Frames
    "earth_rotation",
    "rotation_eci_to_ecef",
    "rotation_ecef_to_eci",
    "state_eci_to_ecef",
    "state_ecef_to_eci",
    # Relative Motion
    "rotation_rtn_to_eci",
    "rotation_eci_to_rtn",
    "state_eci_to_rtn",
    "state_rtn_to_eci",
    "hcw_derivative",
    # Coordinates
    "position_geocentric_to_ecef",
    "position_ecef_to_geocentric",
    "position_geodetic_to_ecef",
    "position_ecef_to_geodetic",
    "state_koe_to_eci",
    "state_eci_to_koe",
    # Orbits
    "orbital_period",
    "orbital_period_from_state",
    "semimajor_axis_from_orbital_period",
    "semimajor_axis",
    "mean_motion",
    "perigee_velocity",
    "apogee_velocity",
    "periapsis_distance",
    "apoapsis_distance",
    "perigee_altitude",
    "apogee_altitude",
    "sun_synchronous_inclination",
    "geo_sma",
    "anomaly_eccentric_to_mean",
    "anomaly_mean_to_eccentric",
    "anomaly_true_to_eccentric",
    "anomaly_eccentric_to_true",
    "anomaly_true_to_mean",
    "anomaly_mean_to_true",
]