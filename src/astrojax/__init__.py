"""
astrojax is a simple, minimal satellite orbit and attitude dynamics library implemented in JAX.
"""

from .constants import (
    DEG2RAD,
    RAD2DEG,
    AS2RAD,
    RAD2AS,
    JD_MJD_OFFSET,
    MJD2000,
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

from .orbit_dynamics import (
    sun_position,
    moon_position,
    accel_point_mass,
    accel_gravity,
    GravityModel,
    accel_gravity_spherical_harmonics,
    accel_third_body_sun,
    accel_third_body_moon,
    density_harris_priester,
    accel_drag,
    accel_srp,
    eclipse_conical,
    eclipse_cylindrical,
    ForceModelConfig,
    SpacecraftParams,
    create_orbit_dynamics,
)

from .integrators import (
    StepResult,
    AdaptiveConfig,
    rk4_step,
    rkf45_step,
    dp54_step,
    rkn1210_step,
)

__all__ = [
    # Constants
    "DEG2RAD",
    "RAD2DEG",
    "AS2RAD",
    "RAD2AS",
    "JD_MJD_OFFSET",
    "MJD2000",
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
    # Orbit Dynamics
    "sun_position",
    "moon_position",
    "accel_point_mass",
    "accel_gravity",
    "GravityModel",
    "accel_gravity_spherical_harmonics",
    "accel_third_body_sun",
    "accel_third_body_moon",
    "density_harris_priester",
    "accel_drag",
    "accel_srp",
    "eclipse_conical",
    "eclipse_cylindrical",
    "ForceModelConfig",
    "SpacecraftParams",
    "create_orbit_dynamics",
    # Integrators
    "StepResult",
    "AdaptiveConfig",
    "rk4_step",
    "rkf45_step",
    "dp54_step",
    "rkn1210_step",
]