"""
Earth gravity constants for the SGP4/SDP4 propagator.

Provides three standard gravity models: WGS72OLD, WGS72 (standard), and WGS84.
Values match the reference ``sgp4`` Python library exactly.
"""

from math import sqrt
from typing import NamedTuple


class EarthGravity(NamedTuple):
    """Earth gravity model constants for SGP4 propagation.

    Attributes:
        tumin: Time units per minute (1/xke).
        mu: Gravitational parameter [km^3/s^2].
        radiusearthkm: Earth equatorial radius [km].
        xke: Reciprocal of tumin (sqrt(GM) in SGP4 time units).
        j2: Second zonal harmonic.
        j3: Third zonal harmonic.
        j4: Fourth zonal harmonic.
        j3oj2: Ratio j3/j2.
    """

    tumin: float
    mu: float
    radiusearthkm: float
    xke: float
    j2: float
    j3: float
    j4: float
    j3oj2: float


# WGS 72 Old gravity constants
_mu_72old = 398600.79964
_re_72old = 6378.135
_xke_72old = 0.0743669161
_j2_72old = 0.001082616
_j3_72old = -0.00000253881
_j4_72old = -0.00000165597

WGS72OLD = EarthGravity(
    tumin=1.0 / _xke_72old,
    mu=_mu_72old,
    radiusearthkm=_re_72old,
    xke=_xke_72old,
    j2=_j2_72old,
    j3=_j3_72old,
    j4=_j4_72old,
    j3oj2=_j3_72old / _j2_72old,
)
"""WGS 72 Old gravity model (legacy)."""

# WGS 72 gravity constants (standard)
_mu_72 = 398600.8
_re_72 = 6378.135
_xke_72 = 60.0 / sqrt(_re_72**3 / _mu_72)
_j2_72 = 0.001082616
_j3_72 = -0.00000253881
_j4_72 = -0.00000165597

WGS72 = EarthGravity(
    tumin=1.0 / _xke_72,
    mu=_mu_72,
    radiusearthkm=_re_72,
    xke=_xke_72,
    j2=_j2_72,
    j3=_j3_72,
    j4=_j4_72,
    j3oj2=_j3_72 / _j2_72,
)
"""WGS 72 gravity model (standard for SGP4)."""

# WGS 84 gravity constants
_mu_84 = 398600.5
_re_84 = 6378.137
_xke_84 = 60.0 / sqrt(_re_84**3 / _mu_84)
_j2_84 = 0.00108262998905
_j3_84 = -0.00000253215306
_j4_84 = -0.00000161098761

WGS84 = EarthGravity(
    tumin=1.0 / _xke_84,
    mu=_mu_84,
    radiusearthkm=_re_84,
    xke=_xke_84,
    j2=_j2_84,
    j3=_j3_84,
    j4=_j4_84,
    j3oj2=_j3_84 / _j2_84,
)
"""WGS 84 gravity model."""

# Gravity constant lookup by name
GRAVITY_MODELS = {
    "wgs72old": WGS72OLD,
    "wgs72": WGS72,
    "wgs84": WGS84,
}
"""Mapping of gravity model names to ``EarthGravity`` instances."""
