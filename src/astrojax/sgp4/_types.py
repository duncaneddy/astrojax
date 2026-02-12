"""
Data types for the SGP4/SDP4 propagator.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SGP4Elements:
    """Parsed orbital elements from a TLE or OMM record.

    This is a plain Python dataclass (not a JAX pytree) holding the parsed
    orbital elements, epoch information, and metadata extracted from a TLE
    or OMM record. All angular values are stored in their original units
    as parsed (degrees for user-facing, radians for SGP4-internal).

    Attributes:
        satnum_str: Satellite catalog number as a string (e.g. ``'25544'``).
        classification: Classification character (``'U'``, ``'C'``, or ``'S'``).
        intldesg: International designator (e.g. ``'98067A'``).
        epochyr: Two-digit epoch year (0-99).
        epochdays: Day of year with fractional day.
        ndot: First derivative of mean motion divided by 2 [rad/min^2].
        nddot: Second derivative of mean motion divided by 6 [rad/min^3].
        bstar: B* drag coefficient [1/earth_radii].
        ephtype: Ephemeris type (typically 0).
        elnum: Element set number.
        revnum: Revolution number at epoch.
        inclo: Inclination [rad].
        nodeo: Right ascension of ascending node [rad].
        ecco: Eccentricity [dimensionless].
        argpo: Argument of perigee [rad].
        mo: Mean anomaly [rad].
        no_kozai: Mean motion (Kozai) [rad/min].
        jdsatepoch: Julian date of epoch (whole days).
        jdsatepochF: Julian date of epoch (fractional day).
    """

    satnum_str: str
    classification: str
    intldesg: str
    epochyr: int
    epochdays: float
    ndot: float
    nddot: float
    bstar: float
    ephtype: int
    elnum: int
    revnum: int
    inclo: float
    nodeo: float
    ecco: float
    argpo: float
    mo: float
    no_kozai: float
    jdsatepoch: float
    jdsatepochF: float
