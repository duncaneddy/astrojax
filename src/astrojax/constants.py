"""
The `constants` module defines common mathematical and physical constants common in astrodynamics.
"""

from jax.numpy import pi as PI

# Mathematical Constants
"""
Constant to convert degrees to radians. Equal to 2pi/360. Units: *rad/deg*
"""
DEG2RAD = 2.0 * PI / 360.0

"""
Constant to convert radians to degrees. Equal to 360/2pi. Units: *deg/rad*
"""
RAD2DEG = 360.0 / (PI * 2.0)


"""
Constant to convert arcseconds to radians. Equal to 2pi/(360*3600). Units: *rad/as*
"""
AS2RAD = 2.0 * PI / 360.0 / 3600.0

"""
Constant to convert radians to arcseconds. Equal to (360*3600)/(2pi). Units: *as/rad*
"""
RAD2AS = 360.0 * 3600.0 / PI / 2.0

# Time Constants

"""
Offset between Julian Date and Modified Julian Date. Units: *days*
"""
JD_MJD_OFFSET = 2400000.5  # Offset between Julian Date and Modified Julian Date

"""
Modified Julian Date of the J2000.0 epoch (2000-01-01 12:00:00 TT). Units: *days*
"""
MJD2000 = 51544.5

# Physical Constants
"""
Speed of light in vacuum. Units: *m/s*

References:

1. D. Vallado, *Fundamentals of Astrodynamics and Applications (4th Ed.)*, 2010
"""
C_LIGHT = 299792458.0  # [m/s]Exact definition Vallado

"""
Astronomical Unit. Equal to the mean distance of the Earth from the sun.
TDB-compatible value. Units: *m*

References:

1. P. Gérard and B. Luzum, *IERS Technical Note 36*, 2010
"""
AU = 1.49597870700e11  # [m] Astronomical Unit IAU 2010

# Earth Constants
"""
Earth's equatorial radius. [m]

References:

1. GGM05s Gravity Model
"""
R_EARTH = 6.378136300e6  # [m] GGM05s Value

"""
Earth's semi-major axis as defined by the WGS84 geodetic system. [m]

References:

1. NIMA Technical Report TR8350.2
"""
WGS84_a = 6378137.0  # WGS-84 semi-major axis

"""
Earth's ellipsoidal flattening.  WGS84 Value.

References:

1. NIMA Technical Report TR8350.2
"""
WGS84_f = 1.0 / 298.257223563  # WGS-84 flattening

"""
Earth's Gravitational constant [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
Applications*, 2012.
"""
GM_EARTH = 3.986004415e14  # [m^3/s^2] GGM05s Value

"""
Earth's first eccentricity. WGS84 Value. [dimensionless]

References:

1. NIMA Technical Report TR8350.2
"""
ECC_EARTH = 8.1819190842622e-2  # [] First Eccentricity WGS84 Value

"""
Earth's first zonal harmonic. [dimensionless]

References:

1. GGM05s Gravity Model.
"""
J2_EARTH = 0.0010826358191967  # [] GGM05s value

"""
Earth axial rotation rate. [rad/s]

References:

1. D. Vallado, *Fundamentals of Astrodynamics and Applications (4th Ed.)*, p. 222, 2010
"""
OMEGA_EARTH = 7.292115146706979e-5  # [rad/s] Taken from Vallado 4th Ed page 222

# Sun Constants
"""
Gravitational constant of the Sun. [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
Applications*, 2012.
"""
GM_SUN = 132712440041.939400 * 1e9  # Gravitational constant of the Sun

"""
Nominal solar photospheric radius. [m]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
Applications*, 2012.
"""
R_SUN = 6.957 * 1e8  # Nominal solar radius corresponding to photospheric radius

"""
Nominal solar radiation pressure at 1 AU. [N/m^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
Applications*, 2012.
"""
P_SUN = 4.560e-6  # [N/m^2] (~1367 W/m^2) Solar radiation pressure at 1 AU

# Celestial Constants - from JPL DE430 Ephemerides
"""
Gravitational constant of the Moon. [m^3/s^2]

References:

1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
Applications*, 2012.
"""
GM_MOON = 4902.800066 * 1e9
