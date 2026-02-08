"""Harris-Priester atmospheric density model.

Computes the atmospheric density using the modified Harris-Priester
model, which accounts for diurnal density variations caused by solar
heating.  Valid for altitudes between 100 km and 1000 km.

All inputs and outputs use SI base units (metres, kg/m^3).

References:
    1. O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods
       and Applications*, 2012.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.coordinates import position_ecef_to_geodetic

# Harris-Priester model constants
_HP_UPPER_LIMIT = 1000.0   # Upper height limit [km]
_HP_LOWER_LIMIT = 100.0    # Lower height limit [km]
_HP_RA_LAG = 0.523599      # Right ascension lag [rad] (~30 deg)
_HP_N_PRM = 3.0            # Harris-Priester exponent (low inclination)

# Height table [km] — 50 entries
_HP_H = jnp.array([
    100.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
    210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
    320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0,
    520.0, 540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0,
    720.0, 740.0, 760.0, 780.0, 800.0, 840.0, 880.0, 920.0, 960.0, 1000.0,
], )

# Minimum density [g/km^3]
_HP_C_MIN = jnp.array([
    4.974e+05, 2.490e+04, 8.377e+03, 3.899e+03, 2.122e+03, 1.263e+03,
    8.008e+02, 5.283e+02, 3.617e+02, 2.557e+02, 1.839e+02, 1.341e+02,
    9.949e+01, 7.488e+01, 5.709e+01, 4.403e+01, 3.430e+01, 2.697e+01,
    2.139e+01, 1.708e+01, 1.099e+01, 7.214e+00, 4.824e+00, 3.274e+00,
    2.249e+00, 1.558e+00, 1.091e+00, 7.701e-01, 5.474e-01, 3.916e-01,
    2.819e-01, 2.042e-01, 1.488e-01, 1.092e-01, 8.070e-02, 6.012e-02,
    4.519e-02, 3.430e-02, 2.632e-02, 2.043e-02, 1.607e-02, 1.281e-02,
    1.036e-02, 8.496e-03, 7.069e-03, 4.680e-03, 3.200e-03, 2.210e-03,
    1.560e-03, 1.150e-03,
], )

# Maximum density [g/km^3]
_HP_C_MAX = jnp.array([
    4.974e+05, 2.490e+04, 8.710e+03, 4.059e+03, 2.215e+03, 1.344e+03,
    8.758e+02, 6.010e+02, 4.297e+02, 3.162e+02, 2.396e+02, 1.853e+02,
    1.455e+02, 1.157e+02, 9.308e+01, 7.555e+01, 6.182e+01, 5.095e+01,
    4.226e+01, 3.526e+01, 2.511e+01, 1.819e+01, 1.337e+01, 9.955e+00,
    7.492e+00, 5.684e+00, 4.355e+00, 3.362e+00, 2.612e+00, 2.042e+00,
    1.605e+00, 1.267e+00, 1.005e+00, 7.997e-01, 6.390e-01, 5.123e-01,
    4.121e-01, 3.325e-01, 2.691e-01, 2.185e-01, 1.779e-01, 1.452e-01,
    1.190e-01, 9.776e-02, 8.059e-02, 5.741e-02, 4.210e-02, 3.130e-02,
    2.360e-02, 1.810e-02,
], )


def density_harris_priester(
    r_ecef: ArrayLike,
    r_sun: ArrayLike,
) -> Array:
    """Atmospheric density using the Harris-Priester model.

    Computes density accounting for diurnal bulge caused by solar
    heating.  Returns zero outside the valid 100-1000 km altitude range.

    Args:
        r_ecef: Satellite position in the ECEF (or TOD) frame [m].
            Shape ``(3,)``.
        r_sun: Sun position vector [m].  Shape ``(3,)``.  Used only for
            computing the right ascension and declination of the Sun.

    Returns:
        Atmospheric density [kg/m^3] (scalar).

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.orbit_dynamics import density_harris_priester
        r = jnp.array([0.0, 0.0, -6466752.314])
        r_sun = jnp.array([24622331959.58, -133060326832.922, -57688711921.833])
        rho = density_harris_priester(r, r_sun)
        ```
    """
    _float = get_dtype()
    r_ecef = jnp.asarray(r_ecef, dtype=_float)
    r_sun = jnp.asarray(r_sun, dtype=_float)

    # Geodetic altitude
    geod = position_ecef_to_geodetic(r_ecef)  # [lon_rad, lat_rad, alt_m]
    height = geod[2] / _float(1.0e3)  # altitude in [km]

    # Sun right ascension and declination
    ra_sun = jnp.arctan2(r_sun[1], r_sun[0])
    dec_sun = jnp.arctan2(
        r_sun[2], jnp.sqrt(r_sun[0] ** 2 + r_sun[1] ** 2)
    )

    # Unit vector towards diurnal bulge apex
    c_dec = jnp.cos(dec_sun)
    u = jnp.array([
        c_dec * jnp.cos(ra_sun + _float(_HP_RA_LAG)),
        c_dec * jnp.sin(ra_sun + _float(_HP_RA_LAG)),
        jnp.sin(dec_sun),
    ])

    # Cosine of half angle between satellite and apex
    c_psi2 = _float(0.5) + _float(0.5) * jnp.dot(r_ecef, u) / jnp.linalg.norm(r_ecef)

    # Height bracket search using searchsorted (JIT-compatible)
    # searchsorted returns the index where height would be inserted to keep
    # the array sorted.  We subtract 1 to get the left bracket index.
    ih = jnp.searchsorted(_HP_H, height, side="right") - 1
    ih = jnp.clip(ih, 0, 48)  # clamp to valid range [0, N-2]

    # Exponential scale heights
    h_min = (_HP_H[ih] - _HP_H[ih + 1]) / jnp.log(_HP_C_MIN[ih + 1] / _HP_C_MIN[ih])
    h_max = (_HP_H[ih] - _HP_H[ih + 1]) / jnp.log(_HP_C_MAX[ih + 1] / _HP_C_MAX[ih])

    # Interpolated densities [g/km^3]
    d_min = _HP_C_MIN[ih] * jnp.exp((_HP_H[ih] - height) / h_min)
    d_max = _HP_C_MAX[ih] * jnp.exp((_HP_H[ih] - height) / h_max)

    # Density with diurnal variation [g/km^3]
    density = d_min + (d_max - d_min) * c_psi2 ** _HP_N_PRM

    # Convert from g/km^3 to kg/m^3
    density = density * _float(1.0e-12)

    # Return 0 outside valid altitude range
    in_range = (height > _float(_HP_LOWER_LIMIT)) & (height < _float(_HP_UPPER_LIMIT))
    return jnp.where(in_range, density, _float(0.0))
