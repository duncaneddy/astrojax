"""Keplerian orbital element ↔ ECI Cartesian state vector conversions.

Converts between osculating Keplerian orbital elements
``[a, e, i, RAAN, omega, M]`` and inertial Cartesian state vectors
``[x, y, z, vx, vy, vz]``.

Element ordering follows the Brahe convention:

| Index | Element                       | Units         |
|-------|-------------------------------|---------------|
| 0     | *a* — semi-major axis         | m             |
| 1     | *e* — eccentricity            | dimensionless |
| 2     | *i* — inclination             | rad           |
| 3     | *Ω* — right ascension (RAAN)  | rad           |
| 4     | *ω* — argument of perigee     | rad           |
| 5     | *M* — mean anomaly            | rad           |

All inputs and outputs use SI base units (metres, metres/second, radians).

References:
    1. O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods
       and Applications*, Springer, 2012, Sec. 2.2.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.constants import GM_EARTH
from astrojax.orbits import anomaly_eccentric_to_mean, anomaly_mean_to_eccentric


def state_koe_to_eci(
    x_oe: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Convert Keplerian orbital elements to an ECI Cartesian state vector.

    Solves Kepler's equation to obtain the eccentric anomaly, then
    constructs position and velocity via the perifocal P and Q vectors
    (Montenbruck & Gill Eq. 2.43–2.44).

    Args:
        x_oe: Orbital elements ``[a, e, i, RAAN, omega, M]``.
            Semi-major axis in *m*, angles in *rad* (or *deg* if
            ``use_degrees=True``).
        use_degrees: If ``True``, interpret angular elements as degrees.

    Returns:
        ECI state ``[x, y, z, vx, vy, vz]`` in *m* and *m/s*.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH
        from astrojax.coordinates import state_koe_to_eci
        oe = jnp.array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0])
        state = state_koe_to_eci(oe)
        state.shape
        ```

    References:
        O. Montenbruck and E. Gill, *Satellite Orbits*, 2012, Eq. 2.43–2.44.
    """
    x_oe = jnp.asarray(x_oe, dtype=get_dtype())

    a = x_oe[0]
    e = x_oe[1]
    i = x_oe[2]
    raan = x_oe[3]
    omega = x_oe[4]
    M = x_oe[5]

    if use_degrees:
        i = jnp.deg2rad(i)
        raan = jnp.deg2rad(raan)
        omega = jnp.deg2rad(omega)
        M = jnp.deg2rad(M)

    # Solve Kepler's equation: M -> E
    E = anomaly_mean_to_eccentric(M, e)

    # Perifocal unit vectors (Montenbruck & Gill Eq. 2.43)
    cos_o = jnp.cos(omega)
    sin_o = jnp.sin(omega)
    cos_R = jnp.cos(raan)
    sin_R = jnp.sin(raan)
    cos_i = jnp.cos(i)
    sin_i = jnp.sin(i)

    P = jnp.array(
        [
            cos_o * cos_R - sin_o * cos_i * sin_R,
            cos_o * sin_R + sin_o * cos_i * cos_R,
            sin_o * sin_i,
        ]
    )

    Q = jnp.array(
        [
            -sin_o * cos_R - cos_o * cos_i * sin_R,
            -sin_o * sin_R + cos_o * cos_i * cos_R,
            cos_o * sin_i,
        ]
    )

    # Position and velocity in the orbital plane
    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)
    sqrt_1me2 = jnp.sqrt(1.0 - e * e)

    r_vec = a * (cos_E - e) * P + a * sqrt_1me2 * sin_E * Q
    r_mag = jnp.linalg.norm(r_vec)
    v_vec = (jnp.sqrt(GM_EARTH * a) / r_mag) * (-sin_E * P + sqrt_1me2 * cos_E * Q)

    return jnp.concatenate([r_vec, v_vec])


def state_eci_to_koe(
    x_cart: ArrayLike,
    use_degrees: bool = False,
) -> Array:
    """Convert an ECI Cartesian state vector to Keplerian orbital elements.

    Derives the osculating elements from position and velocity using
    angular momentum, vis-viva, and the node/eccentricity vectors
    (Montenbruck & Gill Eq. 2.56–2.68).

    Args:
        x_cart: ECI state ``[x, y, z, vx, vy, vz]`` in *m* and *m/s*.
        use_degrees: If ``True``, return angular elements in degrees.

    Returns:
        Orbital elements ``[a, e, i, RAAN, omega, M]``.
            Semi-major axis in *m*, angles in *rad* (or *deg*).

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH, GM_EARTH
        from astrojax.coordinates import state_eci_to_koe
        sma = R_EARTH + 500e3
        v_circ = jnp.sqrt(GM_EARTH / sma)
        state = jnp.array([sma, 0.0, 0.0, 0.0, v_circ, 0.0])
        oe = state_eci_to_koe(state)
        oe.shape
        ```

    References:
        O. Montenbruck and E. Gill, *Satellite Orbits*, 2012, Eq. 2.56–2.68.
    """
    x_cart = jnp.asarray(x_cart, dtype=get_dtype())

    r = x_cart[:3]
    v = x_cart[3:6]

    r_mag = jnp.linalg.norm(r)
    v_mag = jnp.linalg.norm(v)

    # Angular momentum
    h = jnp.cross(r, v)
    h_mag = jnp.linalg.norm(h)
    W = h / h_mag

    # Inclination
    i = jnp.arctan2(jnp.sqrt(W[0] * W[0] + W[1] * W[1]), W[2])

    # Right ascension of ascending node
    raan = jnp.arctan2(W[0], -W[1])

    # Semi-latus rectum and semi-major axis (vis-viva)
    p = h_mag * h_mag / GM_EARTH
    a = 1.0 / (2.0 / r_mag - v_mag * v_mag / GM_EARTH)

    # Mean motion
    n = jnp.sqrt(GM_EARTH / jnp.abs(a) ** 3)

    # Eccentricity — clamp (1 - p/a) to prevent NaN from sqrt of negative
    ecc = jnp.sqrt(jnp.maximum(1.0 - p / a, 0.0))

    # Eccentric anomaly
    E = jnp.arctan2(jnp.dot(r, v) / (n * a * a), 1.0 - r_mag / a)

    # Mean anomaly via Kepler's equation
    M = anomaly_eccentric_to_mean(E, ecc)

    # Argument of latitude
    u = jnp.arctan2(r[2], -r[0] * W[1] + r[1] * W[0])

    # True anomaly
    sqrt_1me2 = jnp.sqrt(1.0 - ecc * ecc)
    nu = jnp.arctan2(sqrt_1me2 * jnp.sin(E), jnp.cos(E) - ecc)

    # Argument of perigee
    omega = u - nu

    # Normalize angles to [0, 2pi)
    two_pi = 2.0 * jnp.pi
    raan = jnp.mod(raan + two_pi, two_pi)
    omega = jnp.mod(omega + two_pi, two_pi)
    M = jnp.mod(M + two_pi, two_pi)

    if use_degrees:
        i = jnp.rad2deg(i)
        raan = jnp.rad2deg(raan)
        omega = jnp.rad2deg(omega)
        M = jnp.rad2deg(M)

    return jnp.array([a, ecc, i, raan, omega, M])
