"""Solar radiation pressure and eclipse shadow models.

Provides the acceleration due to solar radiation pressure (SRP) and
two shadow models — conical and cylindrical — for determining whether
a spacecraft is illuminated by the Sun.

All inputs and outputs use SI base units (metres, metres/second squared,
N/m^2).

References:
    1. O. Montenbruck and E. Gill, *Satellite Orbits: Models, Methods
       and Applications*, 2012.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.constants import AU, R_EARTH, R_SUN


def accel_srp(
    r_object: ArrayLike,
    r_sun: ArrayLike,
    mass: float,
    cr: float,
    area: float,
    p0: float,
) -> Array:
    """Acceleration due to solar radiation pressure.

    Args:
        r_object: Position of the object [m].  Shape ``(3,)`` or
            ``(6,)`` (only first 3 elements used).
        r_sun: Position of the Sun [m].  Shape ``(3,)``.
        mass: Spacecraft mass [kg].
        cr: Coefficient of reflectivity [dimensionless].
        area: Sun-facing cross-sectional area [m^2].
        p0: Solar radiation pressure at 1 AU [N/m^2].

    Returns:
        SRP acceleration [m/s^2], shape ``(3,)``.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import AU
        from astrojax.orbit_dynamics import accel_srp
        r = jnp.array([AU, 0.0, 0.0])
        r_sun = jnp.zeros(3)
        a = accel_srp(r, r_sun, 1.0, 1.0, 1.0, 4.5e-6)
        ```
    """
    _float = get_dtype()
    r = jnp.asarray(r_object, dtype=_float)[:3]
    r_s = jnp.asarray(r_sun, dtype=_float)

    d = r - r_s
    d_norm = jnp.linalg.norm(d)

    return d * _float(cr) * (_float(area) / _float(mass)) * _float(p0) * _float(AU) ** 2 / d_norm**3


def eclipse_conical(r_object: ArrayLike, r_sun: ArrayLike) -> Array:
    """Illumination fraction using the conical shadow model.

    Computes the fraction of the Sun's disk visible to the spacecraft,
    accounting for partial eclipses (penumbra).

    Args:
        r_object: Position of the object in ECI [m].  Shape ``(3,)``
            or ``(6,)`` (only first 3 elements used).
        r_sun: Position of the Sun in ECI [m].  Shape ``(3,)``.

    Returns:
        Illumination fraction (scalar).
            0.0 = full shadow, 1.0 = full illumination.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH, AU
        from astrojax.orbit_dynamics import eclipse_conical
        r_sun = jnp.array([AU, 0.0, 0.0])
        r_shadow = jnp.array([-R_EARTH - 100e3, 0.0, 0.0])
        nu = eclipse_conical(r_shadow, r_sun)
        ```
    """
    _float = get_dtype()
    r = jnp.asarray(r_object, dtype=_float)[:3]
    r_s = jnp.asarray(r_sun, dtype=_float)

    r_norm = jnp.linalg.norm(r)
    d = r_s - r
    d_norm = jnp.linalg.norm(d)

    # Apparent angular radii
    a = jnp.arcsin(_float(R_SUN) / d_norm)  # Sun
    b = jnp.arcsin(_float(R_EARTH) / r_norm)  # Earth

    # Angular separation between Sun and anti-nadir
    c = jnp.arccos(-jnp.dot(r, d) / (r_norm * d_norm))

    # Partial eclipse (penumbra) geometry
    x = (c**2 + a**2 - b**2) / (_float(2.0) * c)
    y = jnp.sqrt(jnp.maximum(a**2 - x**2, _float(0.0)))
    area_overlap = a**2 * jnp.arccos(x / a) + b**2 * jnp.arccos((c - x) / b) - c * y
    nu_partial = _float(1.0) - area_overlap / (jnp.pi * a**2)

    # Select condition using jnp.where for JAX traceability
    is_partial = (jnp.abs(a - b) < c) & (c < (a + b))
    is_full_illumination = (a + b) <= c

    return jnp.where(
        is_full_illumination,
        _float(1.0),
        jnp.where(is_partial, nu_partial, _float(0.0)),
    )


def eclipse_cylindrical(r_object: ArrayLike, r_sun: ArrayLike) -> Array:
    """Illumination fraction using the cylindrical shadow model.

    A simpler shadow model that treats Earth's shadow as a cylinder
    aligned with the Sun direction.  Returns 0.0 (shadow) or 1.0
    (illuminated) with no penumbra.

    Args:
        r_object: Position of the object in ECI [m].  Shape ``(3,)``
            or ``(6,)`` (only first 3 elements used).
        r_sun: Position of the Sun in ECI [m].  Shape ``(3,)``.

    Returns:
        Illumination fraction (scalar), 0.0 or 1.0.

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.constants import R_EARTH, AU
        from astrojax.orbit_dynamics import eclipse_cylindrical
        r_sun = jnp.array([AU, 0.0, 0.0])
        r_shadow = jnp.array([-R_EARTH - 100e3, 0.0, 0.0])
        nu = eclipse_cylindrical(r_shadow, r_sun)
        ```
    """
    _float = get_dtype()
    r = jnp.asarray(r_object, dtype=_float)[:3]
    r_s = jnp.asarray(r_sun, dtype=_float)

    # Unit vector towards the Sun
    e_sun = r_s / jnp.linalg.norm(r_s)

    # Projection of spacecraft position onto Sun direction
    r_proj = jnp.dot(r, e_sun)

    # Perpendicular distance from shadow axis
    r_perp = jnp.linalg.norm(r - r_proj * e_sun)

    # Illuminated if facing the Sun (projection >= 0) or outside Earth's radius
    is_illuminated = (r_proj >= _float(1.0)) | (r_perp > _float(R_EARTH))
    return jnp.where(is_illuminated, _float(1.0), _float(0.0))
