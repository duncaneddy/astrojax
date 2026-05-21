"""Pure-function converters between SI and nondimensional units.

All converters are JIT/vmap/grad compatible. They cast their output
to the active ``astrojax.config.get_dtype()`` so that downstream
computation stays in the configured precision.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.nondim.system import UnitSystem


def _as_array(x: ArrayLike) -> Array:
    return jnp.asarray(x, dtype=get_dtype())


def to_nondim_position(r_si: ArrayLike, units: UnitSystem) -> Array:
    """Convert position from SI [m] to nondim [LU]."""
    return _as_array(r_si) / units.LU


def from_nondim_position(r_nd: ArrayLike, units: UnitSystem) -> Array:
    """Convert position from nondim [LU] to SI [m]."""
    return _as_array(r_nd) * units.LU


def to_nondim_velocity(v_si: ArrayLike, units: UnitSystem) -> Array:
    """Convert velocity from SI [m/s] to nondim [LU/TU]."""
    return _as_array(v_si) / units.VU


def from_nondim_velocity(v_nd: ArrayLike, units: UnitSystem) -> Array:
    """Convert velocity from nondim [LU/TU] to SI [m/s]."""
    return _as_array(v_nd) * units.VU


def to_nondim_state(x_si: ArrayLike, units: UnitSystem) -> Array:
    """Convert a state vector [r, v] of shape (..., 6) from SI to nondim.

    Supports arbitrary leading batch dimensions — the trailing axis must
    be size 6 with positions in slots 0-2 and velocities in slots 3-5.
    """
    x = _as_array(x_si)
    r_nd = x[..., :3] / units.LU
    v_nd = x[..., 3:] / units.VU
    return jnp.concatenate([r_nd, v_nd], axis=-1)


def from_nondim_state(x_nd: ArrayLike, units: UnitSystem) -> Array:
    """Convert a state vector [r, v] of shape (..., 6) from nondim to SI.

    Supports arbitrary leading batch dimensions — the trailing axis must
    be size 6 with positions in slots 0-2 and velocities in slots 3-5.
    """
    x = _as_array(x_nd)
    r_si = x[..., :3] * units.LU
    v_si = x[..., 3:] * units.VU
    return jnp.concatenate([r_si, v_si], axis=-1)


def to_nondim_time(t_si: ArrayLike, units: UnitSystem) -> Array:
    """Convert a time offset from SI [s] to nondim [TU]."""
    return _as_array(t_si) / units.TU


def from_nondim_time(t_nd: ArrayLike, units: UnitSystem) -> Array:
    """Convert a time offset from nondim [TU] to SI [s]."""
    return _as_array(t_nd) * units.TU


def to_nondim_mu(mu_si: ArrayLike, units: UnitSystem) -> Array:
    """Convert μ from SI [m^3/s^2] to nondim [LU^3/TU^2]."""
    return _as_array(mu_si) / units.mu


def from_nondim_mu(mu_nd: ArrayLike, units: UnitSystem) -> Array:
    """Convert μ from nondim [LU^3/TU^2] to SI [m^3/s^2]."""
    return _as_array(mu_nd) * units.mu


def to_nondim_accel(a_si: ArrayLike, units: UnitSystem) -> Array:
    """Convert acceleration from SI [m/s^2] to nondim [LU/TU^2]."""
    return _as_array(a_si) / units.accel


def from_nondim_accel(a_nd: ArrayLike, units: UnitSystem) -> Array:
    """Convert acceleration from nondim [LU/TU^2] to SI [m/s^2]."""
    return _as_array(a_nd) * units.accel


def to_nondim_density(rho_si: ArrayLike, units: UnitSystem) -> Array:
    """Convert mass density from SI [kg/m^3] to nondim [MU/LU^3]."""
    return _as_array(rho_si) / units.density


def from_nondim_density(rho_nd: ArrayLike, units: UnitSystem) -> Array:
    """Convert mass density from nondim [MU/LU^3] to SI [kg/m^3]."""
    return _as_array(rho_nd) * units.density


def to_nondim_force(F_si: ArrayLike, units: UnitSystem) -> Array:
    """Convert force from SI [N] to nondim [MU·LU/TU^2]."""
    return _as_array(F_si) / units.force


def from_nondim_force(F_nd: ArrayLike, units: UnitSystem) -> Array:
    """Convert force from nondim [MU·LU/TU^2] to SI [N]."""
    return _as_array(F_nd) * units.force
