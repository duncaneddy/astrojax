"""Heliocentric ecliptic state computation for MPC asteroids.

Provides functions to propagate asteroid orbits from MPC Keplerian
elements and compute heliocentric ecliptic J2000 Cartesian state
vectors at arbitrary Julian dates.

All computational functions use JAX primitives and are compatible with
``jax.jit``, ``jax.vmap``, and ``jax.grad``.
"""

from __future__ import annotations

import jax.numpy as jnp
import polars as pl
from jax import Array
from jax.typing import ArrayLike

from astrojax.config import get_dtype
from astrojax.constants import AU, GM_SUN
from astrojax.orbits import anomaly_mean_to_eccentric


def asteroid_state_ecliptic(
    epoch_jd: ArrayLike,
    oe: ArrayLike,
    target_jd: ArrayLike,
    *,
    use_au: bool = False,
) -> Array:
    """Compute heliocentric ecliptic J2000 state for an asteroid.

    Given an epoch Julian Date (TT), orbital elements from the MPC
    catalog, and a target Julian Date (TT), propagates the mean anomaly
    forward and computes the full Cartesian state vector.

    Since MPC orbital elements are referenced to the ecliptic J2000
    frame, the output is directly in heliocentric ecliptic J2000
    coordinates.

    Args:
        epoch_jd: Epoch Julian Date (TT) at which the orbital elements
            are defined.
        oe: Orbital elements ``[a_AU, e, i_deg, node_deg, peri_deg, M_deg]``.
            Semi-major axis in AU, angles in degrees.
        target_jd: Target Julian Date (TT) at which to compute the state.
        use_au: If ``True``, return position in AU and velocity in
            AU/day.  Default ``False`` returns SI units (m, m/s).

    Returns:
        Heliocentric ecliptic J2000 state ``[x, y, z, vx, vy, vz]``.
        Units depend on *use_au*: SI (m, m/s) or (AU, AU/day).

    Examples:
        ```python
        import jax.numpy as jnp
        from astrojax.datasets import asteroid_state_ecliptic

        # Ceres-like elements (approximate)
        oe = jnp.array([2.769, 0.0755, 10.59, 80.31, 73.60, 77.37])
        state = asteroid_state_ecliptic(2460000.5, oe, 2460100.5)
        ```
    """
    dtype = get_dtype()
    epoch_jd = jnp.asarray(epoch_jd, dtype=dtype)
    oe = jnp.asarray(oe, dtype=dtype)
    target_jd = jnp.asarray(target_jd, dtype=dtype)

    # Unpack orbital elements
    a_au = oe[0]
    e = oe[1]
    i_deg = oe[2]
    node_deg = oe[3]
    peri_deg = oe[4]
    M_deg = oe[5]

    # Convert semi-major axis to metres
    a_m = a_au * AU

    # Convert angles to radians
    i = jnp.deg2rad(i_deg)
    node = jnp.deg2rad(node_deg)
    omega = jnp.deg2rad(peri_deg)
    M_epoch = jnp.deg2rad(M_deg)

    # Elapsed time in seconds
    dt = (target_jd - epoch_jd) * 86400.0

    # Mean motion [rad/s]
    n = jnp.sqrt(GM_SUN / a_m**3)

    # Propagated mean anomaly
    M_target = M_epoch + n * dt

    # Solve Kepler's equation: M -> E
    E = anomaly_mean_to_eccentric(M_target, e)

    # Perifocal unit vectors (ecliptic frame)
    cos_o = jnp.cos(omega)
    sin_o = jnp.sin(omega)
    cos_N = jnp.cos(node)
    sin_N = jnp.sin(node)
    cos_i = jnp.cos(i)
    sin_i = jnp.sin(i)

    P = jnp.array(
        [
            cos_o * cos_N - sin_o * cos_i * sin_N,
            cos_o * sin_N + sin_o * cos_i * cos_N,
            sin_o * sin_i,
        ]
    )

    Q = jnp.array(
        [
            -sin_o * cos_N - cos_o * cos_i * sin_N,
            -sin_o * sin_N + cos_o * cos_i * cos_N,
            cos_o * sin_i,
        ]
    )

    # Position and velocity in the orbital plane (SI)
    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)
    sqrt_1me2 = jnp.sqrt(1.0 - e * e)

    r_vec = a_m * (cos_E - e) * P + a_m * sqrt_1me2 * sin_E * Q
    r_mag = jnp.linalg.norm(r_vec)
    v_vec = (jnp.sqrt(GM_SUN * a_m) / r_mag) * (-sin_E * P + sqrt_1me2 * cos_E * Q)

    if use_au:
        # Convert position: m -> AU, velocity: m/s -> AU/day
        r_vec = r_vec / AU
        v_vec = v_vec / AU * 86400.0

    return jnp.concatenate([r_vec, v_vec])


def get_asteroid_ephemeris(df: pl.DataFrame, identifier: int | str) -> dict:
    """Look up an asteroid's orbital elements from the MPC DataFrame.

    Searches by number (if *identifier* is an ``int`` or numeric string)
    or by name (if *identifier* is a non-numeric string).

    Args:
        df: Polars DataFrame as returned by :func:`load_mpc_asteroids`.
        identifier: Asteroid number (int or numeric str) or name (str).

    Returns:
        Dictionary with keys: ``name``, ``number``, ``principal_desig``,
        ``epoch_jd``, ``a``, ``e``, ``i``, ``node``, ``peri``, ``M``,
        ``n``, ``H``.

    Raises:
        KeyError: If the asteroid is not found.

    Examples:
        ```python
        from astrojax.datasets import load_mpc_asteroids, get_asteroid_ephemeris
        df = load_mpc_asteroids()
        ceres = get_asteroid_ephemeris(df, 1)
        print(ceres["name"])  # "Ceres"
        ```
    """
    if isinstance(identifier, int) or (
        isinstance(identifier, str) and identifier.strip().isdigit()
    ):
        num_str = str(int(identifier)).strip()
        result = df.filter(pl.col("number").str.strip_chars() == num_str)
    else:
        result = df.filter(pl.col("name").str.strip_chars() == str(identifier).strip())

    if result.is_empty():
        raise KeyError(f"Asteroid not found: {identifier!r}")

    row = result.row(0, named=True)

    return {
        "name": row["name"],
        "number": row["number"],
        "principal_desig": row["principal_desig"],
        "epoch_jd": row["epoch_jd"],
        "a": row["a"],
        "e": row["e"],
        "i": row["i"],
        "node": row["node"],
        "peri": row["peri"],
        "M": row["M"],
        "n": row["n"],
        "H": row["H"],
    }
