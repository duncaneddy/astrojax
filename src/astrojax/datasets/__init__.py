"""Asteroid and small-body datasets for astrojax.

Provides access to the Minor Planet Center (MPC) asteroid orbit catalog,
including download/caching, Polars DataFrame loading, per-asteroid
lookups, and heliocentric ecliptic state vector computation.

Typical usage::

    from astrojax.datasets import load_mpc_asteroids, get_asteroid_ephemeris, asteroid_state_ecliptic
    import jax.numpy as jnp

    df = load_mpc_asteroids()
    eph = get_asteroid_ephemeris(df, 1)  # Ceres
    oe = jnp.array([eph["a"], eph["e"], eph["i"], eph["node"], eph["peri"], eph["M"]])
    state = asteroid_state_ecliptic(eph["epoch_jd"], oe, 2460000.5)
"""

from astrojax.datasets._mpc_download import download_mpc_file
from astrojax.datasets._mpc_parsers import (
    load_mpc_json_to_dataframe,
    packed_mpc_epoch_to_jd,
    unpack_mpc_epoch,
)
from astrojax.datasets._mpc_providers import (
    load_mpc_asteroids,
    load_mpc_from_file,
)
from astrojax.datasets._mpc_state import (
    asteroid_state_ecliptic,
    get_asteroid_ephemeris,
)

__all__ = [
    "asteroid_state_ecliptic",
    "download_mpc_file",
    "get_asteroid_ephemeris",
    "load_mpc_asteroids",
    "load_mpc_from_file",
    "load_mpc_json_to_dataframe",
    "packed_mpc_epoch_to_jd",
    "unpack_mpc_epoch",
]
