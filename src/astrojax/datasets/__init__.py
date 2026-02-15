"""Asteroid and small-body datasets for astrojax.

Provides access to the Minor Planet Center (MPC) asteroid orbit catalog,
the SBN Archive asteroid masses compilation, and the DAMIT asteroid
shape model database, including download/caching, Polars DataFrame
loading, per-asteroid lookups, and heliocentric ecliptic state vector
computation.

Typical usage::

    from astrojax.datasets import load_mpc_asteroids, get_asteroid_ephemeris, asteroid_state_ecliptic
    import jax.numpy as jnp

    df = load_mpc_asteroids()
    eph = get_asteroid_ephemeris(df, 1)  # Ceres
    oe = jnp.array([eph["a"], eph["e"], eph["i"], eph["node"], eph["peri"], eph["M"]])
    state = asteroid_state_ecliptic(eph["epoch_jd"], oe, 2460000.5)

    from astrojax.datasets import load_asteroid_masses
    masses_df = load_asteroid_masses()

    from astrojax.datasets import load_damit_models, damit_spin_to_rotation
    models = load_damit_models()
"""

from astrojax.datasets._astmass_download import download_astmass_file
from astrojax.datasets._astmass_parsers import load_astmass_tab_to_dataframe
from astrojax.datasets._astmass_providers import (
    load_asteroid_masses,
    load_astmass_from_file,
)
from astrojax.datasets._damit_download import download_damit_file, extract_damit_archive
from astrojax.datasets._damit_parsers import (
    load_shape_for_model,
    parse_damit_asteroids_table,
    parse_damit_models_table,
    parse_shape_file,
)
from astrojax.datasets._damit_providers import (
    get_damit_shape,
    get_damit_spin,
    load_damit_asteroids,
    load_damit_models,
)
from astrojax.datasets._damit_shapes import (
    export_shape_glb,
    export_shape_stl,
    shape_to_trimesh,
)
from astrojax.datasets._damit_spin import (
    damit_spin_to_rotation,
    rotate_shape_points,
    scale_shape_vertices,
)
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
    "damit_spin_to_rotation",
    "download_astmass_file",
    "download_damit_file",
    "extract_damit_archive",
    "download_mpc_file",
    "export_shape_glb",
    "export_shape_stl",
    "get_asteroid_ephemeris",
    "get_damit_shape",
    "get_damit_spin",
    "load_asteroid_masses",
    "load_astmass_from_file",
    "load_astmass_tab_to_dataframe",
    "load_damit_asteroids",
    "load_damit_models",
    "load_mpc_asteroids",
    "load_mpc_from_file",
    "load_mpc_json_to_dataframe",
    "load_shape_for_model",
    "packed_mpc_epoch_to_jd",
    "parse_damit_asteroids_table",
    "parse_damit_models_table",
    "parse_shape_file",
    "rotate_shape_points",
    "scale_shape_vertices",
    "shape_to_trimesh",
    "unpack_mpc_epoch",
]
