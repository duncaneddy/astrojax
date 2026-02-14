"""Earth Orientation Parameters (EOP) for JAX-compatible lookups.

Provides JIT-compatible EOP data storage and interpolation using sorted
JAX arrays and ``jnp.searchsorted``. All query functions work inside
``jax.jit``, ``jax.vmap``, and ``jax.grad``.

Typical usage::

    from astrojax.eop import load_default_eop, get_ut1_utc
    eop = load_default_eop()
    ut1_utc = get_ut1_utc(eop, 59569.5)
"""

from astrojax.eop._download import download_standard_eop_file
from astrojax.eop._lookup import get_dxdy, get_eop, get_lod, get_pm, get_ut1_utc
from astrojax.eop._providers import (
    load_cached_eop,
    load_default_eop,
    load_eop_from_file,
    load_eop_from_standard_file,
    static_eop,
    zero_eop,
)
from astrojax.eop._types import EOPData, EOPExtrapolation

__all__ = [
    "EOPData",
    "EOPExtrapolation",
    "download_standard_eop_file",
    "get_dxdy",
    "get_eop",
    "get_lod",
    "get_pm",
    "get_ut1_utc",
    "load_cached_eop",
    "load_default_eop",
    "load_eop_from_file",
    "load_eop_from_standard_file",
    "static_eop",
    "zero_eop",
]
