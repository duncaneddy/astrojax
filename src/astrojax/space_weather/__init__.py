"""Space Weather data for JAX-compatible lookups.

Provides JIT-compatible space weather data storage and lookup using sorted
JAX arrays and ``jnp.searchsorted``. All query functions work inside
``jax.jit``, ``jax.vmap``, and ``jax.grad``.

Typical usage::

    from astrojax.space_weather import load_default_sw, get_sw_f107_obs
    sw = load_default_sw()
    f107 = get_sw_f107_obs(sw, 59569.5)
"""

from astrojax.space_weather._download import download_sw_file
from astrojax.space_weather._lookup import (
    get_sw_ap,
    get_sw_ap_array,
    get_sw_ap_daily,
    get_sw_f107_adj,
    get_sw_f107_obs,
    get_sw_f107_obs_ctr81,
    get_sw_f107_obs_lst81,
    get_sw_kp,
)
from astrojax.space_weather._providers import (
    load_cached_sw,
    load_default_sw,
    load_sw_from_file,
    static_space_weather,
    zero_space_weather,
)
from astrojax.space_weather._types import SpaceWeatherData

__all__ = [
    "SpaceWeatherData",
    "download_sw_file",
    "get_sw_ap",
    "get_sw_ap_array",
    "get_sw_ap_daily",
    "get_sw_f107_adj",
    "get_sw_f107_obs",
    "get_sw_f107_obs_ctr81",
    "get_sw_f107_obs_lst81",
    "get_sw_kp",
    "load_cached_sw",
    "load_default_sw",
    "load_sw_from_file",
    "static_space_weather",
    "zero_space_weather",
]
