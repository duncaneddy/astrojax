"""Celestrak module.

Provides a client for querying satellite catalog data from Celestrak.
"""

from astrojax._gp_record import GPRecord
from astrojax.celestrak._client import CelestrakClient
from astrojax.celestrak._query import CelestrakQuery
from astrojax.celestrak._responses import CelestrakSATCATRecord
from astrojax.celestrak._types import (
    CelestrakOutputFormat,
    CelestrakQueryType,
    SupGPSource,
)

__all__ = [
    # Enums
    "CelestrakQueryType",
    "CelestrakOutputFormat",
    "SupGPSource",
    # Query builder
    "CelestrakQuery",
    # Client
    "CelestrakClient",
    # Response types
    "GPRecord",
    "CelestrakSATCATRecord",
]
