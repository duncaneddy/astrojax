"""Celestrak module.

Provides a client for querying satellite catalog data from Celestrak.
"""

from astrojax.celestrak.client import CelestrakClient
from astrojax.celestrak.query import CelestrakQuery
from astrojax.celestrak.responses import CelestrakSATCATRecord
from astrojax.celestrak.types import (
    CelestrakOutputFormat,
    CelestrakQueryType,
    SupGPSource,
)
from astrojax.gp_record import GPRecord

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
