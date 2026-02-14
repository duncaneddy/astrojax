"""SpaceTrack module.

Provides a client for querying satellite catalog data from Space-Track.org.
"""

from astrojax._gp_record import GPRecord
from astrojax.spacetrack._client import SpaceTrackClient
from astrojax.spacetrack._operators import operators
from astrojax.spacetrack._query import SpaceTrackQuery
from astrojax.spacetrack._rate_limiter import RateLimitConfig
from astrojax.spacetrack._responses import SATCATRecord
from astrojax.spacetrack._types import (
    OutputFormat,
    RequestClass,
    RequestController,
    SortOrder,
)

__all__ = [
    # Enums
    "RequestController",
    "RequestClass",
    "SortOrder",
    "OutputFormat",
    # Rate limiting
    "RateLimitConfig",
    # Query builder
    "SpaceTrackQuery",
    # Client
    "SpaceTrackClient",
    # Response types
    "GPRecord",
    "SATCATRecord",
    # Operators namespace
    "operators",
]
