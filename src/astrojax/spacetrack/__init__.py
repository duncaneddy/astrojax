"""SpaceTrack module.

Provides a client for querying satellite catalog data from Space-Track.org.
"""

from astrojax.gp_record import GPRecord
from astrojax.spacetrack.client import SpaceTrackClient
from astrojax.spacetrack.operators import operators
from astrojax.spacetrack.query import SpaceTrackQuery
from astrojax.spacetrack.rate_limiter import RateLimitConfig
from astrojax.spacetrack.responses import SATCATRecord
from astrojax.spacetrack.types import (
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
