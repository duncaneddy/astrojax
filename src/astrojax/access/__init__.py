"""Satellite access (visibility) prediction and constraint functions.

Provides tools for computing when a satellite is visible from a ground
station, including JIT-compilable and hybrid (JIT + Python) approaches.

Available components:

- :class:`GroundLocation` -- Ground station with pre-computed ECEF/ENZ
- :class:`AccessWindow` -- Single visibility window result
- :class:`AccessResult` -- Fixed-shape JIT-compatible result
- :func:`ground_location` -- Construct a GroundLocation
- :func:`compute_elevation` -- Elevation angle computation
- :func:`compute_azel` -- Azimuth/elevation/range computation
- :func:`find_access_windows` -- Hybrid access window finder
- :func:`find_access_windows_jit` -- Fully JIT-compilable finder
- :func:`find_all_access_windows` -- Paging wrapper over JIT finder
- :func:`find_access_windows_from_ephemeris` -- From GCRF ephemeris

Constraint functions:

- :func:`elevation_constraint` -- Min/max elevation band
- :func:`elevation_mask_constraint` -- Azimuth-dependent elevation mask
- :func:`off_nadir_constraint` -- Off-nadir angle bounds
- :func:`constraint_all` -- AND composition
- :func:`constraint_any` -- OR composition
- :func:`constraint_not` -- Negation
"""

from astrojax.access.constraints import (
    compute_off_nadir,
    constraint_all,
    constraint_any,
    constraint_not,
    elevation_constraint,
    elevation_mask_constraint,
    off_nadir_constraint,
)
from astrojax.access.windows import (
    AccessResult,
    AccessWindow,
    GroundLocation,
    _detect_crossings_jit,  # noqa: F401
    _detect_windows,  # noqa: F401
    _find_max_elevation,  # noqa: F401
    _refine_boundary,  # noqa: F401
    compute_azel,
    compute_elevation,
    find_access_windows,
    find_access_windows_from_ephemeris,
    find_access_windows_jit,
    find_all_access_windows,
    ground_location,
)

__all__ = [
    # Window computation
    "AccessResult",
    "AccessWindow",
    "GroundLocation",
    "compute_azel",
    "compute_elevation",
    "find_access_windows",
    "find_access_windows_from_ephemeris",
    "find_access_windows_jit",
    "find_all_access_windows",
    "ground_location",
    # Constraints
    "compute_off_nadir",
    "constraint_all",
    "constraint_any",
    "constraint_not",
    "elevation_constraint",
    "elevation_mask_constraint",
    "off_nadir_constraint",
]
