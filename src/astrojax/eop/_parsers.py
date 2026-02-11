"""Parsers for IERS Earth Orientation Parameter data files.

Supports the IERS standard format (``finals.all.iau2000.txt``), also known
as Bulletin A/B format. Column ranges and unit conversions follow the IERS
specification and match the Rust reference implementation in
``refs/brahe_rust/src/eop/standard_parser.rs``.
"""

from __future__ import annotations

import math

from astrojax.constants import AS2RAD

# Column ranges for IERS standard format (0-indexed Python slices)
_MJD_RANGE = slice(6, 15)
_PM_X_RANGE = slice(17, 27)
_PM_Y_RANGE = slice(36, 46)
_UT1_UTC_RANGE = slice(58, 68)
_LOD_RANGE = slice(78, 86)
_DX_RANGE = slice(96, 106)
_DY_RANGE = slice(115, 125)
_STANDARD_LINE_LENGTH = 187


def parse_standard_line(
    line: str,
) -> tuple[float, float, float, float, float, float, float] | None:
    """Parse a single line from an IERS standard format EOP file.

    Lines shorter than 187 characters are padded with spaces (prediction
    lines may have trailing whitespace trimmed). Lines longer than 187
    characters or lines where required fields (MJD, PM_X, PM_Y, UT1-UTC)
    cannot be parsed are skipped (returns None).

    Args:
        line: A single line from the IERS standard format file.

    Returns:
        Tuple of (mjd, pm_x [rad], pm_y [rad], ut1_utc [s],
        lod [s] or NaN, dX [rad] or NaN, dY [rad] or NaN),
        or None if the line cannot be parsed.
    """
    if len(line) > _STANDARD_LINE_LENGTH:
        return None

    # Pad shorter lines to 187 chars
    line = line.ljust(_STANDARD_LINE_LENGTH)

    # Parse required fields
    try:
        mjd = float(line[_MJD_RANGE].strip())
    except ValueError:
        return None

    try:
        pm_x = float(line[_PM_X_RANGE].strip()) * AS2RAD
    except ValueError:
        return None

    try:
        pm_y = float(line[_PM_Y_RANGE].strip()) * AS2RAD
    except ValueError:
        return None

    try:
        ut1_utc = float(line[_UT1_UTC_RANGE].strip())
    except ValueError:
        return None

    # Parse optional fields (NaN if missing)
    try:
        lod = float(line[_LOD_RANGE].strip()) * 1.0e-3  # ms -> s
    except ValueError:
        lod = math.nan

    try:
        dX = float(line[_DX_RANGE].strip()) * 1.0e-3 * AS2RAD  # mas -> rad
    except ValueError:
        dX = math.nan

    try:
        dY = float(line[_DY_RANGE].strip()) * 1.0e-3 * AS2RAD  # mas -> rad
    except ValueError:
        dY = math.nan

    return mjd, pm_x, pm_y, ut1_utc, lod, dX, dY


def parse_standard_file(
    filepath: str,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Parse an entire IERS standard format EOP file.

    Reads all valid lines and returns parallel lists of EOP values.
    Lines that cannot be parsed (e.g. empty prediction lines at the
    end of the file) are silently skipped.

    Args:
        filepath: Path to the IERS standard format file.

    Returns:
        Tuple of 7 lists: (mjd, pm_x, pm_y, ut1_utc, lod, dX, dY).
        Units match :func:`parse_standard_line`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no valid lines were parsed.
    """
    mjds: list[float] = []
    pm_xs: list[float] = []
    pm_ys: list[float] = []
    ut1_utcs: list[float] = []
    lods: list[float] = []
    dXs: list[float] = []
    dYs: list[float] = []

    with open(filepath) as f:
        for line in f:
            result = parse_standard_line(line.rstrip("\n"))
            if result is not None:
                mjd, pm_x, pm_y, ut1_utc, lod, dX, dY = result
                mjds.append(mjd)
                pm_xs.append(pm_x)
                pm_ys.append(pm_y)
                ut1_utcs.append(ut1_utc)
                lods.append(lod)
                dXs.append(dX)
                dYs.append(dY)

    if not mjds:
        raise ValueError(f"No valid EOP data found in {filepath}")

    return mjds, pm_xs, pm_ys, ut1_utcs, lods, dXs, dYs
