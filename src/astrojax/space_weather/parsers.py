"""Parsers for CSSI Space Weather data files.

Supports the CelesTrak CSSI format (``sw19571001.txt``), which provides
historical and predicted geomagnetic indices (Kp, Ap) and solar radio flux
(F10.7). Column ranges and encoding follow the Fortran specification and
match the Rust reference implementation in
``refs/brahe/src/space_weather/parser.rs``.
"""

from __future__ import annotations

import math

from astrojax.time import caldate_to_mjd


def _convert_kp_to_float(kp_int: int) -> float:
    """Convert Kp integer encoding (0-90) to float (0.0-9.0).

    Kp values are stored as integers where:
    - 0, 10, 20, ... 90 represent 0, 1, 2, ... 9
    - 3, 13, 23, ... represent 0+, 1+, 2+, ... (add 1/3)
    - 7, 17, 27, ... represent 1-, 2-, 3-, ... (add 2/3)

    Args:
        kp_int: Integer Kp value (0-90).

    Returns:
        Float Kp value (0.0-9.0).
    """
    base = kp_int // 10
    remainder = kp_int % 10

    if remainder == 0:
        fractional = 0.0
    elif remainder == 3:
        fractional = 1.0 / 3.0
    elif remainder == 7:
        fractional = 2.0 / 3.0
    else:
        fractional = remainder / 10.0

    return base + fractional


def _parse_field(line: str, start: int, end: int) -> str:
    """Extract and strip a fixed-width field from a line.

    Args:
        line: The data line.
        start: Start column (0-indexed).
        end: End column (exclusive).

    Returns:
        Stripped field string.
    """
    if end > len(line):
        return ""
    return line[start:end].strip()


def _parse_float(line: str, start: int, end: int) -> float:
    """Parse a float from a fixed-width field, returning NaN on failure.

    Args:
        line: The data line.
        start: Start column (0-indexed).
        end: End column (exclusive).

    Returns:
        Parsed float value, or NaN if the field is blank or invalid.
    """
    field = _parse_field(line, start, end)
    if not field:
        return math.nan
    try:
        return float(field)
    except ValueError:
        return math.nan


def _parse_int(line: str, start: int, end: int) -> int | None:
    """Parse an int from a fixed-width field, returning None on failure.

    Args:
        line: The data line.
        start: Start column (0-indexed).
        end: End column (exclusive).

    Returns:
        Parsed int value, or None if the field is blank or invalid.
    """
    field = _parse_field(line, start, end)
    if not field:
        return None
    try:
        return int(field)
    except ValueError:
        return None


def is_data_line(line: str) -> bool:
    """Check whether a line starts with a 4-digit year (i.e. is a data line).

    Args:
        line: A line from the CSSI file.

    Returns:
        True if the line is a data line.
    """
    if len(line) < 4:
        return False
    try:
        int(line[0:4].strip())
        return True
    except ValueError:
        return False


def parse_cssi_file(
    filepath: str,
) -> tuple[
    list[float],  # mjds
    list[list[float]],  # kp (each entry is 8 values)
    list[list[float]],  # ap (each entry is 8 values)
    list[float],  # ap_daily
    list[float],  # f107_obs
    list[float],  # f107_adj (f107_adj_ctr81 used as "adjusted")
    list[float],  # f107_obs_ctr81
    list[float],  # f107_obs_lst81
    list[float],  # f107_adj_ctr81
    list[float],  # f107_adj_lst81
]:
    """Parse an entire CSSI space weather file into parallel lists.

    Reads all three sections (OBSERVED, DAILY_PREDICTED, MONTHLY_PREDICTED)
    and returns parallel lists ready for conversion to JAX arrays.

    Args:
        filepath: Path to the CSSI space weather file.

    Returns:
        Tuple of 10 parallel lists: (mjds, kp, ap, ap_daily, f107_obs,
        f107_adj, f107_obs_ctr81, f107_obs_lst81, f107_adj_ctr81,
        f107_adj_lst81).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no valid data lines were parsed.
    """
    mjds: list[float] = []
    kps: list[list[float]] = []
    aps: list[list[float]] = []
    ap_dailys: list[float] = []
    f107_obss: list[float] = []
    f107_adjs: list[float] = []
    f107_obs_ctr81s: list[float] = []
    f107_obs_lst81s: list[float] = []
    f107_adj_ctr81s: list[float] = []
    f107_adj_lst81s: list[float] = []

    current_section: str | None = None

    with open(filepath) as f:
        for line in f:
            line = line.rstrip("\n")
            trimmed = line.strip()

            if not trimmed:
                continue

            # Detect section markers
            if trimmed.startswith("BEGIN OBSERVED"):
                current_section = "OBSERVED"
                continue
            elif trimmed.startswith("BEGIN DAILY_PREDICTED"):
                current_section = "DAILY_PREDICTED"
                continue
            elif trimmed.startswith("BEGIN MONTHLY_PREDICTED"):
                current_section = "MONTHLY_PREDICTED"
                continue
            elif trimmed.startswith("END "):
                continue

            # Skip non-data lines or lines before any section
            if current_section is None or not is_data_line(trimmed):
                continue

            is_monthly = current_section == "MONTHLY_PREDICTED"

            # Use the original line (not trimmed) for fixed-width parsing
            result = _parse_cssi_data_line(line, is_monthly)
            if result is None:
                continue

            (
                mjd,
                kp_row,
                ap_row,
                ap_daily,
                f107_obs,
                f107_adj,
                f107_obs_ctr81,
                f107_obs_lst81,
                f107_adj_ctr81,
                f107_adj_lst81,
            ) = result

            mjds.append(mjd)
            kps.append(kp_row)
            aps.append(ap_row)
            ap_dailys.append(ap_daily)
            f107_obss.append(f107_obs)
            f107_adjs.append(f107_adj)
            f107_obs_ctr81s.append(f107_obs_ctr81)
            f107_obs_lst81s.append(f107_obs_lst81)
            f107_adj_ctr81s.append(f107_adj_ctr81)
            f107_adj_lst81s.append(f107_adj_lst81)

    if not mjds:
        raise ValueError(f"No valid space weather data found in {filepath}")

    return (
        mjds,
        kps,
        aps,
        ap_dailys,
        f107_obss,
        f107_adjs,
        f107_obs_ctr81s,
        f107_obs_lst81s,
        f107_adj_ctr81s,
        f107_adj_lst81s,
    )


def _parse_cssi_data_line(
    line: str,
    is_monthly: bool,
) -> (
    tuple[
        float,
        list[float],
        list[float],
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    | None
):
    """Parse a single CSSI data line.

    Args:
        line: A data line from the CSSI file.
        is_monthly: If True, Kp/Ap fields are blank (MONTHLY_PREDICTED).

    Returns:
        Tuple of parsed values, or None if the line cannot be parsed.
    """
    min_len = 124 if is_monthly else 130

    if len(line) < min_len:
        return None

    # Parse date fields
    year = _parse_int(line, 0, 4)
    month = _parse_int(line, 4, 7)
    day = _parse_int(line, 7, 10)

    if year is None or month is None or day is None:
        return None

    # Calculate MJD
    mjd = float(caldate_to_mjd(year, month, day))

    if is_monthly:
        # Monthly predicted: Kp/Ap are blank
        kp_row = [math.nan] * 8
        ap_row = [math.nan] * 8
        ap_daily = math.nan
    else:
        # Parse 8 Kp indices (position 18, each 3 chars wide)
        kp_row = []
        kp_start = 18
        for i in range(8):
            kp_int = _parse_int(line, kp_start + i * 3, kp_start + (i + 1) * 3)
            if kp_int is None:
                kp_row.append(math.nan)
            else:
                kp_row.append(_convert_kp_to_float(kp_int))

        # Parse 8 Ap indices (position 46, each 4 chars wide)
        ap_row = []
        ap_start = 46
        for i in range(8):
            ap_val = _parse_float(line, ap_start + i * 4, ap_start + (i + 1) * 4)
            ap_row.append(ap_val)

        # Parse Ap daily average (position 78, width 4)
        ap_daily = _parse_float(line, 78, 82)

    # Parse F10.7 observed (position 92, width 6)
    f107_obs = _parse_float(line, 92, 98)

    # Parse F10.7 averages (each width 6)
    f107_adj_ctr81 = _parse_float(line, 100, 106)
    f107_adj_lst81 = _parse_float(line, 106, 112)
    f107_obs_ctr81 = _parse_float(line, 112, 118)
    f107_obs_lst81 = _parse_float(line, 118, 124)

    # Use f107_adj_ctr81 as the "adjusted" F10.7
    f107_adj = f107_adj_ctr81

    return (
        mjd,
        kp_row,
        ap_row,
        ap_daily,
        f107_obs,
        f107_adj,
        f107_obs_ctr81,
        f107_obs_lst81,
        f107_adj_ctr81,
        f107_adj_lst81,
    )
