"""
TLE and OMM parsing for the SGP4/SDP4 propagator.

Provides pure-Python functions to parse Two-Line Element (TLE) sets and
Orbit Mean-Elements Message (OMM) fields into ``SGP4Elements`` suitable
for SGP4 initialization.
"""

from datetime import datetime
from math import pi

from astrojax.sgp4._types import SGP4Elements

# Conversion constants (matching reference sgp4 library exactly)
_DEG2RAD = pi / 180.0
_XPDOTP = 1440.0 / (2.0 * pi)  # 229.1831180523293

# OMM unit conversion constants (from sgp4/omm.py)
_OMM_EPOCH0 = datetime(1949, 12, 31)
_OMM_NDOT_UNITS = 1036800.0 / pi
_OMM_NDDOT_UNITS = 2985984000.0 / 2.0 / pi


def compute_checksum(line: str) -> int:
    """Compute the TLE checksum for a line.

    The checksum is the sum of all digit characters plus 1 for each
    minus sign, modulo 10, computed over the first 68 characters.

    Args:
        line: A TLE line string (at least 68 characters).

    Returns:
        The checksum digit (0-9).
    """
    return sum((int(c) if c.isdigit() else c == "-") for c in line[:68]) % 10


def validate_tle_line(line: str, line_number: int) -> None:
    """Validate a TLE line's format and checksum.

    Args:
        line: A TLE line string.
        line_number: Expected line number (1 or 2).

    Raises:
        ValueError: If the line fails format or checksum validation.
    """
    line = line.rstrip()

    if len(line) < 69:
        raise ValueError(
            f"TLE line {line_number} is too short ({len(line)} chars, expected 69): {line}"
        )

    if line[0] != str(line_number):
        raise ValueError(f"TLE line {line_number} does not start with '{line_number}': {line}")

    checksum_char = line[68]
    if not checksum_char.isdigit():
        raise ValueError(f"TLE line {line_number} has non-digit checksum: {line}")

    expected = compute_checksum(line)
    actual = int(checksum_char)
    if expected != actual:
        raise ValueError(
            f"TLE line {line_number} checksum mismatch: computed {expected}, found {actual}: {line}"
        )


def parse_tle(line1: str, line2: str) -> SGP4Elements:
    """Parse a Two-Line Element set into SGP4 orbital elements.

    Follows the fixed-column TLE format specification. Angular values are
    converted to radians and mean motion to rad/min for SGP4 internal use.

    Args:
        line1: First TLE line (69 characters including checksum).
        line2: Second TLE line (69 characters including checksum).

    Returns:
        Parsed orbital elements ready for SGP4 initialization.

    Raises:
        ValueError: If lines fail format validation, checksum check,
            or satellite numbers don't match.
    """
    # Validate both lines
    validate_tle_line(line1, 1)
    validate_tle_line(line2, 2)

    # Strip trailing whitespace for parsing
    l1 = line1.rstrip()
    l2 = line2.rstrip()

    # Parse line 1
    satnum_str = l1[2:7]
    classification = l1[7] or "U"
    intldesg = l1[9:17].rstrip()
    two_digit_year = int(l1[18:20])
    epochdays = float(l1[20:32])
    ndot = float(l1[33:43])
    nddot = float(l1[44] + "." + l1[45:50])
    nexp = int(l1[50:52])
    bstar = float(l1[53] + "." + l1[54:59])
    ibexp = int(l1[59:61])
    ephtype_str = l1[62]
    elnum = int(l1[64:68])

    # Parse line 2
    if satnum_str != l2[2:7]:
        raise ValueError("Object numbers in lines 1 and 2 do not match")

    inclo = float(l2[8:16])
    nodeo = float(l2[17:25])
    ecco = float("0." + l2[26:33].replace(" ", "0"))
    argpo = float(l2[34:42])
    mo = float(l2[43:51])
    no_kozai = float(l2[52:63])
    revnum = int(l2[63:68])

    # Apply exponents
    nddot = nddot * 10.0**nexp
    bstar = bstar * 10.0**ibexp

    # Convert to SGP4 internal units
    no_kozai = no_kozai / _XPDOTP  # rad/min
    ndot = ndot / (_XPDOTP * 1440.0)  # rad/min^2
    nddot = nddot / (_XPDOTP * 1440.0 * 1440.0)  # rad/min^3

    # Convert angles to radians
    inclo = inclo * _DEG2RAD
    nodeo = nodeo * _DEG2RAD
    argpo = argpo * _DEG2RAD
    mo = mo * _DEG2RAD

    # Compute epoch year (4-digit)
    if two_digit_year < 57:
        year = two_digit_year + 2000
    else:
        year = two_digit_year + 1900

    # Use the same split-JD approach as python-sgp4 Satrec.twoline2rv()
    days_int, fraction = divmod(epochdays, 1.0)
    jdsatepoch = year * 365 + (year - 1) // 4 + int(days_int) + 1721044.5
    jdsatepochF = round(fraction, 8)

    return SGP4Elements(
        satnum_str=satnum_str,
        classification=classification,
        intldesg=intldesg,
        epochyr=year % 100,
        epochdays=epochdays,
        ndot=ndot,
        nddot=nddot,
        bstar=bstar,
        ephtype=int(ephtype_str.strip() or "0"),
        elnum=elnum,
        revnum=revnum,
        inclo=inclo,
        nodeo=nodeo,
        ecco=ecco,
        argpo=argpo,
        mo=mo,
        no_kozai=no_kozai,
        jdsatepoch=jdsatepoch,
        jdsatepochF=jdsatepochF,
    )


def parse_omm(fields: dict[str, str]) -> SGP4Elements:
    """Parse OMM (Orbit Mean-Elements Message) fields into SGP4 orbital elements.

    Accepts a dictionary of OMM field names and string values, as would be
    parsed from a CSV or XML OMM record.

    Args:
        fields: Dictionary mapping OMM field names to string values. Required
            keys: ``EPOCH``, ``MEAN_MOTION``, ``ECCENTRICITY``, ``INCLINATION``,
            ``RA_OF_ASC_NODE``, ``ARG_OF_PERICENTER``, ``MEAN_ANOMALY``,
            ``NORAD_CAT_ID``, ``BSTAR``. Optional keys: ``CLASSIFICATION_TYPE``,
            ``OBJECT_ID``, ``EPHEMERIS_TYPE``, ``ELEMENT_SET_NO``,
            ``REV_AT_EPOCH``, ``MEAN_MOTION_DOT``, ``MEAN_MOTION_DDOT``.

    Returns:
        Parsed orbital elements ready for SGP4 initialization.

    Raises:
        KeyError: If a required field is missing.
    """
    # Parse epoch
    epoch_str = fields["EPOCH"]
    # Handle both with and without microseconds
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            epoch_datetime = datetime.strptime(epoch_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Cannot parse OMM EPOCH: {epoch_str}")

    # Epoch as days since 1949-12-31 (SGP4 convention)
    epoch = (epoch_datetime - _OMM_EPOCH0).total_seconds() / 86400.0

    # Orbital elements
    argpo = float(fields["ARG_OF_PERICENTER"]) * _DEG2RAD
    bstar = float(fields["BSTAR"])
    ecco = float(fields["ECCENTRICITY"])
    inclo = float(fields["INCLINATION"]) * _DEG2RAD
    mo = float(fields["MEAN_ANOMALY"]) * _DEG2RAD
    no_kozai = float(fields["MEAN_MOTION"]) / 720.0 * pi  # rev/day -> rad/min
    nodeo = float(fields["RA_OF_ASC_NODE"]) * _DEG2RAD

    # Optional drag terms
    ndot = float(fields.get("MEAN_MOTION_DOT", "0")) / _OMM_NDOT_UNITS
    nddot = float(fields.get("MEAN_MOTION_DDOT", "0")) / _OMM_NDDOT_UNITS

    # Metadata
    satnum = int(fields["NORAD_CAT_ID"])
    satnum_str = str(satnum).rjust(5)
    classification = fields.get("CLASSIFICATION_TYPE", "U")
    intldesg = fields.get("OBJECT_ID", "")[2:].replace("-", "") if "OBJECT_ID" in fields else ""
    ephtype = int(fields.get("EPHEMERIS_TYPE", "0"))
    elnum = int(fields.get("ELEMENT_SET_NO", "999"))
    revnum = int(fields.get("REV_AT_EPOCH", "0"))

    # Compute epoch year and day
    epoch_year = epoch_datetime.year
    jan1 = datetime(epoch_year, 1, 1)
    epochdays = (epoch_datetime - jan1).total_seconds() / 86400.0 + 1.0

    # Compute split Julian date (same as sgp4init expects)
    whole, fraction = divmod(epoch, 1.0)
    jdsatepoch = whole + 2433281.5
    if round(epoch, 8) == epoch:
        fraction = round(fraction, 8)
    jdsatepochF = fraction

    return SGP4Elements(
        satnum_str=satnum_str,
        classification=classification,
        intldesg=intldesg,
        epochyr=epoch_year % 100,
        epochdays=epochdays,
        ndot=ndot,
        nddot=nddot,
        bstar=bstar,
        ephtype=ephtype,
        elnum=elnum,
        revnum=revnum,
        inclo=inclo,
        nodeo=nodeo,
        ecco=ecco,
        argpo=argpo,
        mo=mo,
        no_kozai=no_kozai,
        jdsatepoch=jdsatepoch,
        jdsatepochF=jdsatepochF,
    )
