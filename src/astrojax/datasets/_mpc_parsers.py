"""Parsing utilities for MPC asteroid orbit data.

Provides helpers to decode MPC packed epoch dates and convert the
``mpcorb_extended.json.gz`` file into a Polars DataFrame with computed
Julian Date epochs.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import polars as pl

from astrojax.time import caldate_to_jd

logger = logging.getLogger(__name__)

# MPC packed-date character maps
_CENTURY_MAP: dict[str, int] = {"I": 18, "J": 19, "K": 20}
_CHAR_TO_INT: dict[str, int] = {chr(c): c - ord("A") + 10 for c in range(ord("A"), ord("W"))}
"""Maps A=10, B=11, ... V=31 for month and day packed encoding."""


def unpack_mpc_epoch(packed: str) -> tuple[int, int, int]:
    """Decode an MPC 5-character packed date to ``(year, month, day)``.

    The encoding uses five characters:

    - Position 0: century (``I`` = 18, ``J`` = 19, ``K`` = 20)
    - Positions 1–2: two-digit year within the century
    - Position 3: month (``1``–``9`` for Jan–Sep, ``A`` = Oct,
      ``B`` = Nov, ``C`` = Dec)
    - Position 4: day (``1``–``9``, ``A`` = 10, … ``V`` = 31)

    Args:
        packed: A 5-character MPC packed epoch string.

    Returns:
        Tuple of ``(year, month, day)`` as integers.

    Raises:
        ValueError: If the packed string is malformed.

    Examples:
        >>> unpack_mpc_epoch("K24BN")
        (2024, 11, 23)
        >>> unpack_mpc_epoch("J9611")
        (1996, 1, 1)
    """
    if len(packed) != 5:
        raise ValueError(f"Packed epoch must be 5 characters, got {len(packed)!r}: {packed!r}")

    century_char = packed[0]
    if century_char not in _CENTURY_MAP:
        raise ValueError(
            f"Unknown century character {century_char!r} in packed epoch {packed!r}. "
            f"Expected one of {list(_CENTURY_MAP.keys())}."
        )
    century = _CENTURY_MAP[century_char]
    year = century * 100 + int(packed[1:3])

    month_char = packed[3]
    if month_char.isdigit():
        month = int(month_char)
    elif month_char in _CHAR_TO_INT:
        month = _CHAR_TO_INT[month_char]
    else:
        raise ValueError(f"Unknown month character {month_char!r} in packed epoch {packed!r}")

    day_char = packed[4]
    if day_char.isdigit():
        day = int(day_char)
    elif day_char in _CHAR_TO_INT:
        day = _CHAR_TO_INT[day_char]
    else:
        raise ValueError(f"Unknown day character {day_char!r} in packed epoch {packed!r}")

    return year, month, day


def packed_mpc_epoch_to_jd(packed: str) -> float:
    """Convert an MPC packed epoch to Julian Date (TT).

    Args:
        packed: A 5-character MPC packed epoch string.

    Returns:
        Julian Date (TT) as a float.

    Examples:
        >>> packed_mpc_epoch_to_jd("J9611")  # 1996-01-01
        2450083.5
    """
    year, month, day = unpack_mpc_epoch(packed)
    return float(caldate_to_jd(year, month, day))


def load_mpc_json_to_dataframe(filepath: str | Path) -> pl.DataFrame:
    """Load an MPC ``mpcorb_extended.json.gz`` file into a Polars DataFrame.

    Decompresses the gzipped JSON, extracts the relevant orbital element
    columns, and computes an ``epoch_jd`` column from the packed epoch.

    Args:
        filepath: Path to the ``.json.gz`` file.

    Returns:
        Polars DataFrame with columns: ``number``, ``name``,
        ``principal_desig``, ``epoch_packed``, ``epoch_jd``, ``a``,
        ``e``, ``i``, ``node``, ``peri``, ``M``, ``n``, ``H``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MPC file not found: {filepath}")

    logger.info("Loading MPC data from %s", filepath)
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        raw = json.load(f)

    # Column mapping: (output_name, json_key)
    column_map = [
        ("number", "Number"),
        ("name", "Name"),
        ("principal_desig", "Principal_desig"),
        ("epoch_packed", "Epoch"),
        ("a", "a"),
        ("e", "e"),
        ("i", "i"),
        ("node", "Node"),
        ("peri", "Peri"),
        ("M", "M"),
        ("n", "n"),
        ("H", "H"),
    ]

    rows: dict[str, list] = {col: [] for col, _ in column_map}
    rows["epoch_jd"] = []

    for record in raw:
        for col, key in column_map:
            rows[col].append(record.get(key))
        epoch_str = record.get("Epoch", "")
        if epoch_str and len(epoch_str) == 5:
            try:
                rows["epoch_jd"].append(packed_mpc_epoch_to_jd(epoch_str))
            except (ValueError, KeyError):
                rows["epoch_jd"].append(None)
        else:
            rows["epoch_jd"].append(None)

    df = pl.DataFrame(
        {
            "number": pl.Series(rows["number"], dtype=pl.Utf8),
            "name": pl.Series(rows["name"], dtype=pl.Utf8),
            "principal_desig": pl.Series(rows["principal_desig"], dtype=pl.Utf8),
            "epoch_packed": pl.Series(rows["epoch_packed"], dtype=pl.Utf8),
            "epoch_jd": pl.Series(rows["epoch_jd"], dtype=pl.Float64),
            "a": pl.Series(rows["a"], dtype=pl.Float64),
            "e": pl.Series(rows["e"], dtype=pl.Float64),
            "i": pl.Series(rows["i"], dtype=pl.Float64),
            "node": pl.Series(rows["node"], dtype=pl.Float64),
            "peri": pl.Series(rows["peri"], dtype=pl.Float64),
            "M": pl.Series(rows["M"], dtype=pl.Float64),
            "n": pl.Series(rows["n"], dtype=pl.Float64),
            "H": pl.Series(rows["H"], dtype=pl.Float64),
        }
    )

    logger.info("Loaded %d asteroid records", len(df))
    return df
