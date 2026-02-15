"""Parsing utilities for the SBN Archive asteroid masses dataset.

Extracts the fixed-width ``astmass12.tab`` table from the downloaded ZIP
archive and converts it into a Polars DataFrame with typed columns for
mass, density, shape, and reference information.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

_TAB_FILENAME: str = "astmass12.tab"
"""Name of the fixed-width data file inside the ZIP archive."""

# Each entry: (column_name, start_0idx, length, polars_dtype)
_FIELD_DEFS: list[tuple[str, int, int, pl.DataType]] = [
    ("ast_number", 0, 7, pl.Int64),
    ("ast_name", 8, 17, pl.Utf8),
    ("prov_desig", 26, 10, pl.Utf8),
    ("satellite_name", 37, 11, pl.Utf8),
    ("mass_sol", 49, 9, pl.Float64),
    ("mass_sol_unc", 59, 8, pl.Float64),
    ("mass_kg", 68, 8, pl.Float64),
    ("mass_kg_unc", 77, 8, pl.Float64),
    ("bulk_density", 86, 4, pl.Float64),
    ("bulk_density_unc", 91, 4, pl.Float64),
    ("axis_a", 96, 7, pl.Float64),
    ("axis_b", 104, 7, pl.Float64),
    ("axis_c", 112, 7, pl.Float64),
    ("equiv_radius_unc", 120, 6, pl.Float64),
    ("mass_ref", 182, 25, pl.Utf8),
]
"""Field definitions for the fixed-width ``astmass12.tab`` format."""


def _parse_field(raw: str, dtype: pl.DataType) -> object:
    """Parse a single stripped field value, returning ``None`` for blanks.

    Args:
        raw: The stripped string value from the fixed-width field.
        dtype: The target Polars data type.

    Returns:
        Parsed value or ``None`` if the field is blank.
    """
    if not raw:
        return None
    if dtype == pl.Int64:
        return int(raw)
    if dtype == pl.Float64:
        return float(raw)
    return raw


def load_astmass_tab_to_dataframe(filepath: str | Path) -> pl.DataFrame:
    """Load the asteroid masses ``.tab`` file from a ZIP into a Polars DataFrame.

    Opens the ZIP at *filepath*, searches for an entry whose name ends with
    ``astmass12.tab`` (to handle subdirectory paths inside the archive),
    and parses each line of the fixed-width table by byte-offset slicing.

    Args:
        filepath: Path to the ``compil.ast.masses.zip`` file.

    Returns:
        Polars DataFrame with 15 typed columns.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If the ZIP does not contain a ``astmass12.tab`` entry.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Asteroid masses file not found: {filepath}")

    with zipfile.ZipFile(filepath, "r") as zf:
        # Find the .tab entry (may be nested in a subdirectory)
        tab_entry: str | None = None
        for name in zf.namelist():
            if name.endswith(_TAB_FILENAME):
                tab_entry = name
                break

        if tab_entry is None:
            raise ValueError(f"ZIP archive does not contain a '{_TAB_FILENAME}' file: {filepath}")

        raw_bytes = zf.read(tab_entry)

    text = raw_bytes.decode("ascii")
    lines = text.splitlines()

    # Build column data
    columns: dict[str, list[object]] = {name: [] for name, _, _, _ in _FIELD_DEFS}

    for line in lines:
        if not line.strip():
            continue
        for name, start, length, dtype in _FIELD_DEFS:
            raw = line[start : start + length].strip()
            columns[name].append(_parse_field(raw, dtype))

    df = pl.DataFrame(
        {name: pl.Series(columns[name], dtype=dtype) for name, _, _, dtype in _FIELD_DEFS}
    )

    logger.info("Loaded %d asteroid mass records from %s", len(df), filepath)
    return df
