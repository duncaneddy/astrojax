"""Parsing utilities for the JPL Small Body Database (SBDB) asteroid dataset.

Reads the SBDB CSV file and converts it into a Polars DataFrame with
column names matching the MPC convention for downstream compatibility
with :func:`~astrojax.datasets.asteroid_state_ecliptic`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

_COLUMN_RENAMES: dict[str, str] = {
    "pdes": "number",
    "epoch": "epoch_jd",
    "om": "node",
    "w": "peri",
    "ma": "M",
}
"""Mapping from SBDB CSV column names to MPC-compatible names."""


def load_sbdb_csv_to_dataframe(filepath: str | Path) -> pl.DataFrame:
    """Load the SBDB CSV file into a Polars DataFrame.

    Reads the CSV at *filepath*, renames orbital-element columns to
    match the MPC naming convention (``om`` -> ``node``, ``w`` -> ``peri``,
    ``ma`` -> ``M``, ``epoch`` -> ``epoch_jd``, ``pdes`` -> ``number``),
    and casts the ``number`` column to ``Utf8`` for parity with the MPC
    DataFrame.

    Args:
        filepath: Path to the ``sbdb_asteroid_data.csv`` file.

    Returns:
        Polars DataFrame with 21 columns including orbital elements,
        physical properties (``diameter``, ``GM``), and epoch metadata.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"SBDB data file not found: {filepath}")

    df = pl.read_csv(filepath)

    # Rename columns for MPC compatibility
    df = df.rename(_COLUMN_RENAMES)

    # Cast number to Utf8 for parity with MPC DataFrame
    df = df.with_columns(pl.col("number").cast(pl.Utf8))

    logger.info("Loaded %d SBDB asteroid records from %s", len(df), filepath)
    return df
