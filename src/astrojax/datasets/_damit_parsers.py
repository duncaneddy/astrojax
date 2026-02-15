"""Parsing utilities for the DAMIT asteroid shape model dataset.

Extracts CSV tables and shape files from the DAMIT complete export
tar.gz archive.  The archive contains ``tables/`` with CSV metadata
and ``files/`` with per-asteroid shape models.

The canonical data source is ``asteroid_models.csv`` which contains
all spin parameters (lambda, beta, period, etc.) alongside physical
properties, eliminating the need to parse individual ``spin.txt`` files.
"""

from __future__ import annotations

import io
import logging
import tarfile
from pathlib import Path

import jax.numpy as jnp
import polars as pl
from jax import Array

logger = logging.getLogger(__name__)


def _find_tar_prefix(tf: tarfile.TarFile) -> str:
    """Find the top-level directory name inside the tar archive.

    DAMIT archives are named ``damit-YYYYMMDDTHHMMSSZ/`` at the top
    level.  This helper inspects the first member to discover that
    prefix so that subsequent lookups do not need to hard-code it.

    Args:
        tf: An open :class:`tarfile.TarFile` instance.

    Returns:
        The top-level directory prefix (without trailing ``/``).

    Raises:
        ValueError: If the archive appears to be empty.
    """
    for member in tf.getmembers():
        parts = member.name.split("/")
        if parts[0]:
            return parts[0]
    raise ValueError("DAMIT tar archive appears to be empty")


def _find_table_member(tf: tarfile.TarFile, table_name: str) -> str:
    """Find the full member path for a table CSV file.

    Searches for ``<prefix>/tables/<table_name>`` regardless of the
    archive's top-level directory name.

    Args:
        tf: An open :class:`tarfile.TarFile` instance.
        table_name: Filename to find (e.g. ``"asteroids.csv"``).

    Returns:
        Full member path inside the archive.

    Raises:
        ValueError: If no matching member is found.
    """
    suffix = f"tables/{table_name}"
    for member in tf.getmembers():
        if member.name.endswith(suffix):
            return member.name
    raise ValueError(f"DAMIT archive does not contain 'tables/{table_name}'")


def _extract_member_bytes(tf: tarfile.TarFile, member_name: str) -> bytes:
    """Extract a single tar member as raw bytes.

    Args:
        tf: An open :class:`tarfile.TarFile` instance.
        member_name: Full path of the member inside the archive.

    Returns:
        Raw bytes of the extracted member.

    Raises:
        KeyError: If the member does not exist.
    """
    f = tf.extractfile(member_name)
    if f is None:
        raise KeyError(f"Cannot extract member '{member_name}' (may be a directory)")
    return f.read()


def _extract_member_text(tf: tarfile.TarFile, member_name: str) -> str:
    """Extract a single tar member as UTF-8 text.

    Args:
        tf: An open :class:`tarfile.TarFile` instance.
        member_name: Full path of the member inside the archive.

    Returns:
        Decoded text content of the member.
    """
    return _extract_member_bytes(tf, member_name).decode("utf-8")


def parse_damit_asteroids_table(filepath: str | Path) -> pl.DataFrame:
    """Parse the ``asteroids.csv`` table from a DAMIT tar.gz archive.

    Reads ``tables/asteroids.csv`` from the compressed archive and
    returns a Polars DataFrame with asteroid identity information.
    The ``created`` and ``modified`` timestamp columns are dropped.

    Args:
        filepath: Path to the DAMIT ``tar.gz`` archive file.

    Returns:
        Polars DataFrame with columns: ``id`` (Int64), ``number``
        (Int64), ``name`` (Utf8), ``designation`` (Utf8), ``comment``
        (Utf8).

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If the archive does not contain ``asteroids.csv``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DAMIT archive not found: {filepath}")

    with tarfile.open(filepath, "r:gz") as tf:
        member_name = _find_table_member(tf, "asteroids.csv")
        raw = _extract_member_bytes(tf, member_name)

    df = pl.read_csv(
        io.BytesIO(raw),
        schema_overrides={"id": pl.Int64, "number": pl.Int64},
    )

    # Drop timestamp columns if present
    drop_cols = [c for c in ("created", "modified") if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    logger.info("Loaded %d DAMIT asteroid records from %s", len(df), filepath)
    return df


def parse_damit_models_table(filepath: str | Path) -> pl.DataFrame:
    """Parse the ``asteroid_models.csv`` table from a DAMIT tar.gz archive.

    This table is the canonical source for spin parameters (lambda, beta,
    period, yorp, jd0, phi0) as well as physical properties (diameter,
    albedo, thermal inertia) and model quality flags.

    Args:
        filepath: Path to the DAMIT ``tar.gz`` archive file.

    Returns:
        Polars DataFrame with spin and physical property columns.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If the archive does not contain ``asteroid_models.csv``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DAMIT archive not found: {filepath}")

    with tarfile.open(filepath, "r:gz") as tf:
        member_name = _find_table_member(tf, "asteroid_models.csv")
        raw = _extract_member_bytes(tf, member_name)

    df = pl.read_csv(
        io.BytesIO(raw),
        schema_overrides={
            "id": pl.Int64,
            "asteroid_id": pl.Int64,
            "nonconvex": pl.Boolean,
        },
        null_values=["", "NA", "NULL"],
    )

    # Drop timestamp columns if present
    drop_cols = [c for c in ("created", "modified") if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    logger.info("Loaded %d DAMIT model records from %s", len(df), filepath)
    return df


def parse_shape_file(shape_text: str) -> tuple[Array, Array]:
    """Parse a DAMIT ``shape.txt`` file into vertices and facets.

    The file format is:

    - Line 1: ``N_vertices N_facets``
    - Next *N_vertices* lines: ``x y z`` whitespace-separated coordinates
    - Next *N_facets* lines: ``v1 v2 v3`` whitespace-separated 1-indexed
      vertex indices

    Facet indices are converted from 1-indexed to 0-indexed.

    Args:
        shape_text: Raw text content of a ``shape.txt`` file.

    Returns:
        Tuple of ``(vertices, facets)`` where *vertices* is a
        ``(N, 3)`` float32 JAX array and *facets* is a ``(M, 3)``
        int32 JAX array with 0-indexed vertex references.

    Raises:
        ValueError: If the file format is invalid.
    """
    lines = shape_text.strip().splitlines()
    if not lines:
        raise ValueError("Empty shape file")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Invalid shape file header: {lines[0]!r}")

    n_verts = int(header[0])
    n_facets = int(header[1])

    if len(lines) < 1 + n_verts + n_facets:
        raise ValueError(
            f"Shape file too short: expected {1 + n_verts + n_facets} lines, got {len(lines)}"
        )

    # Parse vertices
    vert_data = []
    for i in range(1, 1 + n_verts):
        parts = lines[i].split()
        vert_data.append([float(parts[0]), float(parts[1]), float(parts[2])])

    # Parse facets (convert 1-indexed to 0-indexed)
    facet_data = []
    for i in range(1 + n_verts, 1 + n_verts + n_facets):
        parts = lines[i].split()
        facet_data.append([int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1])

    vertices = jnp.array(vert_data, dtype=jnp.float32)
    facets = jnp.array(facet_data, dtype=jnp.int32)

    return vertices, facets


def load_shape_for_model(filepath: str | Path, model_id: int) -> tuple[Array, Array]:
    """Load the shape mesh for a specific DAMIT model from a tar.gz archive.

    Searches the archive for ``model_{model_id}/shape.txt`` and parses
    it into vertices and facets.

    Args:
        filepath: Path to the DAMIT ``tar.gz`` archive file.
        model_id: DAMIT model ID (from the ``id`` column of
            ``asteroid_models.csv``).

    Returns:
        Tuple of ``(vertices, facets)`` as JAX arrays.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        KeyError: If no shape file is found for the given *model_id*.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DAMIT archive not found: {filepath}")

    target_suffix = f"model_{model_id}/shape.txt"

    with tarfile.open(filepath, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name.endswith(target_suffix):
                text = _extract_member_text(tf, member.name)
                return parse_shape_file(text)

    raise KeyError(f"No shape file found for model_id={model_id} in {filepath}")
