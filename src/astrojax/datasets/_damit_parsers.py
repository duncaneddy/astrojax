"""Parsing utilities for the DAMIT asteroid shape model dataset.

Extracts CSV tables and shape files from the DAMIT complete export
tar.gz archive.  The archive contains ``tables/`` with CSV metadata
and ``files/`` with per-asteroid shape models.

The canonical data source is ``asteroid_models.csv`` which contains
all spin parameters (lambda, beta, period, etc.) alongside physical
properties, eliminating the need to parse individual ``spin.txt`` files.
"""

from __future__ import annotations

import functools
import io
import json
import logging
import re
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


@functools.lru_cache(maxsize=4)
def _find_extracted_prefix(extracted_dir: Path) -> str:
    """Find the top-level directory name inside an extracted DAMIT directory.

    After extraction the layout is ``extracted/<prefix>/tables/...``.
    This helper discovers *prefix* by listing the single top-level
    subdirectory.

    Args:
        extracted_dir: Root of the extracted archive.

    Returns:
        The top-level prefix directory name.

    Raises:
        ValueError: If the extracted directory is empty or ambiguous.
    """
    subdirs = [p for p in extracted_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(subdirs) == 0:
        raise ValueError(f"Extracted DAMIT directory is empty: {extracted_dir}")
    if len(subdirs) > 1:
        # Multiple prefixes — pick the first alphabetically for determinism.
        subdirs.sort()
    return subdirs[0].name


def build_shape_index(files_dir: Path) -> dict[int, str]:
    """Build a mapping from model ID to relative shape file path.

    Walks the ``files/`` directory tree looking for directories matching
    ``model_<N>`` that contain a ``shape.txt`` file.

    Args:
        files_dir: The ``files/`` directory inside the extracted DAMIT archive.

    Returns:
        Dictionary mapping model ID (int) to the relative path from
        *files_dir* to the ``shape.txt`` file (e.g.
        ``"asteroid_1/model_42/shape.txt"``).
    """
    model_re = re.compile(r"^model_(\d+)$")
    index: dict[int, str] = {}

    if not files_dir.is_dir():
        return index

    for asteroid_dir in files_dir.iterdir():
        if not asteroid_dir.is_dir():
            continue
        for model_dir in asteroid_dir.iterdir():
            if not model_dir.is_dir():
                continue
            m = model_re.match(model_dir.name)
            if m is None:
                continue
            shape_path = model_dir / "shape.txt"
            if shape_path.exists():
                rel = shape_path.relative_to(files_dir)
                index[int(m.group(1))] = str(rel)

    return index


def write_shape_index(extracted_dir: Path) -> Path:
    """Build and persist the shape index JSON file.

    Locates the ``files/`` directory inside *extracted_dir*, builds
    the model-ID-to-path mapping, and writes it as
    ``extracted_dir/shape_index.json``.

    Args:
        extracted_dir: Root of the extracted archive.

    Returns:
        Path to the written ``shape_index.json`` file.
    """
    prefix = _find_extracted_prefix(extracted_dir)
    files_dir = extracted_dir / prefix / "files"
    index = build_shape_index(files_dir)

    # Serialise with string keys for JSON compatibility
    json_path = extracted_dir / "shape_index.json"
    json_path.write_text(
        json.dumps({str(k): v for k, v in index.items()}, indent=None),
        encoding="utf-8",
    )
    logger.info("Wrote shape index with %d models to %s", len(index), json_path)
    return json_path


class _ShapePathIndex:
    """In-memory cache for the shape index.

    Loads ``shape_index.json`` from the extracted directory on first
    access and serves O(1) lookups thereafter.  If the JSON file is
    missing (e.g. pre-existing extraction), it is built and persisted
    automatically as a backward-compatibility fallback.
    """

    def __init__(self) -> None:
        self._index: dict[int, Path] | None = None
        self._extracted_dir: Path | None = None

    def _load(self, extracted_dir: Path) -> None:
        """Load or build the index for *extracted_dir*."""
        prefix = _find_extracted_prefix(extracted_dir)
        files_dir = extracted_dir / prefix / "files"

        json_path = extracted_dir / "shape_index.json"
        if not json_path.exists():
            # Backward compat: build and persist
            write_shape_index(extracted_dir)

        raw = json.loads(json_path.read_text(encoding="utf-8"))
        self._index = {int(k): files_dir / v for k, v in raw.items()}
        self._extracted_dir = extracted_dir

    def lookup(self, extracted_dir: Path, model_id: int) -> Path | None:
        """Return the absolute path to ``shape.txt`` for *model_id*, or ``None``.

        Args:
            extracted_dir: Root of the extracted archive.
            model_id: DAMIT model ID.

        Returns:
            Absolute path to the shape file, or ``None`` if the model
            is not in the index.
        """
        if self._index is None or self._extracted_dir != extracted_dir:
            self._load(extracted_dir)
        assert self._index is not None  # noqa: S101
        return self._index.get(model_id)

    def invalidate(self) -> None:
        """Clear the cached index so the next lookup reloads from disk."""
        self._index = None
        self._extracted_dir = None


_shape_index = _ShapePathIndex()
"""Module-level singleton for shape path lookups."""


def parse_damit_asteroids_table(
    filepath: str | Path,
    *,
    extracted_dir: str | Path | None = None,
) -> pl.DataFrame:
    """Parse the ``asteroids.csv`` table from a DAMIT tar.gz archive.

    Reads ``tables/asteroids.csv`` from the compressed archive and
    returns a Polars DataFrame with asteroid identity information.
    The ``created`` and ``modified`` timestamp columns are dropped.

    When *extracted_dir* is provided and contains the expected CSV file,
    it is read directly from disk (fast path) instead of decompressing
    the tar.gz archive.

    Args:
        filepath: Path to the DAMIT ``tar.gz`` archive file.
        extracted_dir: Optional path to the extracted archive directory.
            When provided, the CSV is read directly from disk.

    Returns:
        Polars DataFrame with columns: ``id`` (Int64), ``number``
        (Int64), ``name`` (Utf8), ``designation`` (Utf8), ``comment``
        (Utf8).

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If the archive does not contain ``asteroids.csv``.
    """
    # Fast path: read directly from extracted directory
    if extracted_dir is not None:
        extracted_dir = Path(extracted_dir)
        if extracted_dir.is_dir():
            prefix = _find_extracted_prefix(extracted_dir)
            csv_path = extracted_dir / prefix / "tables" / "asteroids.csv"
            if csv_path.exists():
                raw = csv_path.read_bytes()
                return _parse_asteroids_bytes(raw, source=str(csv_path))

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DAMIT archive not found: {filepath}")

    with tarfile.open(filepath, "r:gz") as tf:
        member_name = _find_table_member(tf, "asteroids.csv")
        raw = _extract_member_bytes(tf, member_name)

    return _parse_asteroids_bytes(raw, source=str(filepath))


def _parse_asteroids_bytes(raw: bytes, *, source: str) -> pl.DataFrame:
    """Parse raw asteroids CSV bytes into a Polars DataFrame.

    Args:
        raw: Raw CSV bytes.
        source: Description of the data source for logging.

    Returns:
        Polars DataFrame with asteroid identity columns.
    """
    df = pl.read_csv(
        io.BytesIO(raw),
        schema_overrides={"id": pl.Int64, "number": pl.Int64},
    )

    # Drop timestamp columns if present
    drop_cols = [c for c in ("created", "modified") if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    logger.info("Loaded %d DAMIT asteroid records from %s", len(df), source)
    return df


def parse_damit_models_table(
    filepath: str | Path,
    *,
    extracted_dir: str | Path | None = None,
) -> pl.DataFrame:
    """Parse the ``asteroid_models.csv`` table from a DAMIT tar.gz archive.

    This table is the canonical source for spin parameters (lambda, beta,
    period, yorp, jd0, phi0) as well as physical properties (diameter,
    albedo, thermal inertia) and model quality flags.

    When *extracted_dir* is provided and contains the expected CSV file,
    it is read directly from disk (fast path) instead of decompressing
    the tar.gz archive.

    Args:
        filepath: Path to the DAMIT ``tar.gz`` archive file.
        extracted_dir: Optional path to the extracted archive directory.
            When provided, the CSV is read directly from disk.

    Returns:
        Polars DataFrame with spin and physical property columns.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        ValueError: If the archive does not contain ``asteroid_models.csv``.
    """
    # Fast path: read directly from extracted directory
    if extracted_dir is not None:
        extracted_dir = Path(extracted_dir)
        if extracted_dir.is_dir():
            prefix = _find_extracted_prefix(extracted_dir)
            csv_path = extracted_dir / prefix / "tables" / "asteroid_models.csv"
            if csv_path.exists():
                raw = csv_path.read_bytes()
                return _parse_models_bytes(raw, source=str(csv_path))

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"DAMIT archive not found: {filepath}")

    with tarfile.open(filepath, "r:gz") as tf:
        member_name = _find_table_member(tf, "asteroid_models.csv")
        raw = _extract_member_bytes(tf, member_name)

    return _parse_models_bytes(raw, source=str(filepath))


def _parse_models_bytes(raw: bytes, *, source: str) -> pl.DataFrame:
    """Parse raw asteroid_models CSV bytes into a Polars DataFrame.

    Args:
        raw: Raw CSV bytes.
        source: Description of the data source for logging.

    Returns:
        Polars DataFrame with spin and physical property columns.
    """
    # Read everything as Utf8 first (infer_schema_length=0) to avoid
    # type-inference failures on mixed-format columns in the live data.
    df = pl.read_csv(
        io.BytesIO(raw),
        infer_schema_length=0,
        null_values=["", "NA", "NULL"],
    )

    # Cast columns to their expected types.  Only touch columns that exist
    # so this is robust to archive-version changes.
    int_cols = ["id", "asteroid_id"]
    float_cols = [
        "lambda",
        "beta",
        "period",
        "yorp",
        "jd0",
        "phi0",
        "lsm",
        "lsm_p1",
        "lsm_p2",
        "lsm_p3",
        "lsm_p4",
        "lsm_p5",
        "calibrated_size",
        "equiv_diameter",
        "equiv_diameter_err",
        "thermal_inertia",
        "thermal_inertia_min",
        "thermal_inertia_max",
        "visual_albedo",
        "visual_albedo_err",
        "craters_angle",
        "craters_surface_density",
        "quality_flag",
    ]

    cast_exprs: list[pl.Expr] = []
    for col in int_cols:
        if col in df.columns:
            cast_exprs.append(pl.col(col).cast(pl.Int64, strict=False))
    for col in float_cols:
        if col in df.columns:
            cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False))
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    # The nonconvex column may contain "true"/"false" or "0"/"1" depending
    # on the archive version.  Normalise to Boolean.
    if "nonconvex" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("nonconvex").is_in(["true", "1"]))
            .then(True)
            .when(pl.col("nonconvex").is_in(["false", "0"]))
            .then(False)
            .otherwise(None)
            .alias("nonconvex")
        )

    # Drop timestamp columns if present
    drop_cols = [c for c in ("created", "modified") if c in df.columns]
    if drop_cols:
        df = df.drop(drop_cols)

    logger.info("Loaded %d DAMIT model records from %s", len(df), source)
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


def load_shape_for_model(
    filepath: str | Path,
    model_id: int,
    *,
    extracted_dir: str | Path | None = None,
) -> tuple[Array, Array]:
    """Load the shape mesh for a specific DAMIT model.

    When *extracted_dir* is provided, looks for the shape file directly
    on disk (fast O(1) read).  Otherwise falls back to scanning the
    tar.gz archive.

    Args:
        filepath: Path to the DAMIT ``tar.gz`` archive file.
        model_id: DAMIT model ID (from the ``id`` column of
            ``asteroid_models.csv``).
        extracted_dir: Optional path to the extracted archive directory.
            When provided, the shape file is read directly from disk.

    Returns:
        Tuple of ``(vertices, facets)`` as JAX arrays.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        KeyError: If no shape file is found for the given *model_id*.
    """
    # Fast path: read directly from extracted directory
    if extracted_dir is not None:
        extracted_dir = Path(extracted_dir)
        if extracted_dir.is_dir():
            shape_path = _find_extracted_shape(extracted_dir, model_id)
            if shape_path is not None:
                text = shape_path.read_text(encoding="utf-8")
                return parse_shape_file(text)

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


def _find_extracted_shape(extracted_dir: Path, model_id: int) -> Path | None:
    """Find the shape.txt file for a model in the extracted directory.

    Uses the persisted shape index for O(1) lookups instead of globbing
    across thousands of asteroid subdirectories.

    Args:
        extracted_dir: Root of the extracted archive.
        model_id: DAMIT model ID.

    Returns:
        Path to the shape file, or ``None`` if not found.
    """
    return _shape_index.lookup(extracted_dir, model_id)
