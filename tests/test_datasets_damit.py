"""Tests for the astrojax.datasets DAMIT asteroid shape model module."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from astrojax.datasets._damit_download import (
    _EXTRACTED_MARKER,
    _FILENAME,
    extract_damit_archive,
)
from astrojax.datasets._damit_parsers import (
    _find_extracted_prefix,
    _ShapePathIndex,
    build_shape_index,
    load_shape_for_model,
    parse_damit_asteroids_table,
    parse_damit_models_table,
    parse_shape_file,
    write_shape_index,
)
from astrojax.datasets._damit_providers import (
    _is_extraction_stale,
    get_damit_shape,
    get_damit_spin,
    load_damit_asteroids,
    load_damit_models,
)
from astrojax.datasets._damit_shapes import (
    _HAS_TRIMESH,
    compute_spherical_uvs,
    export_shape_glb,
    export_shape_glb_textured,
    export_shape_stl,
    shape_to_trimesh,
)
from astrojax.datasets._damit_spin import (
    damit_spin_to_rotation,
    rotate_shape_points,
    scale_shape_vertices,
)

# ---------------------------------------------------------------------------
# Synthetic test data helpers
# ---------------------------------------------------------------------------

_ASTEROIDS_CSV = """\
id,number,name,designation,comment,created,modified
1,433,Eros,,Near-Earth asteroid,2020-01-01,2020-06-01
2,25143,Itokawa,1998 SF36,Hayabusa target,2020-01-01,2020-06-01
3,101955,Bennu,1999 RQ36,OSIRIS-REx target,2020-01-01,2020-06-01
"""

_MODELS_CSV = """\
id,asteroid_id,lambda,beta,period,yorp,jd0,phi0,lsm,lsm_p1,lsm_p2,lsm_p3,lsm_p4,lsm_p5,calibrated_size,equiv_diameter,equiv_diameter_err,thermal_inertia,thermal_inertia_min,thermal_inertia_max,visual_albedo,visual_albedo_err,quality_flag,nonconvex,created,modified
10,1,11.35,17.22,5.27025,0.0,2451545.0,0.0,0.01,0.02,,,,,,16.84,0.5,,,,0.25,0.02,1,false,2020-01-01,2020-06-01
11,1,191.35,-17.22,5.27026,0.0,2451545.0,180.0,0.015,0.025,,,,,,16.84,0.5,,,,0.25,0.02,2,false,2020-01-01,2020-06-01
20,2,128.5,-89.66,12.1324,0.0001,2452000.0,45.0,0.008,0.01,,,,,,0.535,0.01,,,,0.19,0.03,1,true,2020-01-01,2020-06-01
30,3,85.0,-65.0,4.29746,0.0,2454000.0,0.0,0.005,0.008,,,,,,0.49,0.02,,,,0.046,0.005,1,false,2020-01-01,2020-06-01
"""

_SHAPE_TEXT = """\
4 2
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
-1.0 0.0 0.0
1 2 3
2 3 4
"""

_SHAPE_TEXT_SIMPLE = """\
3 1
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
1 2 3
"""


def _make_damit_tar(
    filepath: Path,
    *,
    asteroids_csv: str = _ASTEROIDS_CSV,
    models_csv: str = _MODELS_CSV,
    shape_models: dict[int, str] | None = None,
    prefix: str = "damit-20260214T010301Z",
) -> None:
    """Create a synthetic DAMIT tar.gz archive for testing.

    Args:
        filepath: Where to write the tar.gz file.
        asteroids_csv: Content for ``tables/asteroids.csv``.
        models_csv: Content for ``tables/asteroid_models.csv``.
        shape_models: Mapping of ``{model_id: shape_text}`` to include.
        prefix: Top-level directory name inside the archive.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if shape_models is None:
        shape_models = {10: _SHAPE_TEXT}

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        # Add directory entry for prefix
        dir_info = tarfile.TarInfo(name=f"{prefix}/")
        dir_info.type = tarfile.DIRTYPE
        dir_info.mode = 0o755
        tf.addfile(dir_info)

        # Add tables directory
        tables_dir = tarfile.TarInfo(name=f"{prefix}/tables/")
        tables_dir.type = tarfile.DIRTYPE
        tables_dir.mode = 0o755
        tf.addfile(tables_dir)

        # Add asteroids.csv
        ast_bytes = asteroids_csv.encode("utf-8")
        ast_info = tarfile.TarInfo(name=f"{prefix}/tables/asteroids.csv")
        ast_info.size = len(ast_bytes)
        ast_info.mode = 0o644
        tf.addfile(ast_info, io.BytesIO(ast_bytes))

        # Add asteroid_models.csv
        mod_bytes = models_csv.encode("utf-8")
        mod_info = tarfile.TarInfo(name=f"{prefix}/tables/asteroid_models.csv")
        mod_info.size = len(mod_bytes)
        mod_info.mode = 0o644
        tf.addfile(mod_info, io.BytesIO(mod_bytes))

        # Add intermediate directories for shape files
        files_dir = tarfile.TarInfo(name=f"{prefix}/files/")
        files_dir.type = tarfile.DIRTYPE
        files_dir.mode = 0o755
        tf.addfile(files_dir)

        ast1_dir = tarfile.TarInfo(name=f"{prefix}/files/asteroid_1/")
        ast1_dir.type = tarfile.DIRTYPE
        ast1_dir.mode = 0o755
        tf.addfile(ast1_dir)

        # Add shape files
        for model_id, shape_text in shape_models.items():
            # Add model directory entry
            model_dir = tarfile.TarInfo(name=f"{prefix}/files/asteroid_1/model_{model_id}/")
            model_dir.type = tarfile.DIRTYPE
            model_dir.mode = 0o755
            tf.addfile(model_dir)

            shape_bytes = shape_text.encode("utf-8")
            shape_info = tarfile.TarInfo(
                name=f"{prefix}/files/asteroid_1/model_{model_id}/shape.txt"
            )
            shape_info.size = len(shape_bytes)
            shape_info.mode = 0o644
            tf.addfile(shape_info, io.BytesIO(shape_bytes))

    filepath.write_bytes(buf.getvalue())


# ---------------------------------------------------------------------------
# Parser tests — shape file
# ---------------------------------------------------------------------------


class TestParseShapeFile:
    """Tests for parse_shape_file."""

    def test_vertex_count(self) -> None:
        """Parsed vertices should have expected count."""
        verts, _ = parse_shape_file(_SHAPE_TEXT)
        assert verts.shape == (4, 3)

    def test_facet_count(self) -> None:
        """Parsed facets should have expected count."""
        _, facets = parse_shape_file(_SHAPE_TEXT)
        assert facets.shape == (2, 3)

    def test_zero_indexed_facets(self) -> None:
        """Facet indices should be converted from 1-indexed to 0-indexed."""
        _, facets = parse_shape_file(_SHAPE_TEXT)
        # First facet was "1 2 3" -> [0, 1, 2]
        assert jnp.array_equal(facets[0], jnp.array([0, 1, 2], dtype=jnp.int32))

    def test_vertex_values(self) -> None:
        """Vertex coordinates should match input data."""
        verts, _ = parse_shape_file(_SHAPE_TEXT)
        assert jnp.allclose(verts[0], jnp.array([1.0, 0.0, 0.0]))
        assert jnp.allclose(verts[3], jnp.array([-1.0, 0.0, 0.0]))

    def test_vertex_dtype(self) -> None:
        """Vertices should be float32."""
        verts, _ = parse_shape_file(_SHAPE_TEXT)
        assert verts.dtype == jnp.float32

    def test_facet_dtype(self) -> None:
        """Facets should be int32."""
        _, facets = parse_shape_file(_SHAPE_TEXT)
        assert facets.dtype == jnp.int32

    def test_empty_file_raises(self) -> None:
        """An empty file should raise ValueError."""
        with pytest.raises(ValueError, match="Empty shape file"):
            parse_shape_file("")

    def test_bad_header_raises(self) -> None:
        """A header with fewer than 2 numbers should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid shape file header"):
            parse_shape_file("5\n")

    def test_truncated_file_raises(self) -> None:
        """A file shorter than expected should raise ValueError."""
        with pytest.raises(ValueError, match="too short"):
            parse_shape_file("3 1\n1.0 0.0 0.0\n")

    def test_simple_triangle(self) -> None:
        """A minimal single-triangle shape should parse correctly."""
        verts, facets = parse_shape_file(_SHAPE_TEXT_SIMPLE)
        assert verts.shape == (3, 3)
        assert facets.shape == (1, 3)
        assert jnp.array_equal(facets[0], jnp.array([0, 1, 2], dtype=jnp.int32))


# ---------------------------------------------------------------------------
# Parser tests — CSV tables
# ---------------------------------------------------------------------------


class TestParseDamitAsteroidsTable:
    """Tests for parse_damit_asteroids_table."""

    def test_row_count(self, tmp_path: Path) -> None:
        """Should load all rows from the synthetic data."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_asteroids_table(fp)
        assert df.shape[0] == 3

    def test_columns(self, tmp_path: Path) -> None:
        """Should contain expected columns and not timestamps."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_asteroids_table(fp)
        assert "id" in df.columns
        assert "number" in df.columns
        assert "name" in df.columns
        assert "designation" in df.columns
        assert "comment" in df.columns
        assert "created" not in df.columns
        assert "modified" not in df.columns

    def test_dtypes(self, tmp_path: Path) -> None:
        """ID and number columns should be Int64."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_asteroids_table(fp)
        assert df["id"].dtype == pl.Int64
        assert df["number"].dtype == pl.Int64

    def test_values(self, tmp_path: Path) -> None:
        """Verify specific row values."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_asteroids_table(fp)
        eros = df.filter(pl.col("number") == 433).row(0, named=True)
        assert eros["name"] == "Eros"
        assert eros["id"] == 1

    def test_missing_file(self, tmp_path: Path) -> None:
        """Raise FileNotFoundError for nonexistent archive."""
        with pytest.raises(FileNotFoundError):
            parse_damit_asteroids_table(tmp_path / "nonexistent.tar.gz")

    def test_missing_table(self, tmp_path: Path) -> None:
        """Raise ValueError when the archive lacks asteroids.csv."""
        fp = tmp_path / "bad.tar.gz"
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            data = b"hello"
            info = tarfile.TarInfo(name="damit/tables/other.csv")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        fp.write_bytes(buf.getvalue())

        with pytest.raises(ValueError, match="does not contain"):
            parse_damit_asteroids_table(fp)


class TestParseDamitModelsTable:
    """Tests for parse_damit_models_table."""

    def test_row_count(self, tmp_path: Path) -> None:
        """Should load all model rows."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_models_table(fp)
        assert df.shape[0] == 4

    def test_spin_columns_present(self, tmp_path: Path) -> None:
        """All spin parameter columns should be present."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_models_table(fp)
        for col in ("lambda", "beta", "period", "yorp", "jd0", "phi0"):
            assert col in df.columns, f"Missing column: {col}"

    def test_physical_columns_present(self, tmp_path: Path) -> None:
        """Physical property columns should be present."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_models_table(fp)
        for col in ("equiv_diameter", "visual_albedo", "quality_flag"):
            assert col in df.columns, f"Missing column: {col}"

    def test_no_timestamps(self, tmp_path: Path) -> None:
        """Timestamp columns should be dropped."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_models_table(fp)
        assert "created" not in df.columns
        assert "modified" not in df.columns

    def test_spin_values(self, tmp_path: Path) -> None:
        """Verify specific spin parameter values."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_models_table(fp)
        model_10 = df.filter(pl.col("id") == 10).row(0, named=True)
        assert abs(model_10["lambda"] - 11.35) < 0.01
        assert abs(model_10["beta"] - 17.22) < 0.01
        assert abs(model_10["period"] - 5.27025) < 0.0001

    def test_yorp_value(self, tmp_path: Path) -> None:
        """Models with nonzero YORP should have the correct value."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_models_table(fp)
        model_20 = df.filter(pl.col("id") == 20).row(0, named=True)
        assert abs(model_20["yorp"] - 0.0001) < 1e-6

    def test_nonconvex_column(self, tmp_path: Path) -> None:
        """The nonconvex column should be boolean."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_models_table(fp)
        assert df["nonconvex"].dtype == pl.Boolean
        model_20 = df.filter(pl.col("id") == 20).row(0, named=True)
        assert model_20["nonconvex"] is True

    def test_multiple_models_per_asteroid(self, tmp_path: Path) -> None:
        """Asteroid_id=1 should have two models."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_models_table(fp)
        subset = df.filter(pl.col("asteroid_id") == 1)
        assert len(subset) == 2


# ---------------------------------------------------------------------------
# Parser tests — shape from tar
# ---------------------------------------------------------------------------


class TestLoadShapeForModel:
    """Tests for load_shape_for_model."""

    def test_loads_shape(self, tmp_path: Path) -> None:
        """Should load vertices and facets for an existing model."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp, shape_models={10: _SHAPE_TEXT})
        verts, facets = load_shape_for_model(fp, 10)
        assert verts.shape == (4, 3)
        assert facets.shape == (2, 3)

    def test_model_not_found(self, tmp_path: Path) -> None:
        """Should raise KeyError for a nonexistent model_id."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        with pytest.raises(KeyError, match="model_id=9999"):
            load_shape_for_model(fp, 9999)

    def test_missing_file(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for nonexistent archive."""
        with pytest.raises(FileNotFoundError):
            load_shape_for_model(Path("/nonexistent.tar.gz"), 10)


# ---------------------------------------------------------------------------
# Provider tests — caching
# ---------------------------------------------------------------------------


class TestLoadDamitAsteroids:
    """Tests for load_damit_asteroids caching behavior."""

    def test_uses_existing_fresh_file(self, tmp_path: Path) -> None:
        """Should load from file without downloading if fresh."""
        fp = tmp_path / "damit" / _FILENAME
        _make_damit_tar(fp)
        df = load_damit_asteroids(fp, max_age_days=999.0)
        assert df.shape[0] == 3

    def test_raises_when_no_cache_and_download_fails(self, tmp_path: Path) -> None:
        """Should raise RuntimeError when download fails and no cache exists."""
        fp = tmp_path / "damit" / "nonexistent.tar.gz"

        with patch(
            "astrojax.datasets._damit_providers.download_damit_file",
            side_effect=ConnectionError("mocked"),
        ):
            with pytest.raises(RuntimeError, match="Failed to download DAMIT"):
                load_damit_asteroids(fp, max_age_days=0.0)

    def test_falls_back_to_stale_cache(self, tmp_path: Path) -> None:
        """Should fall back to stale cache when download fails."""
        fp = tmp_path / "damit" / _FILENAME
        _make_damit_tar(fp)

        with patch(
            "astrojax.datasets._damit_providers.download_damit_file",
            side_effect=ConnectionError("mocked"),
        ):
            df = load_damit_asteroids(fp, max_age_days=0.0)
            assert df.shape[0] == 3


class TestLoadDamitModels:
    """Tests for load_damit_models caching behavior."""

    def test_uses_existing_fresh_file(self, tmp_path: Path) -> None:
        """Should load from file without downloading if fresh."""
        fp = tmp_path / "damit" / _FILENAME
        _make_damit_tar(fp)
        df = load_damit_models(fp, max_age_days=999.0)
        assert df.shape[0] == 4

    def test_raises_when_no_cache_and_download_fails(self, tmp_path: Path) -> None:
        """Should raise RuntimeError when download fails and no cache exists."""
        fp = tmp_path / "damit" / "nonexistent.tar.gz"

        with patch(
            "astrojax.datasets._damit_providers.download_damit_file",
            side_effect=ConnectionError("mocked"),
        ):
            with pytest.raises(RuntimeError, match="Failed to download DAMIT"):
                load_damit_models(fp, max_age_days=0.0)


# ---------------------------------------------------------------------------
# Provider tests — lookup
# ---------------------------------------------------------------------------


class TestGetDamitSpin:
    """Tests for get_damit_spin."""

    def test_basic_lookup(self, tmp_path: Path) -> None:
        """Should return spin params for an existing asteroid."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_models_table(fp)
        spin = get_damit_spin(df, asteroid_id=1)
        assert abs(spin["lambda"] - 11.35) < 0.01
        assert abs(spin["beta"] - 17.22) < 0.01
        assert abs(spin["period"] - 5.27025) < 0.0001

    def test_not_found(self, tmp_path: Path) -> None:
        """Should raise KeyError for nonexistent asteroid_id."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_models_table(fp)
        with pytest.raises(KeyError, match="No DAMIT models"):
            get_damit_spin(df, asteroid_id=9999)

    def test_model_index(self, tmp_path: Path) -> None:
        """Should return the second model when model_index=1."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_models_table(fp)
        spin = get_damit_spin(df, asteroid_id=1, model_index=1)
        assert spin["id"] == 11
        assert abs(spin["phi0"] - 180.0) < 0.01

    def test_model_index_out_of_range(self, tmp_path: Path) -> None:
        """Should raise KeyError for out-of-range model_index."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_models_table(fp)
        with pytest.raises(KeyError, match="model_index=5 out of range"):
            get_damit_spin(df, asteroid_id=1, model_index=5)


class TestGetDamitShape:
    """Tests for get_damit_shape."""

    def test_loads_shape(self, tmp_path: Path) -> None:
        """Should load vertices and facets."""
        fp = tmp_path / "damit" / _FILENAME
        _make_damit_tar(fp, shape_models={10: _SHAPE_TEXT})
        verts, facets = get_damit_shape(fp, model_id=10, max_age_days=999.0)
        assert verts.shape == (4, 3)
        assert facets.shape == (2, 3)

    def test_model_not_found(self, tmp_path: Path) -> None:
        """Should raise KeyError for nonexistent model."""
        fp = tmp_path / "damit" / _FILENAME
        _make_damit_tar(fp)
        with pytest.raises(KeyError):
            get_damit_shape(fp, model_id=9999, max_age_days=999.0)


# ---------------------------------------------------------------------------
# Archive extraction tests
# ---------------------------------------------------------------------------


class TestExtractDamitArchive:
    """Tests for extract_damit_archive."""

    def test_extracts_files(self, tmp_path: Path) -> None:
        """Should extract the archive contents to the target directory."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        # Check that the prefix directory exists
        subdirs = [p for p in extract_dir.iterdir() if p.is_dir()]
        assert len(subdirs) == 1
        assert subdirs[0].name == "damit-20260214T010301Z"

    def test_marker_file_created(self, tmp_path: Path) -> None:
        """Should create a .extracted marker file after extraction."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        marker = extract_dir / _EXTRACTED_MARKER
        assert marker.exists()

    def test_tables_extracted(self, tmp_path: Path) -> None:
        """Should extract CSV tables to the correct location."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        prefix = "damit-20260214T010301Z"
        assert (extract_dir / prefix / "tables" / "asteroids.csv").exists()
        assert (extract_dir / prefix / "tables" / "asteroid_models.csv").exists()

    def test_shapes_extracted(self, tmp_path: Path) -> None:
        """Should extract shape files to the correct location."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp, shape_models={10: _SHAPE_TEXT, 20: _SHAPE_TEXT_SIMPLE})
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        prefix = "damit-20260214T010301Z"
        assert (extract_dir / prefix / "files" / "asteroid_1" / "model_10" / "shape.txt").exists()
        assert (extract_dir / prefix / "files" / "asteroid_1" / "model_20" / "shape.txt").exists()

    def test_missing_archive_raises(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for nonexistent archive."""
        with pytest.raises(FileNotFoundError):
            extract_damit_archive(tmp_path / "nonexistent.tar.gz", tmp_path / "out")

    def test_creates_extract_dir(self, tmp_path: Path) -> None:
        """Should create the extraction directory if it doesn't exist."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        extract_dir = tmp_path / "deep" / "nested" / "extracted"
        extract_damit_archive(fp, extract_dir)
        assert extract_dir.is_dir()


class TestIsExtractionStale:
    """Tests for _is_extraction_stale."""

    def test_missing_marker_is_stale(self, tmp_path: Path) -> None:
        """Should return True when the marker file doesn't exist."""
        tar_path = tmp_path / "test.tar.gz"
        tar_path.write_bytes(b"dummy")
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        assert _is_extraction_stale(tar_path, extract_dir) is True

    def test_fresh_extraction_is_not_stale(self, tmp_path: Path) -> None:
        """Should return False when marker is newer than the tar.gz."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)
        assert _is_extraction_stale(fp, extract_dir) is False

    def test_updated_tar_makes_extraction_stale(self, tmp_path: Path) -> None:
        """Should return True when tar.gz is newer than the marker."""
        import os
        import time

        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        # Make the tar newer than the marker
        time.sleep(0.05)
        _make_damit_tar(fp)
        # Ensure mtime is actually updated
        os.utime(fp, None)
        assert _is_extraction_stale(fp, extract_dir) is True


# ---------------------------------------------------------------------------
# Fast-path parser tests (reading from extracted directory)
# ---------------------------------------------------------------------------


class TestParseDamitAsteroidsTableFastPath:
    """Tests for parse_damit_asteroids_table with extracted_dir."""

    def test_reads_from_extracted_dir(self, tmp_path: Path) -> None:
        """Should load asteroids from extracted directory."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        df = parse_damit_asteroids_table(fp, extracted_dir=extract_dir)
        assert df.shape[0] == 3

    def test_same_result_as_tar_path(self, tmp_path: Path) -> None:
        """Fast-path and tar-path should produce identical DataFrames."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        df_tar = parse_damit_asteroids_table(fp)
        df_fast = parse_damit_asteroids_table(fp, extracted_dir=extract_dir)
        assert df_tar.equals(df_fast)

    def test_falls_back_to_tar_when_no_extracted_dir(self, tmp_path: Path) -> None:
        """Should fall back to tar when extracted_dir doesn't exist."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        df = parse_damit_asteroids_table(fp, extracted_dir=tmp_path / "nonexistent")
        assert df.shape[0] == 3


class TestParseDamitModelsTableFastPath:
    """Tests for parse_damit_models_table with extracted_dir."""

    def test_reads_from_extracted_dir(self, tmp_path: Path) -> None:
        """Should load models from extracted directory."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        df = parse_damit_models_table(fp, extracted_dir=extract_dir)
        assert df.shape[0] == 4

    def test_same_result_as_tar_path(self, tmp_path: Path) -> None:
        """Fast-path and tar-path should produce identical DataFrames."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        df_tar = parse_damit_models_table(fp)
        df_fast = parse_damit_models_table(fp, extracted_dir=extract_dir)
        assert df_tar.equals(df_fast)


class TestLoadShapeForModelFastPath:
    """Tests for load_shape_for_model with extracted_dir."""

    def test_reads_from_extracted_dir(self, tmp_path: Path) -> None:
        """Should load shape from extracted directory."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp, shape_models={10: _SHAPE_TEXT})
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        verts, facets = load_shape_for_model(fp, 10, extracted_dir=extract_dir)
        assert verts.shape == (4, 3)
        assert facets.shape == (2, 3)

    def test_same_result_as_tar_path(self, tmp_path: Path) -> None:
        """Fast-path and tar-path should produce identical arrays."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp, shape_models={10: _SHAPE_TEXT})
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        verts_tar, facets_tar = load_shape_for_model(fp, 10)
        verts_fast, facets_fast = load_shape_for_model(fp, 10, extracted_dir=extract_dir)
        assert jnp.array_equal(verts_tar, verts_fast)
        assert jnp.array_equal(facets_tar, facets_fast)

    def test_model_not_found_in_extracted_dir(self, tmp_path: Path) -> None:
        """Should raise KeyError for nonexistent model in extracted dir."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        with pytest.raises(KeyError, match="model_id=9999"):
            load_shape_for_model(fp, 9999, extracted_dir=extract_dir)

    def test_falls_back_to_tar_when_no_extracted_dir(self, tmp_path: Path) -> None:
        """Should fall back to tar when extracted_dir doesn't exist."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp, shape_models={10: _SHAPE_TEXT})
        verts, facets = load_shape_for_model(fp, 10, extracted_dir=tmp_path / "nonexistent")
        assert verts.shape == (4, 3)
        assert facets.shape == (2, 3)

    def test_multiple_shapes(self, tmp_path: Path) -> None:
        """Should load different shapes from extracted directory."""
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp, shape_models={10: _SHAPE_TEXT, 20: _SHAPE_TEXT_SIMPLE})
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        verts_10, facets_10 = load_shape_for_model(fp, 10, extracted_dir=extract_dir)
        verts_20, facets_20 = load_shape_for_model(fp, 20, extracted_dir=extract_dir)
        assert verts_10.shape == (4, 3)
        assert verts_20.shape == (3, 3)


class TestProviderExtractionIntegration:
    """Tests for provider functions using extraction."""

    def test_get_damit_shape_extracts_and_reads(self, tmp_path: Path) -> None:
        """get_damit_shape should extract the archive and read from disk."""
        fp = tmp_path / "damit" / _FILENAME
        _make_damit_tar(fp, shape_models={10: _SHAPE_TEXT})

        verts, facets = get_damit_shape(fp, model_id=10, max_age_days=999.0)
        assert verts.shape == (4, 3)
        assert facets.shape == (2, 3)

        # Verify extraction happened
        extract_dir = fp.parent / "extracted"
        assert extract_dir.is_dir()
        assert (extract_dir / _EXTRACTED_MARKER).exists()

    def test_load_damit_asteroids_uses_extracted(self, tmp_path: Path) -> None:
        """load_damit_asteroids should extract and use the fast path."""
        fp = tmp_path / "damit" / _FILENAME
        _make_damit_tar(fp)

        df = load_damit_asteroids(fp, max_age_days=999.0)
        assert df.shape[0] == 3

        # Verify extraction happened
        extract_dir = fp.parent / "extracted"
        assert extract_dir.is_dir()

    def test_load_damit_models_uses_extracted(self, tmp_path: Path) -> None:
        """load_damit_models should extract and use the fast path."""
        fp = tmp_path / "damit" / _FILENAME
        _make_damit_tar(fp)

        df = load_damit_models(fp, max_age_days=999.0)
        assert df.shape[0] == 4

        # Verify extraction happened
        extract_dir = fp.parent / "extracted"
        assert extract_dir.is_dir()

    def test_re_extraction_after_delete(self, tmp_path: Path) -> None:
        """Deleting the extracted dir should trigger re-extraction on next call."""
        import shutil

        fp = tmp_path / "damit" / _FILENAME
        _make_damit_tar(fp, shape_models={10: _SHAPE_TEXT})

        # First call: extract
        verts1, _ = get_damit_shape(fp, model_id=10, max_age_days=999.0)
        extract_dir = fp.parent / "extracted"
        assert extract_dir.is_dir()

        # Delete extracted dir
        shutil.rmtree(extract_dir)
        assert not extract_dir.exists()

        # Second call: should re-extract transparently
        verts2, _ = get_damit_shape(fp, model_id=10, max_age_days=999.0)
        assert extract_dir.is_dir()
        assert jnp.array_equal(verts1, verts2)


# ---------------------------------------------------------------------------
# Shape index tests
# ---------------------------------------------------------------------------


class TestBuildShapeIndex:
    """Tests for build_shape_index."""

    def test_maps_model_ids_to_paths(self, tmp_path: Path) -> None:
        """Should map model IDs to relative paths from files_dir."""
        files_dir = tmp_path / "files"
        (files_dir / "asteroid_1" / "model_10").mkdir(parents=True)
        (files_dir / "asteroid_1" / "model_10" / "shape.txt").write_text("dummy")
        (files_dir / "asteroid_2" / "model_20").mkdir(parents=True)
        (files_dir / "asteroid_2" / "model_20" / "shape.txt").write_text("dummy")

        index = build_shape_index(files_dir)
        assert index == {
            10: "asteroid_1/model_10/shape.txt",
            20: "asteroid_2/model_20/shape.txt",
        }

    def test_skips_dirs_without_shape(self, tmp_path: Path) -> None:
        """Should skip model directories that lack shape.txt."""
        files_dir = tmp_path / "files"
        (files_dir / "asteroid_1" / "model_10").mkdir(parents=True)
        # No shape.txt inside model_10

        index = build_shape_index(files_dir)
        assert index == {}

    def test_empty_dir(self, tmp_path: Path) -> None:
        """Should return empty dict for empty files_dir."""
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        assert build_shape_index(files_dir) == {}

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        """Should return empty dict for nonexistent directory."""
        assert build_shape_index(tmp_path / "no_such_dir") == {}


class TestWriteShapeIndex:
    """Tests for write_shape_index."""

    def test_writes_json_file(self, tmp_path: Path) -> None:
        """Should write shape_index.json with expected entries."""
        # Reset lru_cache to avoid cross-test pollution
        _find_extracted_prefix.cache_clear()

        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp, shape_models={10: _SHAPE_TEXT, 20: _SHAPE_TEXT_SIMPLE})
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        # Remove the index written by extraction to test write_shape_index directly
        json_path = extract_dir / "shape_index.json"
        if json_path.exists():
            json_path.unlink()

        result = write_shape_index(extract_dir)
        assert result == json_path
        assert json_path.exists()

        import json

        data = json.loads(json_path.read_text())
        assert "10" in data
        assert "20" in data
        assert data["10"].endswith("shape.txt")


class TestShapePathIndex:
    """Tests for _ShapePathIndex."""

    def _setup_extracted(self, tmp_path: Path) -> Path:
        """Create a synthetic extracted directory with shape index."""
        _find_extracted_prefix.cache_clear()
        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp, shape_models={10: _SHAPE_TEXT, 20: _SHAPE_TEXT_SIMPLE})
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)
        return extract_dir

    def test_lookup_returns_correct_path(self, tmp_path: Path) -> None:
        """Should return the correct absolute path for a known model."""
        extract_dir = self._setup_extracted(tmp_path)
        idx = _ShapePathIndex()
        result = idx.lookup(extract_dir, 10)
        assert result is not None
        assert result.exists()
        assert result.name == "shape.txt"
        assert "model_10" in str(result)

    def test_lookup_missing_model_returns_none(self, tmp_path: Path) -> None:
        """Should return None for an unknown model_id."""
        extract_dir = self._setup_extracted(tmp_path)
        idx = _ShapePathIndex()
        assert idx.lookup(extract_dir, 99999) is None

    def test_fallback_builds_index(self, tmp_path: Path) -> None:
        """When shape_index.json is missing, should build and persist it."""
        extract_dir = self._setup_extracted(tmp_path)
        json_path = extract_dir / "shape_index.json"
        # Remove the index
        json_path.unlink()
        assert not json_path.exists()

        idx = _ShapePathIndex()
        result = idx.lookup(extract_dir, 10)
        assert result is not None
        assert result.exists()
        # Should have been persisted
        assert json_path.exists()

    def test_invalidate_forces_reload(self, tmp_path: Path) -> None:
        """After invalidate(), the next lookup should reload from disk."""
        extract_dir = self._setup_extracted(tmp_path)
        idx = _ShapePathIndex()

        # Initial load
        result1 = idx.lookup(extract_dir, 10)
        assert result1 is not None

        # Invalidate
        idx.invalidate()
        assert idx._index is None

        # Should reload successfully
        result2 = idx.lookup(extract_dir, 10)
        assert result2 is not None
        assert result1 == result2


class TestFindExtractedPrefixCache:
    """Tests for _find_extracted_prefix lru_cache behavior."""

    def test_cache_hits(self, tmp_path: Path) -> None:
        """Repeated calls with the same arg should hit the cache."""
        _find_extracted_prefix.cache_clear()

        fp = tmp_path / "test.tar.gz"
        _make_damit_tar(fp)
        extract_dir = tmp_path / "extracted"
        extract_damit_archive(fp, extract_dir)

        _find_extracted_prefix.cache_clear()

        # First call: cache miss
        result1 = _find_extracted_prefix(extract_dir)
        info1 = _find_extracted_prefix.cache_info()
        assert info1.misses >= 1

        # Second call: cache hit
        result2 = _find_extracted_prefix(extract_dir)
        info2 = _find_extracted_prefix.cache_info()
        assert info2.hits >= 1
        assert result1 == result2


# ---------------------------------------------------------------------------
# Spin computation tests
# ---------------------------------------------------------------------------


class TestDamitSpinToRotation:
    """Tests for damit_spin_to_rotation."""

    def test_output_shape(self) -> None:
        """Should return a 3x3 matrix."""
        spin = jnp.array([30.0, 60.0, 5.0, 2460000.5, 0.0, 0.0])
        R = damit_spin_to_rotation(spin, 2460001.5)
        assert R.shape == (3, 3)

    def test_orthogonal(self) -> None:
        """Rotation matrix should be orthogonal (R @ R.T = I)."""
        spin = jnp.array([30.0, 60.0, 5.0, 2460000.5, 0.0, 0.0])
        R = damit_spin_to_rotation(spin, 2460001.5)
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-5)

    def test_determinant_one(self) -> None:
        """Rotation matrix should have determinant +1."""
        spin = jnp.array([45.0, -30.0, 8.0, 2451545.0, 90.0, 0.0])
        R = damit_spin_to_rotation(spin, 2451546.0)
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-5)

    def test_identity_at_t0_with_zero_angles(self) -> None:
        """At t=t0 with lambda=0, beta=90, phi0=0, should approximate identity.

        When lambda=0, Rz(0)=I; when beta=90, Ry(0)=I; when phi0=0 and
        t=t0, Rz(0)=I.  So R = I @ I @ I = I.
        """
        spin = jnp.array([0.0, 90.0, 5.0, 2460000.5, 0.0, 0.0])
        R = damit_spin_to_rotation(spin, 2460000.5)
        assert jnp.allclose(R, jnp.eye(3), atol=1e-5)

    def test_period_return(self) -> None:
        """Rotation should repeat after one full period."""
        spin = jnp.array([30.0, 60.0, 6.0, 2460000.5, 0.0, 0.0])
        t0 = 2460000.5
        period_days = 6.0 / 24.0  # Convert hours to days

        R1 = damit_spin_to_rotation(spin, t0 + 0.1)
        R2 = damit_spin_to_rotation(spin, t0 + 0.1 + period_days)
        assert jnp.allclose(R1, R2, atol=1e-4)

    def test_yorp_effect(self) -> None:
        """Nonzero YORP should change the rotation relative to zero YORP."""
        spin_no_yorp = jnp.array([30.0, 60.0, 5.0, 2460000.5, 0.0, 0.0])
        spin_yorp = jnp.array([30.0, 60.0, 5.0, 2460000.5, 0.0, 10.0])
        t = 2460010.5  # 10 days later

        R_no = damit_spin_to_rotation(spin_no_yorp, t)
        R_yes = damit_spin_to_rotation(spin_yorp, t)
        # Should be different due to YORP contribution
        assert not jnp.allclose(R_no, R_yes, atol=1e-3)

    def test_jit_compatible(self) -> None:
        """Should be compatible with jax.jit."""
        spin = jnp.array([30.0, 60.0, 5.0, 2460000.5, 0.0, 0.0])

        @jax.jit
        def compute(s, t):
            return damit_spin_to_rotation(s, t)

        R = compute(spin, 2460001.5)
        assert R.shape == (3, 3)
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-5)

    def test_vmap_compatible(self) -> None:
        """Should be compatible with jax.vmap over time."""
        spin = jnp.array([30.0, 60.0, 5.0, 2460000.5, 0.0, 0.0])
        times = jnp.array([2460000.5, 2460001.5, 2460002.5])

        batched = jax.vmap(damit_spin_to_rotation, in_axes=(None, 0))
        Rs = batched(spin, times)
        assert Rs.shape == (3, 3, 3)


# ---------------------------------------------------------------------------
# Shape scaling and rotation tests
# ---------------------------------------------------------------------------


class TestScaleShapeVertices:
    """Tests for scale_shape_vertices."""

    def test_max_extent_correct(self) -> None:
        """After scaling, the max vertex distance should equal max_extent_m."""
        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5]])
        scaled = scale_shape_vertices(verts, 100.0)
        max_dist = jnp.max(jnp.linalg.norm(scaled, axis=1))
        assert jnp.allclose(max_dist, 100.0, atol=1e-3)

    def test_proportions_preserved(self) -> None:
        """Relative distances between vertices should be preserved."""
        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        scaled = scale_shape_vertices(verts, 50.0)
        # Ratio of distances should be preserved: 1:2
        d0 = jnp.linalg.norm(scaled[0])
        d1 = jnp.linalg.norm(scaled[1])
        assert jnp.allclose(d1 / d0, 2.0, atol=1e-4)

    def test_output_shape(self) -> None:
        """Output should have the same shape as input."""
        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        scaled = scale_shape_vertices(verts, 10.0)
        assert scaled.shape == verts.shape


class TestRotateShapePoints:
    """Tests for rotate_shape_points."""

    def test_output_shape(self) -> None:
        """Output should have same shape as input vertices."""
        spin = jnp.array([0.0, 90.0, 5.0, 2460000.5, 0.0, 0.0])
        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rotated = rotate_shape_points(spin, 2460000.5, verts)
        assert rotated.shape == (2, 3)

    def test_identity_rotation(self) -> None:
        """With identity-producing params, vertices should be unchanged."""
        spin = jnp.array([0.0, 90.0, 5.0, 2460000.5, 0.0, 0.0])
        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        rotated = rotate_shape_points(spin, 2460000.5, verts)
        assert jnp.allclose(rotated, verts, atol=1e-5)

    def test_consistency_with_manual_rotation(self) -> None:
        """Result should match manual R @ v computation."""
        spin = jnp.array([30.0, 60.0, 5.0, 2460000.5, 0.0, 0.0])
        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        t = 2460001.5

        rotated = rotate_shape_points(spin, t, verts)
        R = damit_spin_to_rotation(spin, t)
        expected = (R @ verts.T).T
        assert jnp.allclose(rotated, expected, atol=1e-5)

    def test_norm_preserved(self) -> None:
        """Rotation should preserve vertex distances from origin."""
        spin = jnp.array([45.0, -30.0, 8.0, 2451545.0, 90.0, 0.0])
        verts = jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        rotated = rotate_shape_points(spin, 2451546.0, verts)

        orig_norms = jnp.linalg.norm(verts, axis=1)
        new_norms = jnp.linalg.norm(rotated, axis=1)
        assert jnp.allclose(orig_norms, new_norms, atol=1e-4)


# ---------------------------------------------------------------------------
# Mesh export tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TRIMESH, reason="trimesh not installed")
class TestShapeToTrimesh:
    """Tests for shape_to_trimesh."""

    def test_returns_trimesh(self) -> None:
        """Should return a trimesh.Trimesh object."""
        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        facets = jnp.array([[0, 1, 2]])
        mesh = shape_to_trimesh(verts, facets)
        assert hasattr(mesh, "vertices")
        assert hasattr(mesh, "faces")
        assert mesh.vertices.shape == (3, 3)
        assert mesh.faces.shape == (1, 3)


@pytest.mark.skipif(not _HAS_TRIMESH, reason="trimesh not installed")
class TestExportShapeGlb:
    """Tests for export_shape_glb."""

    def test_creates_file(self, tmp_path: Path) -> None:
        """Should create a non-empty GLB file."""
        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        facets = jnp.array([[0, 1, 2]])
        fp = tmp_path / "test.glb"
        result = export_shape_glb(verts, facets, fp)
        assert result.exists()
        assert result.stat().st_size > 0


@pytest.mark.skipif(not _HAS_TRIMESH, reason="trimesh not installed")
class TestExportShapeStl:
    """Tests for export_shape_stl."""

    def test_creates_file(self, tmp_path: Path) -> None:
        """Should create a non-empty STL file."""
        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        facets = jnp.array([[0, 1, 2]])
        fp = tmp_path / "test.stl"
        result = export_shape_stl(verts, facets, fp)
        assert result.exists()
        assert result.stat().st_size > 0


class TestMeshExportWithoutTrimesh:
    """Tests for mesh export functions when trimesh is not available."""

    def test_shape_to_trimesh_import_error(self) -> None:
        """Should raise ImportError when trimesh is unavailable."""
        with patch("astrojax.datasets._damit_shapes._HAS_TRIMESH", False):
            with pytest.raises(ImportError, match="trimesh is required"):
                shape_to_trimesh(
                    jnp.array([[1.0, 0.0, 0.0]]),
                    jnp.array([[0, 0, 0]]),
                )

    def test_export_glb_import_error(self) -> None:
        """Should raise ImportError when trimesh is unavailable."""
        with patch("astrojax.datasets._damit_shapes._HAS_TRIMESH", False):
            with pytest.raises(ImportError, match="trimesh is required"):
                export_shape_glb(
                    jnp.array([[1.0, 0.0, 0.0]]),
                    jnp.array([[0, 0, 0]]),
                    "/tmp/test.glb",
                )

    def test_export_stl_import_error(self) -> None:
        """Should raise ImportError when trimesh is unavailable."""
        with patch("astrojax.datasets._damit_shapes._HAS_TRIMESH", False):
            with pytest.raises(ImportError, match="trimesh is required"):
                export_shape_stl(
                    jnp.array([[1.0, 0.0, 0.0]]),
                    jnp.array([[0, 0, 0]]),
                    "/tmp/test.stl",
                )


# ---------------------------------------------------------------------------
# Spherical UV tests
# ---------------------------------------------------------------------------


class TestComputeSphericalUvs:
    """Tests for compute_spherical_uvs."""

    def test_output_shape(self) -> None:
        """Should return (N, 2) float32 array."""
        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
        uvs = compute_spherical_uvs(verts)
        assert uvs.shape == (4, 2)
        assert uvs.dtype == np.float32

    def test_uv_range(self) -> None:
        """All UV values should be in [0, 1]."""
        verts = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
        uvs = compute_spherical_uvs(verts)
        assert np.all(uvs >= 0.0)
        assert np.all(uvs <= 1.0)

    def test_known_geometry(self) -> None:
        """Symmetric vertices (centroid=origin) should produce expected UVs.

        With centroid at origin the direction vector equals the vertex position:
        - (1, 0, 0): theta=atan2(0,1)=0 -> u=0.5, phi=acos(0)=pi/2 -> v=0.5
        - (-1, 0, 0): theta=atan2(0,-1)=pi -> u=1.0, phi=acos(0)=pi/2 -> v=0.5
        - (0, 0, 1): theta=atan2(0,0)=0 -> u=0.5, phi=acos(1)=0 -> v=0.0
        - (0, 0, -1): theta=atan2(0,0)=0 -> u=0.5, phi=acos(-1)=pi -> v=1.0
        """
        verts = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        uvs = compute_spherical_uvs(verts)
        # (1,0,0): u=0.5, v=0.5
        assert abs(uvs[0, 0] - 0.5) < 1e-5
        assert abs(uvs[0, 1] - 0.5) < 1e-5
        # (-1,0,0): u=1.0, v=0.5
        assert abs(uvs[1, 0] - 1.0) < 1e-5
        assert abs(uvs[1, 1] - 0.5) < 1e-5
        # (0,0,1): u=0.5, v=0.0
        assert abs(uvs[2, 0] - 0.5) < 1e-5
        assert abs(uvs[2, 1] - 0.0) < 1e-5
        # (0,0,-1): u=0.5, v=1.0
        assert abs(uvs[3, 0] - 0.5) < 1e-5
        assert abs(uvs[3, 1] - 1.0) < 1e-5

    def test_single_vertex(self) -> None:
        """A single vertex should still produce valid UVs."""
        verts = np.array([[5.0, 3.0, 2.0]])
        uvs = compute_spherical_uvs(verts)
        assert uvs.shape == (1, 2)
        # Centroid == vertex itself, direction is zero -> clipped, should still be in range
        assert np.all(uvs >= 0.0)
        assert np.all(uvs <= 1.0)


# ---------------------------------------------------------------------------
# Textured GLB export tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TRIMESH, reason="trimesh not installed")
class TestExportShapeGlbTextured:
    """Tests for export_shape_glb_textured."""

    def test_creates_file(self, tmp_path: Path) -> None:
        """Should create a non-empty GLB file with texture."""
        from PIL import Image

        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        facets = jnp.array([[0, 1, 2]])
        # Create a small synthetic texture image
        tex_img = Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8))
        fp = tmp_path / "test_textured.glb"
        result = export_shape_glb_textured(verts, facets, fp, tex_img)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_file_is_valid_glb(self, tmp_path: Path) -> None:
        """Exported file should be loadable by trimesh."""
        import trimesh
        from PIL import Image

        verts = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        facets = jnp.array([[0, 1, 2]])
        tex_img = Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8))
        fp = tmp_path / "valid.glb"
        export_shape_glb_textured(verts, facets, fp, tex_img)

        scene = trimesh.load(str(fp))
        # GLB loads as a Scene; verify it has geometry
        assert len(scene.geometry) > 0

    def test_import_error_without_trimesh(self) -> None:
        """Should raise ImportError when trimesh is unavailable."""
        from PIL import Image

        tex_img = Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8))
        with patch("astrojax.datasets._damit_shapes._HAS_TRIMESH", False):
            with pytest.raises(ImportError, match="trimesh is required"):
                export_shape_glb_textured(
                    jnp.array([[1.0, 0.0, 0.0]]),
                    jnp.array([[0, 0, 0]]),
                    "/tmp/test.glb",
                    tex_img,
                )
