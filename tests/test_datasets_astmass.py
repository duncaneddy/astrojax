"""Tests for the astrojax.datasets asteroid masses module."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from astrojax.datasets._astmass_download import _FILENAME
from astrojax.datasets._astmass_parsers import (
    _FIELD_DEFS,
    _TAB_FILENAME,
    load_astmass_tab_to_dataframe,
)
from astrojax.datasets._astmass_providers import load_asteroid_masses, load_astmass_from_file

# ---------------------------------------------------------------------------
# Synthetic test data helpers
# ---------------------------------------------------------------------------

# Total line length for astmass12.tab records (at least 207 chars)
_LINE_LENGTH = 210


def _make_test_record(
    ast_number: str = "1",
    ast_name: str = "Ceres",
    prov_desig: str = "",
    satellite_name: str = "",
    mass_sol: str = "4.72e-10",
    mass_sol_unc: str = "0.03e-1",
    mass_kg: str = "9.39e+2",
    mass_kg_unc: str = "0.06e+2",
    bulk_density: str = "2.16",
    bulk_density_unc: str = "0.01",
    axis_a: str = "487.3",
    axis_b: str = "487.3",
    axis_c: str = "454.7",
    equiv_radius_unc: str = "1.8",
    mass_ref: str = "Konopliv+2011",
) -> str:
    """Build a fixed-width line matching the astmass12.tab format.

    Each field is placed at its correct byte offset from ``_FIELD_DEFS``.
    The returned string is exactly ``_LINE_LENGTH`` characters long.
    """
    line = [" "] * _LINE_LENGTH
    field_values = {
        "ast_number": ast_number,
        "ast_name": ast_name,
        "prov_desig": prov_desig,
        "satellite_name": satellite_name,
        "mass_sol": mass_sol,
        "mass_sol_unc": mass_sol_unc,
        "mass_kg": mass_kg,
        "mass_kg_unc": mass_kg_unc,
        "bulk_density": bulk_density,
        "bulk_density_unc": bulk_density_unc,
        "axis_a": axis_a,
        "axis_b": axis_b,
        "axis_c": axis_c,
        "equiv_radius_unc": equiv_radius_unc,
        "mass_ref": mass_ref,
    }
    for name, start, length, _ in _FIELD_DEFS:
        value = field_values[name]
        # Right-pad or truncate to fit
        padded = value.ljust(length)[:length]
        for i, ch in enumerate(padded):
            line[start + i] = ch
    return "".join(line)


_CERES_RECORD = _make_test_record(
    ast_number="1",
    ast_name="Ceres",
    mass_sol="4.72e-10",
    mass_sol_unc="0.03e-1",
    mass_kg="9.39e+2",
    mass_kg_unc="0.06e+2",
    bulk_density="2.16",
    bulk_density_unc="0.01",
    axis_a="487.3",
    axis_b="487.3",
    axis_c="454.7",
    equiv_radius_unc="1.8",
    mass_ref="Konopliv+2011",
)

_VESTA_RECORD = _make_test_record(
    ast_number="4",
    ast_name="Vesta",
    mass_sol="1.30e-10",
    mass_sol_unc="0.01e-1",
    mass_kg="2.59e+2",
    mass_kg_unc="0.02e+2",
    bulk_density="3.46",
    bulk_density_unc="0.03",
    axis_a="286.3",
    axis_b="278.6",
    axis_c="223.2",
    equiv_radius_unc="0.1",
    mass_ref="Russell+2012",
)


def _make_test_zip(filepath: Path, records: list[str]) -> None:
    """Create a ZIP containing ``astmass12.tab`` from record lines."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(records) + "\n"
    with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_TAB_FILENAME, content)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestLoadAstmassTabToDataframe:
    """Tests for load_astmass_tab_to_dataframe."""

    def test_basic_load(self, tmp_path: Path) -> None:
        """Load two records and verify DataFrame structure."""
        fp = tmp_path / "test.zip"
        _make_test_zip(fp, [_CERES_RECORD, _VESTA_RECORD])
        df = load_astmass_tab_to_dataframe(fp)

        assert df.shape[0] == 2
        expected_cols = [name for name, _, _, _ in _FIELD_DEFS]
        assert df.columns == expected_cols

    def test_column_types(self, tmp_path: Path) -> None:
        """Verify that each column has the expected Polars dtype."""
        fp = tmp_path / "test.zip"
        _make_test_zip(fp, [_CERES_RECORD])
        df = load_astmass_tab_to_dataframe(fp)

        for name, _, _, expected_dtype in _FIELD_DEFS:
            assert df[name].dtype == expected_dtype, (
                f"Column {name!r}: expected {expected_dtype}, got {df[name].dtype}"
            )

    def test_ceres_values(self, tmp_path: Path) -> None:
        """Verify Ceres record values are parsed correctly."""
        fp = tmp_path / "test.zip"
        _make_test_zip(fp, [_CERES_RECORD])
        df = load_astmass_tab_to_dataframe(fp)

        row = df.row(0, named=True)
        assert row["ast_number"] == 1
        assert row["ast_name"].strip() == "Ceres"
        assert abs(row["mass_sol"] - 4.72e-10) < 1e-12
        assert abs(row["bulk_density"] - 2.16) < 1e-4
        assert abs(row["axis_a"] - 487.3) < 0.1

    def test_vesta_values(self, tmp_path: Path) -> None:
        """Verify Vesta record values are parsed correctly."""
        fp = tmp_path / "test.zip"
        _make_test_zip(fp, [_VESTA_RECORD])
        df = load_astmass_tab_to_dataframe(fp)

        row = df.row(0, named=True)
        assert row["ast_number"] == 4
        assert row["ast_name"].strip() == "Vesta"
        assert abs(row["mass_sol"] - 1.30e-10) < 1e-12
        assert abs(row["bulk_density"] - 3.46) < 1e-4
        assert abs(row["axis_b"] - 278.6) < 0.1

    def test_missing_file(self, tmp_path: Path) -> None:
        """Raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_astmass_tab_to_dataframe(tmp_path / "nonexistent.zip")

    def test_blank_fields_as_null(self, tmp_path: Path) -> None:
        """Records with blank numeric fields should produce null values."""
        record = _make_test_record(
            ast_number="99",
            ast_name="TestAst",
            mass_sol="",
            mass_sol_unc="",
            mass_kg="",
            mass_kg_unc="",
            bulk_density="",
            bulk_density_unc="",
            axis_a="",
            axis_b="",
            axis_c="",
            equiv_radius_unc="",
            mass_ref="",
        )
        fp = tmp_path / "test.zip"
        _make_test_zip(fp, [record])
        df = load_astmass_tab_to_dataframe(fp)

        row = df.row(0, named=True)
        assert row["ast_number"] == 99
        assert row["mass_sol"] is None
        assert row["bulk_density"] is None
        assert row["axis_a"] is None
        assert row["mass_ref"] is None

    def test_invalid_zip_no_tab_file(self, tmp_path: Path) -> None:
        """A ZIP without astmass12.tab should raise ValueError."""
        fp = tmp_path / "bad.zip"
        fp.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(fp, "w") as zf:
            zf.writestr("other_file.txt", "nothing here")

        with pytest.raises(ValueError, match="does not contain"):
            load_astmass_tab_to_dataframe(fp)

    def test_subdirectory_tab_entry(self, tmp_path: Path) -> None:
        """A .tab file nested in a subdirectory inside the ZIP should be found."""
        fp = tmp_path / "nested.zip"
        fp.parent.mkdir(parents=True, exist_ok=True)
        content = _CERES_RECORD + "\n"
        with zipfile.ZipFile(fp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"data/subdir/{_TAB_FILENAME}", content)

        df = load_astmass_tab_to_dataframe(fp)
        assert df.shape[0] == 1
        assert df.row(0, named=True)["ast_number"] == 1


# ---------------------------------------------------------------------------
# Provider tests
# ---------------------------------------------------------------------------


class TestLoadAstmassFromFile:
    """Tests for load_astmass_from_file."""

    def test_delegates_to_parser(self, tmp_path: Path) -> None:
        """load_astmass_from_file should return the same DataFrame as the parser."""
        fp = tmp_path / "test.zip"
        _make_test_zip(fp, [_CERES_RECORD])
        df = load_astmass_from_file(fp)
        assert df.shape[0] == 1
        assert df.columns[0] == "ast_number"


class TestLoadAsteroidMasses:
    """Tests for load_asteroid_masses caching behavior."""

    def test_uses_existing_fresh_file(self, tmp_path: Path) -> None:
        """Should load from file without downloading if fresh."""
        fp = tmp_path / "astmass" / _FILENAME
        _make_test_zip(fp, [_CERES_RECORD, _VESTA_RECORD])

        df = load_asteroid_masses(fp, max_age_days=999.0)
        assert df.shape[0] == 2

    def test_raises_when_no_cache_and_download_fails(self, tmp_path: Path) -> None:
        """Should raise RuntimeError when download fails and no cache exists."""
        fp = tmp_path / "astmass" / "nonexistent.zip"

        with patch(
            "astrojax.datasets._astmass_providers.download_astmass_file",
            side_effect=ConnectionError("mocked"),
        ):
            with pytest.raises(RuntimeError, match="Failed to download asteroid masses"):
                load_asteroid_masses(fp, max_age_days=0.0)

    def test_falls_back_to_stale_cache(self, tmp_path: Path) -> None:
        """Should fall back to stale cache when download fails."""
        fp = tmp_path / "astmass" / _FILENAME
        _make_test_zip(fp, [_CERES_RECORD])

        with patch(
            "astrojax.datasets._astmass_providers.download_astmass_file",
            side_effect=ConnectionError("mocked"),
        ):
            # max_age_days=0 forces staleness check
            df = load_asteroid_masses(fp, max_age_days=0.0)
            assert df.shape[0] == 1
