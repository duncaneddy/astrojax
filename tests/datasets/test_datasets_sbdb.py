"""Tests for the astrojax.datasets SBDB asteroid module."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import polars as pl
import pytest

from astrojax.constants import AU, GM_SUN
from astrojax.datasets.mpc_state import asteroid_state_ecliptic
from astrojax.datasets.sbdb_download import _FILENAME
from astrojax.datasets.sbdb_parsers import load_sbdb_csv_to_dataframe
from astrojax.datasets.sbdb_providers import (
    get_sbdb_asteroid_ephemeris,
    load_sbdb_asteroids,
    load_sbdb_from_file,
)

# ---------------------------------------------------------------------------
# Synthetic test data helpers
# ---------------------------------------------------------------------------

_CSV_HEADER = (
    "full_name,pdes,name,diameter,GM,epoch,epoch_mjd,epoch_cal,"
    "e,a,q,i,om,w,ma,ad,n,tp,tp_cal,per,per_y"
)

_CERES_ROW = (
    "1 Ceres,1,Ceres,939.4,62.6284,2460600.5,60600,20241015,"
    "0.0785095,2.7674796,2.5501209,10.58769,80.26791,73.73161,130.21777,"
    "2.9848383,0.21411530,2459990.5,20230129,1681.63,4.605"
)

_PALLAS_ROW = (
    "2 Pallas,2,Pallas,513.0,,2460600.5,60600,20241015,"
    "0.2312736,2.7720940,2.1312268,34.83293,173.02525,310.04889,222.44757,"
    "3.4129612,0.21358220,2459561.5,20211127,1685.83,4.616"
)

_VESTA_ROW = (
    "4 Vesta,4,Vesta,525.4,17.2882,2460600.5,60600,20241015,"
    "0.0887426,2.3615126,2.1520165,7.14190,103.80908,149.55455,20.86384,"
    "2.5710087,0.27150190,2460523.5,20240730,1325.74,3.629"
)


def _make_test_csv(filepath: Path, rows: list[str]) -> None:
    """Write a synthetic SBDB CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    content = _CSV_HEADER + "\n" + "\n".join(rows) + "\n"
    filepath.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestLoadSbdbCsvToDataframe:
    """Tests for load_sbdb_csv_to_dataframe."""

    def test_basic_load(self, tmp_path: Path) -> None:
        """Load two records and verify DataFrame structure."""
        fp = tmp_path / "test.csv"
        _make_test_csv(fp, [_CERES_ROW, _PALLAS_ROW])
        df = load_sbdb_csv_to_dataframe(fp)

        assert df.shape[0] == 2
        assert "number" in df.columns
        assert "epoch_jd" in df.columns
        assert "node" in df.columns
        assert "peri" in df.columns
        assert "M" in df.columns

    def test_column_renames(self, tmp_path: Path) -> None:
        """Verify SBDB columns are renamed to MPC convention."""
        fp = tmp_path / "test.csv"
        _make_test_csv(fp, [_CERES_ROW])
        df = load_sbdb_csv_to_dataframe(fp)

        # Original SBDB names should not be present
        assert "pdes" not in df.columns
        assert "epoch" not in df.columns
        assert "om" not in df.columns
        assert "w" not in df.columns
        assert "ma" not in df.columns

        # MPC-compatible names should be present
        assert "number" in df.columns
        assert "epoch_jd" in df.columns
        assert "node" in df.columns
        assert "peri" in df.columns
        assert "M" in df.columns

    def test_value_parsing(self, tmp_path: Path) -> None:
        """Verify Ceres record values are parsed correctly."""
        fp = tmp_path / "test.csv"
        _make_test_csv(fp, [_CERES_ROW])
        df = load_sbdb_csv_to_dataframe(fp)

        row = df.row(0, named=True)
        assert row["full_name"] == "1 Ceres"
        assert row["name"] == "Ceres"
        assert abs(row["a"] - 2.7674796) < 1e-7
        assert abs(row["e"] - 0.0785095) < 1e-7
        assert abs(row["i"] - 10.58769) < 1e-5
        assert abs(row["node"] - 80.26791) < 1e-5
        assert abs(row["peri"] - 73.73161) < 1e-5
        assert abs(row["M"] - 130.21777) < 1e-5
        assert abs(row["diameter"] - 939.4) < 0.1
        assert abs(row["GM"] - 62.6284) < 0.01

    def test_sparse_nulls_gm(self, tmp_path: Path) -> None:
        """Pallas record should have null GM (missing in source)."""
        fp = tmp_path / "test.csv"
        _make_test_csv(fp, [_PALLAS_ROW])
        df = load_sbdb_csv_to_dataframe(fp)

        row = df.row(0, named=True)
        assert row["GM"] is None

    def test_number_is_utf8(self, tmp_path: Path) -> None:
        """Number column should be Utf8 for MPC parity."""
        fp = tmp_path / "test.csv"
        _make_test_csv(fp, [_CERES_ROW])
        df = load_sbdb_csv_to_dataframe(fp)

        assert df["number"].dtype == pl.Utf8

    def test_missing_file(self, tmp_path: Path) -> None:
        """Raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_sbdb_csv_to_dataframe(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# Provider tests
# ---------------------------------------------------------------------------


class TestLoadSbdbFromFile:
    """Tests for load_sbdb_from_file."""

    def test_delegates_to_parser(self, tmp_path: Path) -> None:
        """load_sbdb_from_file should return the same DataFrame as the parser."""
        fp = tmp_path / "test.csv"
        _make_test_csv(fp, [_CERES_ROW])
        df = load_sbdb_from_file(fp)
        assert df.shape[0] == 1
        assert "number" in df.columns


class TestLoadSbdbAsteroids:
    """Tests for load_sbdb_asteroids caching behavior."""

    def test_uses_existing_fresh_file(self, tmp_path: Path) -> None:
        """Should load from file without downloading if fresh."""
        fp = tmp_path / "sbdb" / _FILENAME
        _make_test_csv(fp, [_CERES_ROW, _PALLAS_ROW])

        df = load_sbdb_asteroids(fp, max_age_days=999.0)
        assert df.shape[0] == 2

    def test_raises_when_no_cache_and_download_fails(self, tmp_path: Path) -> None:
        """Should raise RuntimeError when download fails and no cache exists."""
        fp = tmp_path / "sbdb" / "nonexistent.csv"

        with patch(
            "astrojax.datasets.sbdb_providers.download_sbdb_file",
            side_effect=ConnectionError("mocked"),
        ):
            with pytest.raises(RuntimeError, match="Failed to download SBDB data"):
                load_sbdb_asteroids(fp, max_age_days=0.0)

    def test_falls_back_to_stale_cache(self, tmp_path: Path) -> None:
        """Should fall back to stale cache when download fails."""
        fp = tmp_path / "sbdb" / _FILENAME
        _make_test_csv(fp, [_CERES_ROW])

        with patch(
            "astrojax.datasets.sbdb_providers.download_sbdb_file",
            side_effect=ConnectionError("mocked"),
        ):
            df = load_sbdb_asteroids(fp, max_age_days=0.0)
            assert df.shape[0] == 1


# ---------------------------------------------------------------------------
# Asteroid lookup
# ---------------------------------------------------------------------------


class TestGetSbdbAsteroidEphemeris:
    """Tests for get_sbdb_asteroid_ephemeris."""

    @pytest.fixture()
    def df(self, tmp_path: Path) -> pl.DataFrame:
        """Create a test DataFrame."""
        fp = tmp_path / "test.csv"
        _make_test_csv(fp, [_CERES_ROW, _PALLAS_ROW, _VESTA_ROW])
        return load_sbdb_csv_to_dataframe(fp)

    def test_lookup_by_number_int(self, df: pl.DataFrame) -> None:
        """Look up Ceres by integer number."""
        eph = get_sbdb_asteroid_ephemeris(df, 1)
        assert eph["name"] == "Ceres"
        assert abs(eph["a"] - 2.7674796) < 1e-7

    def test_lookup_by_number_str(self, df: pl.DataFrame) -> None:
        """Look up Pallas by string number."""
        eph = get_sbdb_asteroid_ephemeris(df, "2")
        assert eph["name"] == "Pallas"

    def test_lookup_by_name(self, df: pl.DataFrame) -> None:
        """Look up Ceres by name."""
        eph = get_sbdb_asteroid_ephemeris(df, "Ceres")
        assert eph["number"] == "1"

    def test_lookup_not_found(self, df: pl.DataFrame) -> None:
        """Raise KeyError for nonexistent asteroid."""
        with pytest.raises(KeyError, match="Asteroid not found"):
            get_sbdb_asteroid_ephemeris(df, 99999)

    def test_returned_keys(self, df: pl.DataFrame) -> None:
        """Verify all expected keys are present in the result."""
        eph = get_sbdb_asteroid_ephemeris(df, 1)
        expected_keys = {
            "name",
            "full_name",
            "number",
            "epoch_jd",
            "a",
            "e",
            "i",
            "node",
            "peri",
            "M",
            "n",
            "diameter",
            "GM",
        }
        assert set(eph.keys()) == expected_keys

    def test_ephemeris_compatible_with_asteroid_state_ecliptic(self, df: pl.DataFrame) -> None:
        """SBDB ephemeris should be directly usable with asteroid_state_ecliptic."""
        eph = get_sbdb_asteroid_ephemeris(df, 1)
        oe = jnp.array([eph["a"], eph["e"], eph["i"], eph["node"], eph["peri"], eph["M"]])
        state = asteroid_state_ecliptic(eph["epoch_jd"], oe, eph["epoch_jd"])
        assert state.shape == (6,)
        # Position should be nonzero
        r = float(jnp.linalg.norm(state[:3]))
        assert r > 0.0


# ---------------------------------------------------------------------------
# State computation with SBDB elements
# ---------------------------------------------------------------------------


class TestSbdbStateComputation:
    """Tests for state computation using SBDB orbital elements."""

    @pytest.fixture()
    def ceres_eph(self, tmp_path: Path) -> dict:
        """Load Ceres ephemeris from synthetic SBDB data."""
        fp = tmp_path / "test.csv"
        _make_test_csv(fp, [_CERES_ROW])
        df = load_sbdb_csv_to_dataframe(fp)
        return get_sbdb_asteroid_ephemeris(df, 1)

    def test_ceres_distance_reasonable(self, ceres_eph: dict) -> None:
        """Ceres distance should be between 2.0 and 3.5 AU."""
        oe = jnp.array(
            [
                ceres_eph["a"],
                ceres_eph["e"],
                ceres_eph["i"],
                ceres_eph["node"],
                ceres_eph["peri"],
                ceres_eph["M"],
            ]
        )
        state = asteroid_state_ecliptic(
            ceres_eph["epoch_jd"], oe, ceres_eph["epoch_jd"], use_au=True
        )
        r_au = float(jnp.linalg.norm(state[:3]))
        assert 2.0 < r_au < 3.5

    def test_energy_conservation_with_sbdb_elements(self, ceres_eph: dict) -> None:
        """Specific orbital energy should be conserved at different times."""
        a_au = ceres_eph["a"]
        oe = jnp.array(
            [
                a_au,
                ceres_eph["e"],
                ceres_eph["i"],
                ceres_eph["node"],
                ceres_eph["peri"],
                ceres_eph["M"],
            ]
        )
        epoch_jd = ceres_eph["epoch_jd"]
        a_m = a_au * AU

        # Theoretical specific energy
        energy_expected = -GM_SUN / (2.0 * a_m)

        for dt_days in [0.0, 50.0, 200.0, 500.0]:
            state = asteroid_state_ecliptic(epoch_jd, oe, epoch_jd + dt_days)
            r = float(jnp.linalg.norm(state[:3]))
            v = float(jnp.linalg.norm(state[3:6]))
            energy = 0.5 * v**2 - GM_SUN / r
            assert abs(energy - float(energy_expected)) / abs(float(energy_expected)) < 1e-8

    def test_period_return_with_sbdb_elements(self, ceres_eph: dict) -> None:
        """After one orbital period, the state should return to the initial state."""
        oe = jnp.array(
            [
                ceres_eph["a"],
                ceres_eph["e"],
                ceres_eph["i"],
                ceres_eph["node"],
                ceres_eph["peri"],
                ceres_eph["M"],
            ]
        )
        epoch_jd = ceres_eph["epoch_jd"]

        # Orbital period in days
        a_m = ceres_eph["a"] * AU
        T_seconds = 2.0 * math.pi * math.sqrt(a_m**3 / GM_SUN)
        T_days = T_seconds / 86400.0

        state0 = asteroid_state_ecliptic(epoch_jd, oe, epoch_jd)
        state_T = asteroid_state_ecliptic(epoch_jd, oe, epoch_jd + T_days)

        assert jnp.allclose(state0, state_T, atol=1e-3)
