"""Tests for the astrojax.datasets MPC asteroid module."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import polars as pl
import pytest

from astrojax.constants import AU, GM_SUN
from astrojax.datasets._mpc_download import _FILENAME
from astrojax.datasets._mpc_parsers import (
    load_mpc_json_to_dataframe,
    packed_mpc_epoch_to_jd,
    unpack_mpc_epoch,
)
from astrojax.datasets._mpc_providers import load_mpc_asteroids, load_mpc_from_file
from astrojax.datasets._mpc_state import asteroid_state_ecliptic, get_asteroid_ephemeris
from astrojax.time import caldate_to_jd

# ---------------------------------------------------------------------------
# Packed epoch parsing
# ---------------------------------------------------------------------------


class TestUnpackMpcEpoch:
    """Tests for unpack_mpc_epoch."""

    def test_K24BN(self) -> None:
        """K24BN should decode to 2024-11-23."""
        assert unpack_mpc_epoch("K24BN") == (2024, 11, 23)

    def test_J9611(self) -> None:
        """J9611 should decode to 1996-01-01."""
        assert unpack_mpc_epoch("J9611") == (1996, 1, 1)

    def test_K25A1(self) -> None:
        """K25A1 should decode to 2025-10-01."""
        assert unpack_mpc_epoch("K25A1") == (2025, 10, 1)

    def test_K00C9(self) -> None:
        """K00C9 should decode to 2000-12-09."""
        assert unpack_mpc_epoch("K00C9") == (2000, 12, 9)

    def test_I9919(self) -> None:
        """I9919 should decode to 1899-01-09."""
        assert unpack_mpc_epoch("I9919") == (1899, 1, 9)

    def test_K24AV(self) -> None:
        """K24AV should decode to 2024-10-31."""
        assert unpack_mpc_epoch("K24AV") == (2024, 10, 31)

    def test_K241A(self) -> None:
        """K241A should decode to 2024-01-10."""
        assert unpack_mpc_epoch("K241A") == (2024, 1, 10)

    def test_invalid_length(self) -> None:
        """Should raise ValueError for non-5-character input."""
        with pytest.raises(ValueError, match="5 characters"):
            unpack_mpc_epoch("K24B")

    def test_invalid_century(self) -> None:
        """Should raise ValueError for unknown century character."""
        with pytest.raises(ValueError, match="Unknown century character"):
            unpack_mpc_epoch("Z2411")

    def test_invalid_month(self) -> None:
        """Should raise ValueError for invalid month character."""
        with pytest.raises(ValueError, match="Unknown month character"):
            unpack_mpc_epoch("K24Z1")

    def test_invalid_day(self) -> None:
        """Should raise ValueError for invalid day character."""
        with pytest.raises(ValueError, match="Unknown day character"):
            unpack_mpc_epoch("K241Z")


class TestPackedEpochToJd:
    """Tests for packed_mpc_epoch_to_jd."""

    def test_J9611_jd(self) -> None:
        """J9611 (1996-01-01) should give JD 2450083.5."""
        jd = packed_mpc_epoch_to_jd("J9611")
        assert abs(jd - 2450083.5) < 1e-6

    def test_K24BN_jd(self) -> None:
        """K24BN (2024-11-23) should match caldate_to_jd."""
        expected = float(caldate_to_jd(2024, 11, 23))
        jd = packed_mpc_epoch_to_jd("K24BN")
        assert abs(jd - expected) < 1e-6

    def test_K00C9_jd(self) -> None:
        """K00C9 (2000-12-09) should match caldate_to_jd."""
        expected = float(caldate_to_jd(2000, 12, 9))
        jd = packed_mpc_epoch_to_jd("K00C9")
        assert abs(jd - expected) < 1e-6


# ---------------------------------------------------------------------------
# DataFrame loading (using synthetic test data)
# ---------------------------------------------------------------------------


def _make_test_json_gz(filepath: Path, records: list[dict]) -> None:
    """Write a list of dicts as gzipped JSON to filepath."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(filepath, "wt", encoding="utf-8") as f:
        json.dump(records, f)


_CERES_RECORD = {
    "Number": "1",
    "Name": "Ceres",
    "Principal_desig": "A801 AA",
    "Epoch": "K25A1",
    "a": 2.7674796,
    "e": 0.0785095,
    "i": 10.58769,
    "Node": 80.26791,
    "Peri": 73.73161,
    "M": 130.21777,
    "n": 0.21411530,
    "H": 3.53,
}

_PALLAS_RECORD = {
    "Number": "2",
    "Name": "Pallas",
    "Principal_desig": "A802 FA",
    "Epoch": "K25A1",
    "a": 2.7720940,
    "e": 0.2312736,
    "i": 34.83293,
    "Node": 173.02525,
    "Peri": 310.04889,
    "M": 222.44757,
    "n": 0.21358220,
    "H": 4.22,
}


class TestLoadMpcJsonToDataframe:
    """Tests for load_mpc_json_to_dataframe."""

    def test_basic_load(self, tmp_path: Path) -> None:
        """Load two records and verify DataFrame structure."""
        fp = tmp_path / "test.json.gz"
        _make_test_json_gz(fp, [_CERES_RECORD, _PALLAS_RECORD])
        df = load_mpc_json_to_dataframe(fp)

        assert df.shape[0] == 2
        expected_cols = [
            "number",
            "name",
            "principal_desig",
            "epoch_packed",
            "epoch_jd",
            "a",
            "e",
            "i",
            "node",
            "peri",
            "M",
            "n",
            "H",
        ]
        assert df.columns == expected_cols

    def test_ceres_values(self, tmp_path: Path) -> None:
        """Verify Ceres record values are parsed correctly."""
        fp = tmp_path / "test.json.gz"
        _make_test_json_gz(fp, [_CERES_RECORD])
        df = load_mpc_json_to_dataframe(fp)

        row = df.row(0, named=True)
        assert row["number"].strip() == "1"
        assert row["name"].strip() == "Ceres"
        assert abs(row["a"] - 2.7674796) < 1e-7
        assert abs(row["e"] - 0.0785095) < 1e-7
        assert abs(row["i"] - 10.58769) < 1e-5

    def test_epoch_jd_computed(self, tmp_path: Path) -> None:
        """Verify epoch_jd is computed from packed epoch."""
        fp = tmp_path / "test.json.gz"
        _make_test_json_gz(fp, [_CERES_RECORD])
        df = load_mpc_json_to_dataframe(fp)

        row = df.row(0, named=True)
        expected_jd = float(caldate_to_jd(2025, 10, 1))
        assert row["epoch_jd"] is not None
        assert abs(row["epoch_jd"] - expected_jd) < 1e-6

    def test_missing_file(self, tmp_path: Path) -> None:
        """Raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_mpc_json_to_dataframe(tmp_path / "nonexistent.json.gz")

    def test_missing_fields(self, tmp_path: Path) -> None:
        """Records with missing fields should produce None/null values."""
        record = {"Number": "99999", "Epoch": "K25A1"}
        fp = tmp_path / "test.json.gz"
        _make_test_json_gz(fp, [record])
        df = load_mpc_json_to_dataframe(fp)

        row = df.row(0, named=True)
        assert row["name"] is None
        assert row["a"] is None


class TestLoadMpcFromFile:
    """Tests for load_mpc_from_file."""

    def test_delegates_to_parser(self, tmp_path: Path) -> None:
        """load_mpc_from_file should return the same DataFrame as the parser."""
        fp = tmp_path / "test.json.gz"
        _make_test_json_gz(fp, [_CERES_RECORD])
        df = load_mpc_from_file(fp)
        assert df.shape[0] == 1
        assert df.columns[0] == "number"


# ---------------------------------------------------------------------------
# Cache staleness / load_mpc_asteroids
# ---------------------------------------------------------------------------


class TestLoadMpcAsteroids:
    """Tests for load_mpc_asteroids caching behavior."""

    def test_uses_existing_fresh_file(self, tmp_path: Path) -> None:
        """Should load from file without downloading if fresh."""
        fp = tmp_path / "mpc" / _FILENAME
        _make_test_json_gz(fp, [_CERES_RECORD])

        df = load_mpc_asteroids(fp, max_age_days=999.0)
        assert df.shape[0] == 1

    def test_raises_when_no_cache_and_download_fails(self, tmp_path: Path) -> None:
        """Should raise RuntimeError when download fails and no cache exists."""
        fp = tmp_path / "mpc" / "nonexistent.json.gz"

        with patch(
            "astrojax.datasets._mpc_providers.download_mpc_file",
            side_effect=ConnectionError("mocked"),
        ):
            with pytest.raises(RuntimeError, match="Failed to download MPC data"):
                load_mpc_asteroids(fp, max_age_days=0.0)


# ---------------------------------------------------------------------------
# Asteroid lookup
# ---------------------------------------------------------------------------


class TestGetAsteroidEphemeris:
    """Tests for get_asteroid_ephemeris."""

    @pytest.fixture()
    def df(self, tmp_path: Path) -> pl.DataFrame:
        """Create a test DataFrame."""
        fp = tmp_path / "test.json.gz"
        _make_test_json_gz(fp, [_CERES_RECORD, _PALLAS_RECORD])
        return load_mpc_json_to_dataframe(fp)

    def test_lookup_by_number_int(self, df: pl.DataFrame) -> None:
        """Look up Ceres by integer number."""
        eph = get_asteroid_ephemeris(df, 1)
        assert eph["name"].strip() == "Ceres"
        assert abs(eph["a"] - 2.7674796) < 1e-7

    def test_lookup_by_number_str(self, df: pl.DataFrame) -> None:
        """Look up Pallas by string number."""
        eph = get_asteroid_ephemeris(df, "2")
        assert eph["name"].strip() == "Pallas"

    def test_lookup_by_name(self, df: pl.DataFrame) -> None:
        """Look up Ceres by name."""
        eph = get_asteroid_ephemeris(df, "Ceres")
        assert eph["number"].strip() == "1"

    def test_lookup_not_found(self, df: pl.DataFrame) -> None:
        """Raise KeyError for nonexistent asteroid."""
        with pytest.raises(KeyError, match="Asteroid not found"):
            get_asteroid_ephemeris(df, 99999)

    def test_returned_keys(self, df: pl.DataFrame) -> None:
        """Verify all expected keys are present in the result."""
        eph = get_asteroid_ephemeris(df, 1)
        expected_keys = {
            "name",
            "number",
            "principal_desig",
            "epoch_jd",
            "a",
            "e",
            "i",
            "node",
            "peri",
            "M",
            "n",
            "H",
        }
        assert set(eph.keys()) == expected_keys


# ---------------------------------------------------------------------------
# State computation
# ---------------------------------------------------------------------------


class TestAsteroidStateEcliptic:
    """Tests for asteroid_state_ecliptic."""

    def test_circular_orbit_radius(self) -> None:
        """A circular orbit should have r ~ a at all times."""
        a_au = 1.0  # 1 AU circular orbit
        oe = jnp.array([a_au, 0.0, 0.0, 0.0, 0.0, 0.0])
        epoch_jd = 2451545.0  # J2000
        target_jd = 2451545.0  # same epoch

        state = asteroid_state_ecliptic(epoch_jd, oe, target_jd)
        r = jnp.linalg.norm(state[:3])
        expected_r = a_au * AU

        assert abs(float(r) - expected_r) / expected_r < 1e-10

    def test_circular_orbit_velocity(self) -> None:
        """A circular orbit should have v ~ sqrt(GM_SUN / a)."""
        a_au = 1.0
        oe = jnp.array([a_au, 0.0, 0.0, 0.0, 0.0, 0.0])
        epoch_jd = 2451545.0
        target_jd = 2451545.0

        state = asteroid_state_ecliptic(epoch_jd, oe, target_jd)
        v = jnp.linalg.norm(state[3:6])
        a_m = a_au * AU
        expected_v = jnp.sqrt(GM_SUN / a_m)

        assert abs(float(v) - float(expected_v)) / float(expected_v) < 1e-10

    def test_use_au_flag(self) -> None:
        """use_au=True should return position in AU and velocity in AU/day."""
        a_au = 1.0
        oe = jnp.array([a_au, 0.0, 0.0, 0.0, 0.0, 0.0])
        epoch_jd = 2451545.0
        target_jd = 2451545.0

        state_si = asteroid_state_ecliptic(epoch_jd, oe, target_jd)
        state_au = asteroid_state_ecliptic(epoch_jd, oe, target_jd, use_au=True)

        # Position: AU vs metres
        r_au = jnp.linalg.norm(state_au[:3])
        r_si = jnp.linalg.norm(state_si[:3])
        assert abs(float(r_au) - float(r_si) / AU) < 1e-10

        # Velocity: AU/day vs m/s
        v_au = jnp.linalg.norm(state_au[3:6])
        v_si = jnp.linalg.norm(state_si[3:6])
        assert abs(float(v_au) - float(v_si) / AU * 86400.0) / float(v_au) < 1e-10

    def test_output_shape(self) -> None:
        """Output should be a 6-element array."""
        oe = jnp.array([2.0, 0.1, 10.0, 80.0, 73.0, 45.0])
        state = asteroid_state_ecliptic(2460000.5, oe, 2460100.5)
        assert state.shape == (6,)

    def test_propagation_changes_state(self) -> None:
        """State at a later time should differ from the epoch state."""
        oe = jnp.array([2.0, 0.1, 10.0, 80.0, 73.0, 45.0])
        epoch_jd = 2460000.5
        state0 = asteroid_state_ecliptic(epoch_jd, oe, epoch_jd)
        state1 = asteroid_state_ecliptic(epoch_jd, oe, epoch_jd + 100.0)
        assert not jnp.allclose(state0, state1)

    def test_eccentric_orbit_energy_conservation(self) -> None:
        """Specific orbital energy should be conserved at different times."""
        a_au = 2.5
        e = 0.3
        oe = jnp.array([a_au, e, 15.0, 45.0, 90.0, 30.0])
        epoch_jd = 2460000.5
        a_m = a_au * AU

        # Theoretical specific energy
        energy_expected = -GM_SUN / (2.0 * a_m)

        for dt_days in [0.0, 50.0, 200.0, 500.0]:
            state = asteroid_state_ecliptic(epoch_jd, oe, epoch_jd + dt_days)
            r = float(jnp.linalg.norm(state[:3]))
            v = float(jnp.linalg.norm(state[3:6]))
            energy = 0.5 * v**2 - GM_SUN / r
            assert abs(energy - float(energy_expected)) / abs(float(energy_expected)) < 1e-8

    def test_period_return(self) -> None:
        """After one orbital period, the state should return to the initial state."""
        a_au = 1.0
        oe = jnp.array([a_au, 0.1, 5.0, 30.0, 60.0, 0.0])
        epoch_jd = 2460000.5

        # Orbital period in days
        a_m = a_au * AU
        T_seconds = 2.0 * math.pi * math.sqrt(a_m**3 / GM_SUN)
        T_days = T_seconds / 86400.0

        state0 = asteroid_state_ecliptic(epoch_jd, oe, epoch_jd)
        state_T = asteroid_state_ecliptic(epoch_jd, oe, epoch_jd + T_days)

        assert jnp.allclose(state0, state_T, atol=1e-3)

    def test_zero_mean_anomaly_on_periapsis(self) -> None:
        """M=0 at epoch means the asteroid is at periapsis; r = a(1-e)."""
        a_au = 2.0
        e_val = 0.2
        oe = jnp.array([a_au, e_val, 0.0, 0.0, 0.0, 0.0])
        epoch_jd = 2460000.5

        state = asteroid_state_ecliptic(epoch_jd, oe, epoch_jd)
        r = float(jnp.linalg.norm(state[:3]))
        expected_r = a_au * AU * (1.0 - e_val)

        assert abs(r - expected_r) / expected_r < 1e-10

    def test_ceres_approximate_elements(self) -> None:
        """Ceres state computation with realistic elements should produce sensible results."""
        # Approximate Ceres elements
        oe = jnp.array(
            [
                _CERES_RECORD["a"],
                _CERES_RECORD["e"],
                _CERES_RECORD["i"],
                _CERES_RECORD["Node"],
                _CERES_RECORD["Peri"],
                _CERES_RECORD["M"],
            ]
        )
        epoch_jd = packed_mpc_epoch_to_jd("K25A1")

        state = asteroid_state_ecliptic(epoch_jd, oe, epoch_jd, use_au=True)
        r_au = float(jnp.linalg.norm(state[:3]))

        # Ceres should be roughly between 2.5 and 3.0 AU
        assert 2.0 < r_au < 3.5
