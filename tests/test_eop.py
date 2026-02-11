"""Tests for Earth Orientation Parameters (EOP) module."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from astrojax.config import get_dtype
from astrojax.constants import AS2RAD
from astrojax.eop import (
    EOPData,
    EOPExtrapolation,
    get_dxdy,
    get_eop,
    get_lod,
    get_pm,
    get_ut1_utc,
    load_default_eop,
    load_eop_from_file,
    static_eop,
    zero_eop,
)
from astrojax.eop._parsers import parse_standard_line


def _make_test_eop(
    mjds: list[float],
    ut1_utcs: list[float],
    pm_xs: list[float] | None = None,
    pm_ys: list[float] | None = None,
    dXs: list[float] | None = None,
    dYs: list[float] | None = None,
    lods: list[float] | None = None,
) -> EOPData:
    """Helper to construct EOPData for tests using the configured dtype."""
    dtype = get_dtype()
    n = len(mjds)
    mjd_arr = jnp.array(mjds, dtype=dtype)
    return EOPData(
        mjd=mjd_arr,
        pm_x=jnp.array(pm_xs or [0.0] * n, dtype=dtype),
        pm_y=jnp.array(pm_ys or [0.0] * n, dtype=dtype),
        ut1_utc=jnp.array(ut1_utcs, dtype=dtype),
        dX=jnp.array(dXs or [0.0] * n, dtype=dtype),
        dY=jnp.array(dYs or [0.0] * n, dtype=dtype),
        lod=jnp.array(lods or [0.0] * n, dtype=dtype),
        mjd_min=jnp.array(mjds[0], dtype=dtype),
        mjd_max=jnp.array(mjds[-1], dtype=dtype),
        mjd_last_lod=jnp.array(mjds[-1], dtype=dtype),
        mjd_last_dxdy=jnp.array(mjds[-1], dtype=dtype),
    )


# ---------------------------------------------------------------------------
# Static / Zero provider tests
# ---------------------------------------------------------------------------


class TestZeroEOP:
    """Tests for zero_eop provider."""

    def test_zero_eop_returns_eopdata(self):
        eop = zero_eop()
        assert isinstance(eop, EOPData)

    def test_zero_eop_ut1_utc(self):
        eop = zero_eop()
        val = get_ut1_utc(eop, 59569.0)
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_zero_eop_pm(self):
        eop = zero_eop()
        pm_x, pm_y = get_pm(eop, 59569.0)
        assert pm_x == pytest.approx(0.0, abs=1e-6)
        assert pm_y == pytest.approx(0.0, abs=1e-6)

    def test_zero_eop_dxdy(self):
        eop = zero_eop()
        dx, dy = get_dxdy(eop, 59569.0)
        assert dx == pytest.approx(0.0, abs=1e-6)
        assert dy == pytest.approx(0.0, abs=1e-6)

    def test_zero_eop_lod(self):
        eop = zero_eop()
        val = get_lod(eop, 59569.0)
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_zero_eop_get_eop(self):
        eop = zero_eop()
        pm_x, pm_y, ut1_utc, lod, dx, dy = get_eop(eop, 59569.0)
        assert pm_x == pytest.approx(0.0, abs=1e-6)
        assert pm_y == pytest.approx(0.0, abs=1e-6)
        assert ut1_utc == pytest.approx(0.0, abs=1e-6)
        assert lod == pytest.approx(0.0, abs=1e-6)
        assert dx == pytest.approx(0.0, abs=1e-6)
        assert dy == pytest.approx(0.0, abs=1e-6)


class TestStaticEOP:
    """Tests for static_eop provider."""

    def test_static_eop_constant_ut1_utc(self):
        eop = static_eop(ut1_utc=0.1234)
        val = get_ut1_utc(eop, 50000.0)
        assert val == pytest.approx(0.1234, abs=1e-6)

    def test_static_eop_constant_pm(self):
        eop = static_eop(pm_x=1e-6, pm_y=2e-6)
        pm_x, pm_y = get_pm(eop, 50000.0)
        assert pm_x == pytest.approx(1e-6, abs=1e-10)
        assert pm_y == pytest.approx(2e-6, abs=1e-10)

    def test_static_eop_constant_dxdy(self):
        eop = static_eop(dX=3e-9, dY=4e-9)
        dx, dy = get_dxdy(eop, 50000.0)
        assert dx == pytest.approx(3e-9, abs=1e-13)
        assert dy == pytest.approx(4e-9, abs=1e-13)

    def test_static_eop_constant_lod(self):
        eop = static_eop(lod=0.001)
        val = get_lod(eop, 50000.0)
        assert val == pytest.approx(0.001, abs=1e-6)

    def test_static_eop_any_mjd_in_range(self):
        """Static EOP should return same value for any MJD in range."""
        eop = static_eop(ut1_utc=0.5, mjd_min=50000.0, mjd_max=60000.0)
        for mjd in [50000.0, 55000.0, 59999.5, 60000.0]:
            val = get_ut1_utc(eop, mjd)
            assert val == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Interpolation tests
# ---------------------------------------------------------------------------


class TestInterpolation:
    """Tests for linear interpolation in EOP lookups."""

    @pytest.fixture()
    def three_point_eop(self) -> EOPData:
        """EOP with 3 data points for interpolation testing."""
        return _make_test_eop(
            mjds=[59000.0, 59001.0, 59002.0],
            ut1_utcs=[0.1, 0.2, 0.3],
            pm_xs=[1.0e-6, 2.0e-6, 3.0e-6],
            pm_ys=[0.5e-6, 1.0e-6, 1.5e-6],
            dXs=[1.0e-9, 2.0e-9, 3.0e-9],
            dYs=[4.0e-9, 5.0e-9, 6.0e-9],
            lods=[0.001, 0.002, 0.003],
        )

    def test_interpolate_at_data_point(self, three_point_eop: EOPData):
        """Query at exact data point should return exact value."""
        val = get_ut1_utc(three_point_eop, 59001.0)
        assert val == pytest.approx(0.2, abs=1e-6)

    def test_interpolate_midpoint(self, three_point_eop: EOPData):
        """Query at midpoint between data points."""
        val = get_ut1_utc(three_point_eop, 59000.5)
        assert val == pytest.approx(0.15, abs=1e-6)

    def test_interpolate_quarter(self, three_point_eop: EOPData):
        """Query at 1/4 point between data points."""
        val = get_ut1_utc(three_point_eop, 59000.25)
        assert val == pytest.approx(0.125, abs=1e-5)

    def test_interpolate_pm(self, three_point_eop: EOPData):
        """Interpolation works for polar motion."""
        pm_x, pm_y = get_pm(three_point_eop, 59000.5)
        assert pm_x == pytest.approx(1.5e-6, abs=1e-10)
        assert pm_y == pytest.approx(0.75e-6, abs=1e-10)

    def test_interpolate_dxdy(self, three_point_eop: EOPData):
        """Interpolation works for celestial pole offsets."""
        dx, dy = get_dxdy(three_point_eop, 59001.5)
        assert dx == pytest.approx(2.5e-9, abs=1e-13)
        assert dy == pytest.approx(5.5e-9, abs=1e-13)

    def test_interpolate_lod(self, three_point_eop: EOPData):
        """Interpolation works for LOD."""
        val = get_lod(three_point_eop, 59000.5)
        assert val == pytest.approx(0.0015, abs=1e-6)

    def test_interpolate_at_boundaries(self, three_point_eop: EOPData):
        """Query at first and last data points."""
        val_first = get_ut1_utc(three_point_eop, 59000.0)
        val_last = get_ut1_utc(three_point_eop, 59002.0)
        assert val_first == pytest.approx(0.1, abs=1e-6)
        assert val_last == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# Extrapolation tests
# ---------------------------------------------------------------------------


class TestExtrapolation:
    """Tests for extrapolation modes."""

    @pytest.fixture()
    def bounded_eop(self) -> EOPData:
        """EOP with a narrow MJD range for extrapolation testing."""
        return _make_test_eop(
            mjds=[59000.0, 59001.0, 59002.0],
            ut1_utcs=[0.1, 0.2, 0.3],
            pm_xs=[1e-6, 2e-6, 3e-6],
        )

    def test_hold_extrapolation_before(self, bounded_eop: EOPData):
        """HOLD mode clamps to first value before range."""
        val = get_ut1_utc(bounded_eop, 58999.0, EOPExtrapolation.HOLD)
        assert val == pytest.approx(0.1, abs=1e-6)

    def test_hold_extrapolation_after(self, bounded_eop: EOPData):
        """HOLD mode clamps to last value after range."""
        val = get_ut1_utc(bounded_eop, 59003.0, EOPExtrapolation.HOLD)
        assert val == pytest.approx(0.3, abs=1e-6)

    def test_zero_extrapolation_before(self, bounded_eop: EOPData):
        """ZERO mode returns 0 before range."""
        val = get_ut1_utc(bounded_eop, 58999.0, EOPExtrapolation.ZERO)
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_zero_extrapolation_after(self, bounded_eop: EOPData):
        """ZERO mode returns 0 after range."""
        val = get_ut1_utc(bounded_eop, 59003.0, EOPExtrapolation.ZERO)
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_zero_extrapolation_in_range(self, bounded_eop: EOPData):
        """ZERO mode returns interpolated value in range."""
        val = get_ut1_utc(bounded_eop, 59001.0, EOPExtrapolation.ZERO)
        assert val == pytest.approx(0.2, abs=1e-6)

    def test_default_is_hold(self, bounded_eop: EOPData):
        """Default extrapolation mode is HOLD."""
        val = get_ut1_utc(bounded_eop, 59003.0)
        assert val == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# JIT / vmap compatibility tests
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """Tests that EOP lookups work inside jax.jit and jax.vmap."""

    def test_jit_get_ut1_utc(self):
        """get_ut1_utc works inside jax.jit."""
        eop = static_eop(ut1_utc=0.5)
        jitted = jax.jit(get_ut1_utc, static_argnums=(2,))
        val = jitted(eop, 59569.0)
        assert val == pytest.approx(0.5, abs=1e-6)

    def test_jit_get_pm(self):
        """get_pm works inside jax.jit."""
        eop = static_eop(pm_x=1e-6, pm_y=2e-6)
        jitted = jax.jit(get_pm, static_argnums=(2,))
        pm_x, pm_y = jitted(eop, 59569.0)
        assert pm_x == pytest.approx(1e-6, abs=1e-10)
        assert pm_y == pytest.approx(2e-6, abs=1e-10)

    def test_jit_get_eop(self):
        """get_eop works inside jax.jit."""
        eop = static_eop(ut1_utc=0.1, pm_x=1e-6, lod=0.001)
        jitted = jax.jit(get_eop, static_argnums=(2,))
        pm_x, pm_y, ut1_utc, lod, dx, dy = jitted(eop, 59569.0)
        assert ut1_utc == pytest.approx(0.1, abs=1e-6)
        assert pm_x == pytest.approx(1e-6, abs=1e-10)
        assert lod == pytest.approx(0.001, abs=1e-6)

    def test_vmap_get_ut1_utc(self):
        """get_ut1_utc works with jax.vmap over MJD batch."""
        eop = _make_test_eop(
            mjds=[59000.0, 59001.0, 59002.0],
            ut1_utcs=[0.1, 0.2, 0.3],
        )
        dtype = get_dtype()
        mjd_batch = jnp.array([59000.0, 59000.5, 59001.0, 59001.5, 59002.0], dtype=dtype)
        vmapped = jax.vmap(lambda m: get_ut1_utc(eop, m))
        results = vmapped(mjd_batch)
        expected = jnp.array([0.1, 0.15, 0.2, 0.25, 0.3], dtype=dtype)
        assert jnp.allclose(results, expected, atol=1e-5)

    def test_jit_matches_eager(self):
        """JIT result matches eager result exactly."""
        eop = static_eop(ut1_utc=0.42, pm_x=1.5e-6)
        mjd = 55000.0
        eager_ut1 = get_ut1_utc(eop, mjd)
        eager_pm_x, _ = get_pm(eop, mjd)

        jit_ut1 = jax.jit(get_ut1_utc, static_argnums=(2,))(eop, mjd)
        jit_pm_x, _ = jax.jit(get_pm, static_argnums=(2,))(eop, mjd)

        assert float(jit_ut1) == pytest.approx(float(eager_ut1), abs=1e-10)
        assert float(jit_pm_x) == pytest.approx(float(eager_pm_x), abs=1e-10)


# ---------------------------------------------------------------------------
# EOPData pytree tests
# ---------------------------------------------------------------------------


class TestEOPDataPytree:
    """Tests that EOPData works as a JAX pytree."""

    def test_eopdata_is_namedtuple(self):
        eop = zero_eop()
        assert isinstance(eop, tuple)
        assert hasattr(eop, "_fields")

    def test_eopdata_tree_flatten_unflatten(self):
        """EOPData survives JAX pytree flatten/unflatten."""
        eop = static_eop(ut1_utc=0.5, pm_x=1e-6)
        leaves, treedef = jax.tree.flatten(eop)
        eop2 = jax.tree.unflatten(treedef, leaves)
        assert isinstance(eop2, EOPData)
        assert jnp.array_equal(eop.mjd, eop2.mjd)
        assert jnp.array_equal(eop.ut1_utc, eop2.ut1_utc)


# ---------------------------------------------------------------------------
# Parser tests (Stage 2)
# ---------------------------------------------------------------------------


class TestParseStandardLine:
    """Tests for IERS standard format line parser.

    Test cases mirror the Rust reference tests in
    refs/brahe_rust/src/eop/standard_parser.rs.
    """

    def test_parse_full_line(self):
        """Full line with all fields (Bulletin A + Bulletin B)."""
        line = "2311 1 60249.00 I  0.274620 0.000020  0.268283 0.000018  I 0.0113205 0.0000039 -0.3630 0.0029  I     0.293    0.290    -0.045    0.041  0.274569  0.268315  0.0113342     0.238    -0.039  "
        result = parse_standard_line(line)
        assert result is not None
        mjd, pm_x, pm_y, ut1_utc, lod, dX, dY = result
        assert mjd == pytest.approx(60249.0)
        assert pm_x == pytest.approx(0.274620 * AS2RAD, rel=1e-10)
        assert pm_y == pytest.approx(0.268283 * AS2RAD, rel=1e-10)
        assert ut1_utc == pytest.approx(0.0113205, rel=1e-10)
        assert lod == pytest.approx(-0.3630e-3, rel=1e-10)
        assert dX == pytest.approx(0.293e-3 * AS2RAD, rel=1e-10)
        assert dY == pytest.approx(-0.045e-3 * AS2RAD, rel=1e-10)

    def test_parse_no_bulletin_b(self):
        """Line with Bulletin A data but no Bulletin B."""
        line = "231220 60298.00 I  0.167496 0.000091  0.200643 0.000091  I 0.0109716 0.0000102  0.7706 0.0069  P     0.103    0.128    -0.193    0.160                                                     "
        result = parse_standard_line(line)
        assert result is not None
        mjd, pm_x, pm_y, ut1_utc, lod, dX, dY = result
        assert mjd == pytest.approx(60298.0)
        assert pm_x == pytest.approx(0.167496 * AS2RAD, rel=1e-10)
        assert pm_y == pytest.approx(0.200643 * AS2RAD, rel=1e-10)
        assert ut1_utc == pytest.approx(0.0109716, rel=1e-10)
        assert lod == pytest.approx(0.7706e-3, rel=1e-10)
        assert dX == pytest.approx(0.103e-3 * AS2RAD, rel=1e-10)
        assert dY == pytest.approx(-0.193e-3 * AS2RAD, rel=1e-10)

    def test_parse_prediction_no_lod(self):
        """Prediction line with no LOD."""
        line = "24 3 4 60373.00 P  0.026108 0.007892  0.289637 0.008989  P 0.0110535 0.0072179                 P     0.006    0.128    -0.118    0.160                                                     "
        result = parse_standard_line(line)
        assert result is not None
        mjd, pm_x, pm_y, ut1_utc, lod, dX, dY = result
        assert mjd == pytest.approx(60373.0)
        assert pm_x == pytest.approx(0.026108 * AS2RAD, rel=1e-10)
        assert ut1_utc == pytest.approx(0.0110535, rel=1e-10)
        assert math.isnan(lod)
        assert dX == pytest.approx(0.006e-3 * AS2RAD, rel=1e-10)
        assert dY == pytest.approx(-0.118e-3 * AS2RAD, rel=1e-10)

    def test_parse_prediction_no_lod_no_dxdy(self):
        """Prediction line with no LOD, dX, or dY."""
        line = "241228 60672.00 P  0.173369 0.019841  0.266914 0.028808  P 0.0420038 0.0254096                                                                                                             "
        result = parse_standard_line(line)
        assert result is not None
        mjd, pm_x, pm_y, ut1_utc, lod, dX, dY = result
        assert mjd == pytest.approx(60672.0)
        assert pm_x == pytest.approx(0.173369 * AS2RAD, rel=1e-10)
        assert pm_y == pytest.approx(0.266914 * AS2RAD, rel=1e-10)
        assert ut1_utc == pytest.approx(0.0420038, rel=1e-10)
        assert math.isnan(lod)
        assert math.isnan(dX)
        assert math.isnan(dY)

    def test_parse_only_mjd_returns_none(self):
        """Line with only MJD (no EOP values) returns None."""
        line = "241229 60673.00                                                                                                                                                                            "
        result = parse_standard_line(line)
        assert result is None

    def test_parse_too_long_returns_none(self):
        """Line longer than 187 chars returns None."""
        line = "2311 1 60249.00 I  0.274620 0.000020  0.268283 0.000018  I 0.0113205 0.0000039 -0.3630 0.0029  I     0.293    0.290    -0.045    0.041  0.274569  0.268315  0.0113342     0.238    -0.039  EXTRA"
        result = parse_standard_line(line)
        assert result is None

    def test_parse_empty_string_returns_none(self):
        """Empty string returns None."""
        result = parse_standard_line("")
        assert result is None

    def test_parse_short_line_pads_successfully(self):
        """Short prediction line (trailing whitespace trimmed) still parses."""
        full_line = "241228 60672.00 P  0.173369 0.019841  0.266914 0.028808  P 0.0420038 0.0254096                                                                                                             "
        line = full_line.rstrip()
        assert len(line) < 187
        result = parse_standard_line(line)
        assert result is not None
        assert result[0] == pytest.approx(60672.0)


# ---------------------------------------------------------------------------
# File loading tests (Stage 2)
# ---------------------------------------------------------------------------


class TestLoadEOPFromFile:
    """Tests for loading EOP data from files."""

    @pytest.fixture()
    def finals_path(self) -> str:
        """Path to the bundled finals.all data file."""
        import importlib.resources

        data_pkg = importlib.resources.files("astrojax.data.eop")
        resource = data_pkg.joinpath("finals.all.iau2000.txt")
        with importlib.resources.as_file(resource) as path:
            return str(path)

    def test_load_returns_eopdata(self, finals_path: str):
        eop = load_eop_from_file(finals_path)
        assert isinstance(eop, EOPData)

    def test_load_has_many_points(self, finals_path: str):
        eop = load_eop_from_file(finals_path)
        assert eop.mjd.shape[0] > 1000

    def test_load_mjd_sorted(self, finals_path: str):
        eop = load_eop_from_file(finals_path)
        diffs = jnp.diff(eop.mjd)
        assert jnp.all(diffs > 0)

    def test_load_mjd_min_max(self, finals_path: str):
        eop = load_eop_from_file(finals_path)
        assert float(eop.mjd_min) == float(eop.mjd[0])
        assert float(eop.mjd_max) == float(eop.mjd[-1])

    def test_load_known_value_ut1_utc(self, finals_path: str):
        """Spot-check a known UT1-UTC value from the first line."""
        eop = load_eop_from_file(finals_path)
        # First line: MJD 41684.00, UT1-UTC = 0.8084178
        val = get_ut1_utc(eop, 41684.0)
        assert val == pytest.approx(0.8084178, abs=1e-3)

    def test_load_known_value_pm(self, finals_path: str):
        """Spot-check known polar motion from first line."""
        eop = load_eop_from_file(finals_path)
        # First line: PM_X = 0.120733 arcsec, PM_Y = 0.136966 arcsec
        pm_x, pm_y = get_pm(eop, 41684.0)
        assert pm_x == pytest.approx(0.120733 * AS2RAD, rel=1e-3)
        assert pm_y == pytest.approx(0.136966 * AS2RAD, rel=1e-3)

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_eop_from_file("/nonexistent/path/to/file.txt")

    def test_load_mjd_last_lod_in_range(self, finals_path: str):
        """Last LOD MJD should be within the data range."""
        eop = load_eop_from_file(finals_path)
        assert float(eop.mjd_last_lod) >= float(eop.mjd_min)
        assert float(eop.mjd_last_lod) <= float(eop.mjd_max)

    def test_load_mjd_last_dxdy_in_range(self, finals_path: str):
        """Last dX/dY MJD should be within the data range."""
        eop = load_eop_from_file(finals_path)
        assert float(eop.mjd_last_dxdy) >= float(eop.mjd_min)
        assert float(eop.mjd_last_dxdy) <= float(eop.mjd_max)


class TestLoadDefaultEOP:
    """Tests for load_default_eop (bundled data)."""

    def test_load_default_returns_eopdata(self):
        eop = load_default_eop()
        assert isinstance(eop, EOPData)

    def test_load_default_has_data(self):
        eop = load_default_eop()
        assert eop.mjd.shape[0] > 1000

    def test_load_default_query_works(self):
        """Can query the default EOP data."""
        eop = load_default_eop()
        val = get_ut1_utc(eop, 59569.0)
        assert jnp.isfinite(val)
