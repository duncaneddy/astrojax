"""Cross-validation tests comparing astrojax EOP output against brahe 1.0+.

Both libraries load the same ``finals.all.iau2000.txt`` file. Brahe uses
float64 internally; astrojax stores data as float64 arrays. At integer
MJDs both should return identical values. At fractional MJDs, linear
interpolation should match within floating-point tolerance.
"""

from __future__ import annotations

import brahe as bh
import jax
import jax.numpy as jnp
import pytest

from astrojax.config import set_dtype
from astrojax.eop import (
    EOPData,
    get_dxdy,
    get_eop,
    get_lod,
    get_pm,
    get_ut1_utc,
    load_eop_from_file,
)

# Tolerances: both use float64, so agreement should be tight.
# Only source of discrepancy is floating-point ordering of operations.
_ABS_TOL_UT1 = 1e-9  # seconds
_ABS_TOL_PM = 1e-14  # radians
_ABS_TOL_LOD = 1e-12  # seconds
_ABS_TOL_DXDY = 1e-16  # radians

_DATA_FILE = "tests/data/finals.all.iau2000.txt"


@pytest.fixture(scope="module")
def eop_data() -> EOPData:
    """Load astrojax EOP from the test assets file with float64 precision."""
    set_dtype(jnp.float64)
    return load_eop_from_file(_DATA_FILE)


@pytest.fixture(scope="module", autouse=True)
def _init_brahe() -> None:
    """Initialize brahe with the same data file."""
    provider = bh.FileEOPProvider.from_standard_file(
        _DATA_FILE,
        interpolate=True,
        extrapolate="Hold",
    )
    bh.set_global_eop_provider_from_file_provider(provider)


# Test MJDs: historical dates where data is finalized
_INTEGER_MJDS = [41684.0, 50000.0, 55000.0, 59000.0, 59569.0, 60000.0]
_FRACTIONAL_MJDS = [50000.25, 55000.5, 59000.75, 59569.5]


class TestBraheParityUT1UTC:
    """UT1-UTC comparison with brahe."""

    @pytest.mark.parametrize("mjd", _INTEGER_MJDS)
    def test_ut1_utc_integer_mjd(self, eop_data: EOPData, mjd: float):
        astrojax_val = float(get_ut1_utc(eop_data, mjd))
        brahe_val = bh.get_global_ut1_utc(mjd)
        assert astrojax_val == pytest.approx(brahe_val, abs=_ABS_TOL_UT1)

    @pytest.mark.parametrize("mjd", _FRACTIONAL_MJDS)
    def test_ut1_utc_fractional_mjd(self, eop_data: EOPData, mjd: float):
        astrojax_val = float(get_ut1_utc(eop_data, mjd))
        brahe_val = bh.get_global_ut1_utc(mjd)
        assert astrojax_val == pytest.approx(brahe_val, abs=_ABS_TOL_UT1)


class TestBraheParityPM:
    """Polar motion comparison with brahe."""

    @pytest.mark.parametrize("mjd", _INTEGER_MJDS)
    def test_pm_integer_mjd(self, eop_data: EOPData, mjd: float):
        ax, ay = get_pm(eop_data, mjd)
        bx, by = bh.get_global_pm(mjd)
        assert float(ax) == pytest.approx(bx, abs=_ABS_TOL_PM)
        assert float(ay) == pytest.approx(by, abs=_ABS_TOL_PM)

    @pytest.mark.parametrize("mjd", _FRACTIONAL_MJDS)
    def test_pm_fractional_mjd(self, eop_data: EOPData, mjd: float):
        ax, ay = get_pm(eop_data, mjd)
        bx, by = bh.get_global_pm(mjd)
        assert float(ax) == pytest.approx(bx, abs=_ABS_TOL_PM)
        assert float(ay) == pytest.approx(by, abs=_ABS_TOL_PM)


class TestBraheParityLOD:
    """Length of day comparison with brahe."""

    @pytest.mark.parametrize("mjd", [50000.0, 55000.0, 59000.0, 59569.0, 60000.0])
    def test_lod_integer_mjd(self, eop_data: EOPData, mjd: float):
        astrojax_val = float(get_lod(eop_data, mjd))
        brahe_val = bh.get_global_lod(mjd)
        assert astrojax_val == pytest.approx(brahe_val, abs=_ABS_TOL_LOD)


class TestBraheParityDXDY:
    """Celestial pole offset comparison with brahe."""

    @pytest.mark.parametrize("mjd", _INTEGER_MJDS)
    def test_dxdy_integer_mjd(self, eop_data: EOPData, mjd: float):
        adx, ady = get_dxdy(eop_data, mjd)
        bdx, bdy = bh.get_global_dxdy(mjd)
        assert float(adx) == pytest.approx(bdx, abs=_ABS_TOL_DXDY)
        assert float(ady) == pytest.approx(bdy, abs=_ABS_TOL_DXDY)


class TestBraheParityGetEOP:
    """Full EOP query comparison with brahe."""

    @pytest.mark.parametrize("mjd", [59569.0, 59569.5])
    def test_get_eop_matches_brahe(self, eop_data: EOPData, mjd: float):
        """get_eop returns all 6 values matching brahe individual queries."""
        a_pm_x, a_pm_y, a_ut1, a_lod, a_dx, a_dy = get_eop(eop_data, mjd)
        b_ut1 = bh.get_global_ut1_utc(mjd)
        b_pm_x, b_pm_y = bh.get_global_pm(mjd)
        b_lod = bh.get_global_lod(mjd)
        b_dx, b_dy = bh.get_global_dxdy(mjd)

        assert float(a_ut1) == pytest.approx(b_ut1, abs=_ABS_TOL_UT1)
        assert float(a_pm_x) == pytest.approx(b_pm_x, abs=_ABS_TOL_PM)
        assert float(a_pm_y) == pytest.approx(b_pm_y, abs=_ABS_TOL_PM)
        assert float(a_lod) == pytest.approx(b_lod, abs=_ABS_TOL_LOD)
        assert float(a_dx) == pytest.approx(b_dx, abs=_ABS_TOL_DXDY)
        assert float(a_dy) == pytest.approx(b_dy, abs=_ABS_TOL_DXDY)


class TestJITWithFileData:
    """JIT/vmap tests with file-loaded data."""

    def test_jit_ut1_utc_with_file_data(self, eop_data: EOPData):
        """JIT query on file-loaded data matches eager."""
        mjd = 59569.0
        eager = float(get_ut1_utc(eop_data, mjd))
        jitted = float(jax.jit(get_ut1_utc)(eop_data, mjd))
        assert jitted == pytest.approx(eager, abs=1e-15)

    def test_vmap_ut1_utc_with_file_data(self, eop_data: EOPData):
        """vmap over batch of MJDs with file-loaded data."""
        mjds = jnp.array([59569.0, 59570.0, 59571.0])
        vmapped = jax.vmap(lambda m: get_ut1_utc(eop_data, m))
        results = vmapped(mjds)

        for i, mjd in enumerate([59569.0, 59570.0, 59571.0]):
            expected = float(get_ut1_utc(eop_data, mjd))
            assert float(results[i]) == pytest.approx(expected, abs=1e-15)

    def test_jit_get_eop_with_file_data(self, eop_data: EOPData):
        """JIT get_eop matches eager on file-loaded data."""
        mjd = 55000.0
        eager = get_eop(eop_data, mjd)
        jitted = jax.jit(get_eop)(eop_data, mjd)
        for e_val, j_val in zip(eager, jitted, strict=True):
            assert float(j_val) == pytest.approx(float(e_val), abs=1e-15)
