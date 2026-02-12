"""Tests for the high-level TLE class."""

import jax
import jax.numpy as jnp
import pytest
from sgp4.api import WGS72 as SGP4_WGS72
from sgp4.api import Satrec

from astrojax.config import set_dtype
from astrojax.eop import zero_eop
from astrojax.epoch import Epoch
from astrojax.sgp4 import TLE

jax.config.update("jax_enable_x64", True)
set_dtype(jnp.float64)

# ISS TLE
ISS_LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# Molniya 2-14 (deep-space)
MOLNIYA_L1 = "1 08195U 75081A   06176.33215444  .00000099  00000-0  11873-3 0   813"
MOLNIYA_L2 = "2 08195  64.1586 279.0717 6877146 264.7651  20.2257  2.00491383225656"


def _get_reference(line1: str, line2: str, tsince_min: float) -> tuple:
    """Get reference position and velocity from python-sgp4."""
    sat = Satrec.twoline2rv(line1, line2, SGP4_WGS72)
    e, r, v = sat.sgp4_tsince(tsince_min)
    return e, r, v


class TestTLEInit:
    """Test TLE construction and properties."""

    def test_init_from_tle_lines(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        assert sat.satnum.strip() == "25544"

    def test_init_with_gravity_string(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2, gravity="wgs84")
        assert sat.satnum.strip() == "25544"

    def test_method_near_earth(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        assert sat.method == "n"

    def test_method_deep_space(self) -> None:
        sat = TLE(MOLNIYA_L1, MOLNIYA_L2)
        assert sat.method == "d"

    def test_invalid_tle_raises(self) -> None:
        with pytest.raises(ValueError):
            TLE("bad line 1", ISS_LINE2)


class TestTLEProperties:
    """Test user-friendly orbital element properties."""

    def test_mean_motion_rev_per_day(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        assert sat.n == pytest.approx(15.72125391, rel=1e-8)

    def test_eccentricity(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        assert sat.e == pytest.approx(0.0006703, rel=1e-10)

    def test_inclination_degrees(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        assert sat.i == pytest.approx(51.6416, rel=1e-6)

    def test_raan_degrees(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        assert sat.raan == pytest.approx(247.4627, rel=1e-6)

    def test_argp_degrees(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        assert sat.argp == pytest.approx(130.5360, rel=1e-6)

    def test_mean_anomaly_degrees(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        assert sat.M == pytest.approx(325.0288, rel=1e-6)

    def test_bstar(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        assert sat.bstar == pytest.approx(-0.11606e-4, rel=1e-6)

    def test_epoch_is_epoch_type(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        assert isinstance(sat.epoch, Epoch)


class TestTLEEpoch:
    """Test epoch construction from split JD."""

    def test_epoch_matches_reference(self) -> None:
        """TLE epoch should match python-sgp4 jdsatepoch + jdsatepochF."""
        ref = Satrec.twoline2rv(ISS_LINE1, ISS_LINE2, SGP4_WGS72)
        sat = TLE(ISS_LINE1, ISS_LINE2)

        # Reference JD
        ref_jd = ref.jdsatepoch + ref.jdsatepochF
        our_jd = float(sat.epoch.jd())
        assert our_jd == pytest.approx(ref_jd, abs=1e-6)

    def test_epoch_caldate(self) -> None:
        """ISS TLE epoch should be 2008-09-20."""
        sat = TLE(ISS_LINE1, ISS_LINE2)
        y, m, d, h, mi, s = sat.epoch.caldate()
        assert int(y) == 2008
        assert int(m) == 9
        assert int(d) == 20


class TestTLEPropagate:
    """Test raw propagate method (km, km/s)."""

    def test_at_epoch_matches_reference(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        r, v = sat.propagate(0.0)
        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, 0.0)
        assert e_ref == 0
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)
        assert jnp.allclose(v, jnp.array(v_ref), atol=1e-9)

    def test_at_60min_matches_reference(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        r, v = sat.propagate(60.0)
        e_ref, r_ref, v_ref = _get_reference(ISS_LINE1, ISS_LINE2, 60.0)
        assert e_ref == 0
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)

    def test_deep_space_matches_reference(self) -> None:
        sat = TLE(MOLNIYA_L1, MOLNIYA_L2)
        r, v = sat.propagate(360.0)
        e_ref, r_ref, v_ref = _get_reference(MOLNIYA_L1, MOLNIYA_L2, 360.0)
        assert e_ref == 0
        assert jnp.allclose(r, jnp.array(r_ref), atol=1e-6)


class TestTLEStateTEME:
    """Test state_teme method (SI units)."""

    def test_state_teme_matches_propagate_scaled(self) -> None:
        """state_teme should be propagate * 1e3."""
        sat = TLE(ISS_LINE1, ISS_LINE2)
        r, v = sat.propagate(60.0)
        x_teme = sat.state_teme(60.0 * 60.0)  # 60 min = 3600 sec

        assert jnp.allclose(x_teme[:3], r * 1e3, atol=1e-3)
        assert jnp.allclose(x_teme[3:6], v * 1e3, atol=1e-6)

    def test_state_alias(self) -> None:
        """state() should return same as state_teme()."""
        sat = TLE(ISS_LINE1, ISS_LINE2)
        x1 = sat.state(3600.0)
        x2 = sat.state_teme(3600.0)
        assert jnp.allclose(x1, x2, atol=1e-14)

    def test_epoch_time_argument(self) -> None:
        """Passing an Epoch should work and match float argument."""
        sat = TLE(ISS_LINE1, ISS_LINE2)
        # Get state at epoch + 60 min = epoch + 3600 sec
        x_float = sat.state_teme(3600.0)
        t_epoch = sat.epoch + 3600.0
        x_epoch = sat.state_teme(t_epoch)
        assert jnp.allclose(x_float, x_epoch, atol=1e-3)

    def test_output_shape(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        x = sat.state_teme(0.0)
        assert x.shape == (6,)

    def test_position_magnitude_leo(self) -> None:
        """LEO position should be ~6500-7000 km from Earth center."""
        sat = TLE(ISS_LINE1, ISS_LINE2)
        x = sat.state_teme(0.0)
        r_km = float(jnp.linalg.norm(x[:3])) / 1e3
        assert 6000.0 < r_km < 7200.0


class TestTLEFrameOutputs:
    """Test state methods in non-TEME frames."""

    def test_state_pef_shape(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        x = sat.state_pef(0.0)
        assert x.shape == (6,)

    def test_state_itrf_shape(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        x = sat.state_itrf(0.0)
        assert x.shape == (6,)

    def test_state_gcrf_shape(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        x = sat.state_gcrf(0.0)
        assert x.shape == (6,)

    def test_state_itrf_with_eop(self) -> None:
        """Explicit EOP data should be accepted."""
        sat = TLE(ISS_LINE1, ISS_LINE2)
        eop = zero_eop()
        x = sat.state_itrf(0.0, eop=eop)
        assert x.shape == (6,)
        assert jnp.all(jnp.isfinite(x))

    def test_state_gcrf_with_eop(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        eop = zero_eop()
        x = sat.state_gcrf(0.0, eop=eop)
        assert x.shape == (6,)
        assert jnp.all(jnp.isfinite(x))

    def test_default_eop_gives_finite_result(self) -> None:
        """None EOP (default) should produce finite results."""
        sat = TLE(ISS_LINE1, ISS_LINE2)
        x = sat.state_gcrf(0.0)
        assert jnp.all(jnp.isfinite(x))

    def test_position_magnitudes_match_across_frames(self) -> None:
        """Position magnitude should be preserved across frame transforms."""
        sat = TLE(ISS_LINE1, ISS_LINE2)
        eop = zero_eop()
        x_teme = sat.state_teme(3600.0)
        x_pef = sat.state_pef(3600.0)
        x_itrf = sat.state_itrf(3600.0, eop=eop)
        x_gcrf = sat.state_gcrf(3600.0, eop=eop)

        r_mag = float(jnp.linalg.norm(x_teme[:3]))
        assert float(jnp.linalg.norm(x_pef[:3])) == pytest.approx(r_mag, rel=1e-6)
        assert float(jnp.linalg.norm(x_itrf[:3])) == pytest.approx(r_mag, rel=1e-6)
        assert float(jnp.linalg.norm(x_gcrf[:3])) == pytest.approx(r_mag, rel=1e-6)


class TestTLERepr:
    """Test string representation."""

    def test_repr_contains_satnum(self) -> None:
        sat = TLE(ISS_LINE1, ISS_LINE2)
        assert "25544" in repr(sat)

    def test_repr_contains_method(self) -> None:
        sat = TLE(MOLNIYA_L1, MOLNIYA_L2)
        assert "'d'" in repr(sat)
