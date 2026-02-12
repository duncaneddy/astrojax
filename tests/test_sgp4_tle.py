"""Tests for SGP4 TLE/OMM parsing and gravity constants."""

from math import pi

import pytest

from astrojax.sgp4 import (
    WGS72,
    WGS72OLD,
    WGS84,
    SGP4Elements,
    compute_checksum,
    parse_omm,
    parse_tle,
    validate_tle_line,
)

# ISS TLE from the sgp4 reference test suite
ISS_LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# Polar orbit test TLE (from brahe test fixtures)
POLAR_LINE1 = "1     1U          20  1.00000000  .00000000  00000-0  00000-0 0    07"
POLAR_LINE2 = "2     1  90.0000   0.0000 0010000   0.0000   0.0000 15.21936719    07"


class TestEarthGravityConstants:
    """Test gravity constant sets match reference sgp4 library."""

    def test_wgs72old_values(self) -> None:
        assert WGS72OLD.mu == 398600.79964
        assert WGS72OLD.radiusearthkm == 6378.135
        assert WGS72OLD.xke == pytest.approx(0.0743669161, rel=1e-8)
        assert WGS72OLD.j2 == 0.001082616
        assert WGS72OLD.j3 == -0.00000253881
        assert WGS72OLD.j4 == -0.00000165597
        assert WGS72OLD.j3oj2 == pytest.approx(WGS72OLD.j3 / WGS72OLD.j2, rel=1e-12)

    def test_wgs72_values(self) -> None:
        assert WGS72.mu == 398600.8
        assert WGS72.radiusearthkm == 6378.135
        assert WGS72.j2 == 0.001082616
        assert WGS72.j3 == -0.00000253881
        assert WGS72.j4 == -0.00000165597
        assert WGS72.j3oj2 == pytest.approx(WGS72.j3 / WGS72.j2, rel=1e-12)

    def test_wgs84_values(self) -> None:
        assert WGS84.mu == 398600.5
        assert WGS84.radiusearthkm == 6378.137
        assert WGS84.j2 == 0.00108262998905
        assert WGS84.j3 == -0.00000253215306
        assert WGS84.j4 == -0.00000161098761
        assert WGS84.j3oj2 == pytest.approx(WGS84.j3 / WGS84.j2, rel=1e-12)

    def test_wgs72_xke_derived(self) -> None:
        """xke should be 60/sqrt(re^3/mu)."""
        from math import sqrt

        expected = 60.0 / sqrt(WGS72.radiusearthkm**3 / WGS72.mu)
        assert WGS72.xke == pytest.approx(expected, rel=1e-10)

    def test_wgs84_xke_derived(self) -> None:
        from math import sqrt

        expected = 60.0 / sqrt(WGS84.radiusearthkm**3 / WGS84.mu)
        assert WGS84.xke == pytest.approx(expected, rel=1e-10)

    def test_tumin_is_inverse_xke(self) -> None:
        for grav in (WGS72OLD, WGS72, WGS84):
            assert grav.tumin == pytest.approx(1.0 / grav.xke, rel=1e-12)


class TestChecksum:
    """Test TLE checksum computation."""

    def test_iss_line1_checksum(self) -> None:
        assert compute_checksum(ISS_LINE1) == 7

    def test_iss_line2_checksum(self) -> None:
        assert compute_checksum(ISS_LINE2) == 7

    def test_polar_line1_checksum(self) -> None:
        assert compute_checksum(POLAR_LINE1) == 7

    def test_polar_line2_checksum(self) -> None:
        assert compute_checksum(POLAR_LINE2) == 7

    def test_digits_contribute_value(self) -> None:
        # Line with just "1" at position 0 and rest spaces
        line = "1" + " " * 67
        assert compute_checksum(line) == 1

    def test_minus_contributes_one(self) -> None:
        line = "-" + " " * 67
        assert compute_checksum(line) == 1


class TestValidateTleLine:
    """Test TLE line format validation."""

    def test_valid_iss_line1(self) -> None:
        validate_tle_line(ISS_LINE1, 1)

    def test_valid_iss_line2(self) -> None:
        validate_tle_line(ISS_LINE2, 2)

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            validate_tle_line("1 25544", 1)

    def test_wrong_line_number_raises(self) -> None:
        with pytest.raises(ValueError, match="does not start with"):
            validate_tle_line(ISS_LINE2, 1)

    def test_bad_checksum_raises(self) -> None:
        bad_line = ISS_LINE1[:68] + "0"  # wrong checksum
        with pytest.raises(ValueError, match="checksum mismatch"):
            validate_tle_line(bad_line, 1)


class TestParseTle:
    """Test TLE parsing against reference sgp4 library."""

    def test_iss_satellite_number(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        assert elem.satnum_str == "25544"

    def test_iss_classification(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        assert elem.classification == "U"

    def test_iss_international_designator(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        assert elem.intldesg == "98067A"

    def test_iss_epoch_year(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        assert elem.epochyr == 8  # 2008 -> 08

    def test_iss_epoch_days(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        assert elem.epochdays == pytest.approx(264.51782528, rel=1e-10)

    def test_iss_bstar(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        assert elem.bstar == pytest.approx(-0.11606e-4, rel=1e-6)

    def test_iss_inclination_radians(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        assert elem.inclo == pytest.approx(51.6416 * pi / 180.0, rel=1e-10)

    def test_iss_eccentricity(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        assert elem.ecco == pytest.approx(0.0006703, rel=1e-10)

    def test_iss_raan_radians(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        assert elem.nodeo == pytest.approx(247.4627 * pi / 180.0, rel=1e-10)

    def test_iss_argp_radians(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        assert elem.argpo == pytest.approx(130.5360 * pi / 180.0, rel=1e-10)

    def test_iss_mean_anomaly_radians(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        assert elem.mo == pytest.approx(325.0288 * pi / 180.0, rel=1e-10)

    def test_iss_mean_motion_rad_per_min(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        xpdotp = 1440.0 / (2.0 * pi)
        expected = 15.72125391 / xpdotp
        assert elem.no_kozai == pytest.approx(expected, rel=1e-8)

    def test_iss_matches_reference_sgp4(self) -> None:
        """Verify all parsed elements match the python-sgp4 library exactly."""
        from sgp4.api import WGS72 as SGP4_WGS72
        from sgp4.api import Satrec

        ref = Satrec.twoline2rv(ISS_LINE1, ISS_LINE2, SGP4_WGS72)
        elem = parse_tle(ISS_LINE1, ISS_LINE2)

        assert elem.satnum_str == ref.satnum_str
        assert elem.classification == ref.classification
        assert elem.intldesg == ref.intldesg
        assert elem.epochyr == ref.epochyr
        assert elem.epochdays == pytest.approx(ref.epochdays, rel=1e-12)
        assert elem.bstar == pytest.approx(ref.bstar, rel=1e-10)
        assert elem.ndot == pytest.approx(ref.ndot, rel=1e-10)
        assert elem.nddot == pytest.approx(ref.nddot, abs=1e-20)
        assert elem.inclo == pytest.approx(ref.inclo, rel=1e-10)
        assert elem.nodeo == pytest.approx(ref.nodeo, rel=1e-10)
        assert elem.ecco == pytest.approx(ref.ecco, rel=1e-10)
        assert elem.argpo == pytest.approx(ref.argpo, rel=1e-10)
        assert elem.mo == pytest.approx(ref.mo, rel=1e-10)
        assert elem.no_kozai == pytest.approx(ref.no_kozai, rel=1e-10)
        assert elem.jdsatepoch == pytest.approx(ref.jdsatepoch, rel=1e-10)
        assert elem.jdsatepochF == pytest.approx(ref.jdsatepochF, rel=1e-10)

    def test_polar_orbit(self) -> None:
        elem = parse_tle(POLAR_LINE1, POLAR_LINE2)
        assert elem.inclo == pytest.approx(90.0 * pi / 180.0, rel=1e-10)
        assert elem.ecco == pytest.approx(0.001, rel=1e-10)

    def test_mismatched_satnum_raises(self) -> None:
        # Build a line2 with different satnum but valid checksum
        bad_line2_base = "2 99999  51.6416 247.4627 0006703 130.5360 325.0288 15.7212539156353"
        cs = compute_checksum(bad_line2_base + "0")
        bad_line2 = bad_line2_base + str(cs)
        with pytest.raises(ValueError, match="do not match"):
            parse_tle(ISS_LINE1, bad_line2)

    def test_two_digit_year_2000s(self) -> None:
        """Years 00-56 map to 2000-2056."""
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        # ISS TLE has epoch year 08 -> 2008, stored as 08
        assert elem.epochyr == 8

    def test_two_digit_year_1900s(self) -> None:
        """Years 57-99 map to 1957-1999."""
        # Create a TLE with year 99 (-> 1999)
        # Modify ISS line1 year field to "99"
        modified_l1 = ISS_LINE1[:18] + "99" + ISS_LINE1[20:]
        # Recompute checksum
        checksum = compute_checksum(modified_l1)
        modified_l1 = modified_l1[:68] + str(checksum)
        elem = parse_tle(modified_l1, ISS_LINE2)
        assert elem.epochyr == 99


class TestParseOmm:
    """Test OMM field parsing."""

    def test_basic_omm_parsing(self) -> None:
        fields = {
            "EPOCH": "2008-09-20T12:25:40.104192",
            "MEAN_MOTION": "15.72125391",
            "ECCENTRICITY": "0.0006703",
            "INCLINATION": "51.6416",
            "RA_OF_ASC_NODE": "247.4627",
            "ARG_OF_PERICENTER": "130.5360",
            "MEAN_ANOMALY": "325.0288",
            "NORAD_CAT_ID": "25544",
            "BSTAR": "-0.11606E-4",
            "CLASSIFICATION_TYPE": "U",
            "OBJECT_ID": "1998-067A",
            "EPHEMERIS_TYPE": "0",
            "ELEMENT_SET_NO": "292",
            "REV_AT_EPOCH": "56353",
        }
        elem = parse_omm(fields)
        assert elem.satnum_str.strip() == "25544"
        assert elem.classification == "U"
        assert elem.intldesg == "98067A"
        assert elem.ecco == pytest.approx(0.0006703, rel=1e-10)
        assert elem.inclo == pytest.approx(51.6416 * pi / 180.0, rel=1e-10)
        assert elem.nodeo == pytest.approx(247.4627 * pi / 180.0, rel=1e-10)

    def test_omm_mean_motion_conversion(self) -> None:
        """Mean motion should convert from rev/day to rad/min."""
        fields = {
            "EPOCH": "2020-01-01T00:00:00.000000",
            "MEAN_MOTION": "15.0",
            "ECCENTRICITY": "0.001",
            "INCLINATION": "90.0",
            "RA_OF_ASC_NODE": "0.0",
            "ARG_OF_PERICENTER": "0.0",
            "MEAN_ANOMALY": "0.0",
            "NORAD_CAT_ID": "1",
            "BSTAR": "0.0",
        }
        elem = parse_omm(fields)
        expected = 15.0 / 720.0 * pi  # rev/day -> rad/min
        assert elem.no_kozai == pytest.approx(expected, rel=1e-10)

    def test_omm_optional_fields_default(self) -> None:
        fields = {
            "EPOCH": "2020-01-01T00:00:00.000000",
            "MEAN_MOTION": "15.0",
            "ECCENTRICITY": "0.001",
            "INCLINATION": "90.0",
            "RA_OF_ASC_NODE": "0.0",
            "ARG_OF_PERICENTER": "0.0",
            "MEAN_ANOMALY": "0.0",
            "NORAD_CAT_ID": "1",
            "BSTAR": "0.0",
        }
        elem = parse_omm(fields)
        assert elem.classification == "U"
        assert elem.ndot == pytest.approx(0.0, abs=1e-20)
        assert elem.nddot == pytest.approx(0.0, abs=1e-20)
        assert elem.ephtype == 0
        assert elem.revnum == 0

    def test_omm_epoch_without_microseconds(self) -> None:
        fields = {
            "EPOCH": "2020-01-01T00:00:00",
            "MEAN_MOTION": "15.0",
            "ECCENTRICITY": "0.001",
            "INCLINATION": "90.0",
            "RA_OF_ASC_NODE": "0.0",
            "ARG_OF_PERICENTER": "0.0",
            "MEAN_ANOMALY": "0.0",
            "NORAD_CAT_ID": "1",
            "BSTAR": "0.0",
        }
        elem = parse_omm(fields)
        assert isinstance(elem, SGP4Elements)


class TestSGP4ElementsType:
    """Test SGP4Elements dataclass behavior."""

    def test_frozen_dataclass(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        with pytest.raises(AttributeError):
            elem.ecco = 0.5

    def test_all_fields_present(self) -> None:
        elem = parse_tle(ISS_LINE1, ISS_LINE2)
        expected_fields = [
            "satnum_str",
            "classification",
            "intldesg",
            "epochyr",
            "epochdays",
            "ndot",
            "nddot",
            "bstar",
            "ephtype",
            "elnum",
            "revnum",
            "inclo",
            "nodeo",
            "ecco",
            "argpo",
            "mo",
            "no_kozai",
            "jdsatepoch",
            "jdsatepochF",
        ]
        for field in expected_fields:
            assert hasattr(elem, field), f"Missing field: {field}"
