"""Tests for GPRecord dataclass."""

import pytest

from astrojax._gp_record import GPRecord


class TestGPRecordCreation:
    """Tests for GPRecord construction."""

    def test_default_construction(self):
        record = GPRecord()
        assert record.object_name is None
        assert record.norad_cat_id is None
        assert record.epoch is None
        assert record.mean_motion is None

    def test_keyword_construction(self):
        record = GPRecord(
            object_name="ISS (ZARYA)",
            norad_cat_id=25544,
            epoch="2024-01-15T12:00:00.000000",
            mean_motion=15.5,
            eccentricity=0.0001,
            inclination=51.6,
        )
        assert record.object_name == "ISS (ZARYA)"
        assert record.norad_cat_id == 25544
        assert record.epoch == "2024-01-15T12:00:00.000000"
        assert record.mean_motion == 15.5
        assert record.eccentricity == 0.0001
        assert record.inclination == 51.6


class TestGPRecordFromJsonDict:
    """Tests for GPRecord.from_json_dict()."""

    def test_numeric_json_values(self):
        """Celestrak returns numeric JSON values."""
        d = {
            "OBJECT_NAME": "ISS (ZARYA)",
            "NORAD_CAT_ID": 25544,
            "EPOCH": "2024-01-15T12:00:00.000000",
            "MEAN_MOTION": 15.49260335,
            "ECCENTRICITY": 0.0001234,
            "INCLINATION": 51.6429,
            "RA_OF_ASC_NODE": 120.5,
            "ARG_OF_PERICENTER": 45.3,
            "MEAN_ANOMALY": 300.1,
            "BSTAR": 0.00012,
        }
        record = GPRecord.from_json_dict(d)
        assert record.object_name == "ISS (ZARYA)"
        assert record.norad_cat_id == 25544
        assert record.mean_motion == pytest.approx(15.49260335)
        assert record.eccentricity == pytest.approx(0.0001234)
        assert record.inclination == pytest.approx(51.6429)
        assert record.bstar == pytest.approx(0.00012)

    def test_string_json_values(self):
        """SpaceTrack returns string JSON values."""
        d = {
            "OBJECT_NAME": "ISS (ZARYA)",
            "NORAD_CAT_ID": "25544",
            "EPOCH": "2024-01-15T12:00:00.000000",
            "MEAN_MOTION": "15.49260335",
            "ECCENTRICITY": "0.0001234",
            "INCLINATION": "51.6429",
            "RA_OF_ASC_NODE": "120.5",
            "ARG_OF_PERICENTER": "45.3",
            "MEAN_ANOMALY": "300.1",
            "BSTAR": "0.00012",
        }
        record = GPRecord.from_json_dict(d)
        assert record.object_name == "ISS (ZARYA)"
        assert record.norad_cat_id == 25544
        assert record.mean_motion == pytest.approx(15.49260335)
        assert record.eccentricity == pytest.approx(0.0001234)

    def test_missing_fields(self):
        """Missing fields should be None."""
        d = {"OBJECT_NAME": "ISS"}
        record = GPRecord.from_json_dict(d)
        assert record.object_name == "ISS"
        assert record.norad_cat_id is None
        assert record.epoch is None
        assert record.mean_motion is None

    def test_empty_dict(self):
        record = GPRecord.from_json_dict({})
        assert record.object_name is None
        assert record.norad_cat_id is None

    def test_invalid_numeric_values(self):
        """Invalid numeric values should produce None."""
        d = {
            "NORAD_CAT_ID": "not_a_number",
            "MEAN_MOTION": "invalid",
        }
        record = GPRecord.from_json_dict(d)
        assert record.norad_cat_id is None
        assert record.mean_motion is None

    def test_all_fields(self):
        """Test all fields can be populated."""
        d = {
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "NORAD_CAT_ID": 25544,
            "OBJECT_TYPE": "PAYLOAD",
            "CLASSIFICATION_TYPE": "U",
            "INTLDES": "98067A",
            "EPOCH": "2024-01-15T12:00:00.000000",
            "MEAN_MOTION": 15.49,
            "ECCENTRICITY": 0.0001,
            "INCLINATION": 51.6,
            "RA_OF_ASC_NODE": 120.0,
            "ARG_OF_PERICENTER": 45.0,
            "MEAN_ANOMALY": 300.0,
            "EPHEMERIS_TYPE": 0,
            "ELEMENT_SET_NO": 999,
            "REV_AT_EPOCH": 44000,
            "BSTAR": 0.00012,
            "MEAN_MOTION_DOT": 0.00001,
            "MEAN_MOTION_DDOT": 0.0,
            "SEMIMAJOR_AXIS": 6798.0,
            "PERIOD": 92.87,
            "APOAPSIS": 420.0,
            "PERIAPSIS": 418.0,
            "RCS_SIZE": "LARGE",
            "COUNTRY_CODE": "ISS",
            "LAUNCH_DATE": "1998-11-20",
            "SITE": "TYMSC",
            "DECAY_DATE": None,
            "FILE": 1,
            "GP_ID": 12345,
            "TLE_LINE0": "0 ISS (ZARYA)",
            "TLE_LINE1": "1 25544U ...",
            "TLE_LINE2": "2 25544 ...",
        }
        record = GPRecord.from_json_dict(d)
        assert record.object_name == "ISS (ZARYA)"
        assert record.object_id == "1998-067A"
        assert record.norad_cat_id == 25544
        assert record.object_type == "PAYLOAD"
        assert record.classification_type == "U"
        assert record.intldes == "98067A"
        assert record.semimajor_axis == pytest.approx(6798.0)
        assert record.rcs_size == "LARGE"
        assert record.country_code == "ISS"
        assert record.tle_line0 == "0 ISS (ZARYA)"


class TestGPRecordGetField:
    """Tests for GPRecord.get_field()."""

    def test_get_existing_field(self):
        record = GPRecord(object_name="ISS", norad_cat_id=25544, inclination=51.6)
        assert record.get_field("OBJECT_NAME") == "ISS"
        assert record.get_field("NORAD_CAT_ID") == "25544"
        assert record.get_field("INCLINATION") == "51.6"

    def test_get_none_field(self):
        record = GPRecord(object_name="ISS")
        assert record.get_field("NORAD_CAT_ID") is None
        assert record.get_field("INCLINATION") is None

    def test_get_nonexistent_field(self):
        record = GPRecord()
        assert record.get_field("NONEXISTENT") is None


class TestGPRecordToSgp4Elements:
    """Tests for GPRecord.to_sgp4_elements()."""

    def test_conversion(self):
        """Test conversion with ISS-like data."""
        record = GPRecord(
            object_name="ISS (ZARYA)",
            object_id="1998-067A",
            norad_cat_id=25544,
            classification_type="U",
            epoch="2024-01-15T12:00:00.000000",
            mean_motion=15.49260335,
            eccentricity=0.0001234,
            inclination=51.6429,
            ra_of_asc_node=120.5,
            arg_of_pericenter=45.3,
            mean_anomaly=300.1,
            bstar=0.00012,
            ephemeris_type=0,
            element_set_no=999,
            rev_at_epoch=44000,
            mean_motion_dot=0.00001,
            mean_motion_ddot=0.0,
        )
        elements = record.to_sgp4_elements()
        assert elements.satnum_str.strip() == "25544"
        assert elements.classification == "U"
        assert elements.ecco == pytest.approx(0.0001234)
        assert elements.bstar == pytest.approx(0.00012)

    def test_missing_required_field_raises(self):
        """Missing EPOCH should raise KeyError."""
        record = GPRecord(
            norad_cat_id=25544,
            mean_motion=15.49,
            eccentricity=0.0001,
            inclination=51.6,
        )
        with pytest.raises(KeyError):
            record.to_sgp4_elements()


class TestGPRecordStrRepr:
    """Tests for GPRecord __str__ and __repr__."""

    def test_str(self):
        record = GPRecord(object_name="ISS", norad_cat_id=25544)
        s = str(record)
        assert "ISS" in s
        assert "25544" in s

    def test_repr(self):
        record = GPRecord(object_name="ISS", norad_cat_id=25544)
        r = repr(record)
        assert "GPRecord" in r
        assert "ISS" in r
