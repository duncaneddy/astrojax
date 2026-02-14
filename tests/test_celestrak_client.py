"""Tests for CelestrakClient construction and validation."""

import pytest

import astrojax.celestrak as celestrak


class TestCelestrakClientConstruction:
    """Tests for CelestrakClient construction."""

    def test_default_construction(self):
        client = celestrak.CelestrakClient()
        assert client is not None

    def test_with_cache_age(self):
        client = celestrak.CelestrakClient(cache_max_age=3600.0)
        assert client is not None

    def test_with_base_url(self):
        client = celestrak.CelestrakClient(base_url="https://test.celestrak.org")
        assert client is not None

    def test_with_base_url_and_cache_age(self):
        client = celestrak.CelestrakClient(
            base_url="https://test.celestrak.org", cache_max_age=1800.0
        )
        assert client is not None


class TestCelestrakQueryClassattrs:
    """Tests for CelestrakQuery class attribute constructors."""

    def test_gp_classattr(self):
        query = celestrak.CelestrakQuery.gp
        assert query is not None
        assert "CelestrakQuery" in repr(query)

    def test_sup_gp_classattr(self):
        query = celestrak.CelestrakQuery.sup_gp
        assert query is not None

    def test_satcat_classattr(self):
        query = celestrak.CelestrakQuery.satcat
        assert query is not None

    def test_gp_chaining(self):
        query = celestrak.CelestrakQuery.gp.group("stations")
        assert "GROUP=stations" in query.build_url()

    def test_satcat_chaining(self):
        query = celestrak.CelestrakQuery.satcat.active(True)
        assert "ACTIVE=Y" in query.build_url()


class TestGetGpValidation:
    """Tests for get_gp() argument validation."""

    def test_get_gp_no_args_raises(self):
        client = celestrak.CelestrakClient()
        with pytest.raises(ValueError, match="exactly one"):
            client.get_gp()

    def test_get_gp_multiple_args_raises(self):
        client = celestrak.CelestrakClient()
        with pytest.raises(ValueError, match="exactly one"):
            client.get_gp(catnr=25544, name="ISS")


class TestGetSatcatValidation:
    """Tests for get_satcat() argument validation."""

    def test_get_satcat_no_args_raises(self):
        client = celestrak.CelestrakClient()
        with pytest.raises(ValueError, match="at least one"):
            client.get_satcat()


class TestCelestrakSATCATRecord:
    """Tests for CelestrakSATCATRecord attributes."""

    def test_import(self):
        assert hasattr(celestrak, "CelestrakSATCATRecord")

    def test_from_json_dict(self):
        d = {
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "NORAD_CAT_ID": 25544,
            "OBJECT_TYPE": "PAY",
            "OPS_STATUS_CODE": "+",
            "OWNER": "ISS",
            "LAUNCH_DATE": "1998-11-20",
            "LAUNCH_SITE": "TYMSC",
            "DECAY_DATE": None,
            "PERIOD": "92.87",
            "INCLINATION": "51.64",
            "APOGEE": "420",
            "PERIGEE": "418",
            "RCS": "LARGE",
            "DATA_STATUS_CODE": "",
            "ORBIT_CENTER": "EA",
            "ORBIT_TYPE": "ORB",
        }
        record = celestrak.CelestrakSATCATRecord.from_json_dict(d)
        assert record.object_name == "ISS (ZARYA)"
        assert record.norad_cat_id == 25544
        assert record.inclination == "51.64"
        assert record.orbit_center == "EA"

    def test_get_field(self):
        record = celestrak.CelestrakSATCATRecord(
            object_name="ISS",
            norad_cat_id=25544,
        )
        assert record.get_field("OBJECT_NAME") == "ISS"
        assert record.get_field("NORAD_CAT_ID") == "25544"
        assert record.get_field("INCLINATION") is None

    def test_str_repr(self):
        record = celestrak.CelestrakSATCATRecord(
            object_name="ISS", norad_cat_id=25544
        )
        s = str(record)
        assert "ISS" in s
        assert "25544" in s
