"""Tests for Celestrak type enum bindings."""

import astrojax.celestrak as celestrak


class TestCelestrakQueryType:
    """Tests for CelestrakQueryType enum."""

    def test_gp_variant(self):
        qt = celestrak.CelestrakQueryType.GP
        assert str(qt) == "gp"

    def test_sup_gp_variant(self):
        qt = celestrak.CelestrakQueryType.SUP_GP
        assert str(qt) == "sup_gp"

    def test_satcat_variant(self):
        qt = celestrak.CelestrakQueryType.SATCAT
        assert str(qt) == "satcat"

    def test_equality(self):
        assert celestrak.CelestrakQueryType.GP == celestrak.CelestrakQueryType.GP
        assert celestrak.CelestrakQueryType.GP != celestrak.CelestrakQueryType.SATCAT

    def test_repr(self):
        qt = celestrak.CelestrakQueryType.GP
        assert "GP" in repr(qt)

    def test_as_str(self):
        assert celestrak.CelestrakQueryType.GP.as_str() == "gp"
        assert celestrak.CelestrakQueryType.SUP_GP.as_str() == "sup_gp"
        assert celestrak.CelestrakQueryType.SATCAT.as_str() == "satcat"

    def test_endpoint_path(self):
        assert celestrak.CelestrakQueryType.GP.endpoint_path() == "/NORAD/elements/gp.php"
        assert celestrak.CelestrakQueryType.SUP_GP.endpoint_path() == "/NORAD/elements/supplemental/sup-gp.php"
        assert celestrak.CelestrakQueryType.SATCAT.endpoint_path() == "/satcat/records.php"


class TestCelestrakOutputFormat:
    """Tests for CelestrakOutputFormat enum."""

    def test_all_variants(self):
        assert str(celestrak.CelestrakOutputFormat.TLE) == "TLE"
        assert str(celestrak.CelestrakOutputFormat.TWO_LE) == "2LE"
        assert str(celestrak.CelestrakOutputFormat.THREE_LE) == "3LE"
        assert str(celestrak.CelestrakOutputFormat.XML) == "XML"
        assert str(celestrak.CelestrakOutputFormat.KVN) == "KVN"
        assert str(celestrak.CelestrakOutputFormat.JSON) == "JSON"
        assert str(celestrak.CelestrakOutputFormat.JSON_PRETTY) == "JSON-PRETTY"
        assert str(celestrak.CelestrakOutputFormat.CSV) == "CSV"

    def test_equality(self):
        assert celestrak.CelestrakOutputFormat.JSON == celestrak.CelestrakOutputFormat.JSON
        assert celestrak.CelestrakOutputFormat.JSON != celestrak.CelestrakOutputFormat.CSV

    def test_repr(self):
        fmt = celestrak.CelestrakOutputFormat.JSON
        assert "Json" in repr(fmt)

    def test_is_json(self):
        assert celestrak.CelestrakOutputFormat.JSON.is_json() is True
        assert celestrak.CelestrakOutputFormat.JSON_PRETTY.is_json() is True
        assert celestrak.CelestrakOutputFormat.CSV.is_json() is False
        assert celestrak.CelestrakOutputFormat.TLE.is_json() is False


class TestSupGPSource:
    """Tests for SupGPSource enum."""

    def test_key_variants(self):
        assert str(celestrak.SupGPSource.SPACEX) == "spacex"
        assert str(celestrak.SupGPSource.SPACEX_SUP) == "spacex-sup"
        assert str(celestrak.SupGPSource.PLANET) == "planet"
        assert str(celestrak.SupGPSource.ONEWEB) == "oneweb"
        assert str(celestrak.SupGPSource.STARLINK) == "starlink"
        assert str(celestrak.SupGPSource.STARLINK_SUP) == "starlink-sup"
        assert str(celestrak.SupGPSource.IRIDIUM) == "iridium"
        assert str(celestrak.SupGPSource.IRIDIUM_NEXT) == "iridium-next"
        assert str(celestrak.SupGPSource.ORBCOMM) == "orbcomm"
        assert str(celestrak.SupGPSource.GLOBALSTAR) == "globalstar"

    def test_additional_variants(self):
        assert str(celestrak.SupGPSource.GEO) == "geo"
        assert str(celestrak.SupGPSource.GPS) == "gps"
        assert str(celestrak.SupGPSource.GLONASS) == "glonass"
        assert str(celestrak.SupGPSource.METEOSAT) == "meteosat"
        assert str(celestrak.SupGPSource.INTELSAT) == "intelsat"
        assert str(celestrak.SupGPSource.SES) == "ses"
        assert str(celestrak.SupGPSource.SWARM_TECHNOLOGIES) == "swarm"
        assert str(celestrak.SupGPSource.AMATEUR) == "amateur"
        assert str(celestrak.SupGPSource.CELESTRAK) == "celestrak"
        assert str(celestrak.SupGPSource.KUIPER) == "kuiper"

    def test_equality(self):
        assert celestrak.SupGPSource.SPACEX == celestrak.SupGPSource.SPACEX
        assert celestrak.SupGPSource.SPACEX != celestrak.SupGPSource.PLANET

    def test_repr(self):
        src = celestrak.SupGPSource.SPACEX
        assert "SpaceX" in repr(src)
