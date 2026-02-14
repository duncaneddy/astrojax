"""Comparison tests between astrojax.celestrak and brahe.celestrak.

Validates that astrojax celestrak implementation produces identical
results to the brahe reference implementation for query building,
enum behavior, and type construction.
"""

import pytest

brahe = pytest.importorskip("brahe")
brahe_celestrak = pytest.importorskip("brahe.celestrak")

import astrojax.celestrak as aj_celestrak  # noqa: E402


class TestEnumParity:
    """Verify enum variants and string representations match brahe."""

    def test_query_type_gp(self):
        assert str(aj_celestrak.CelestrakQueryType.GP) == str(brahe_celestrak.CelestrakQueryType.GP)

    def test_query_type_sup_gp(self):
        assert str(aj_celestrak.CelestrakQueryType.SUP_GP) == str(brahe_celestrak.CelestrakQueryType.SUP_GP)

    def test_query_type_satcat(self):
        assert str(aj_celestrak.CelestrakQueryType.SATCAT) == str(brahe_celestrak.CelestrakQueryType.SATCAT)

    def test_output_format_all_variants(self):
        for name in ["TLE", "TWO_LE", "THREE_LE", "XML", "KVN", "JSON", "JSON_PRETTY", "CSV"]:
            aj_fmt = getattr(aj_celestrak.CelestrakOutputFormat, name)
            bh_fmt = getattr(brahe_celestrak.CelestrakOutputFormat, name)
            assert str(aj_fmt) == str(bh_fmt), f"Mismatch for {name}"

    def test_sup_gp_source_all_variants(self):
        for name in [
            "SPACEX", "SPACEX_SUP", "PLANET", "ONEWEB", "STARLINK",
            "STARLINK_SUP", "GEO", "GPS", "GLONASS", "METEOSAT",
            "INTELSAT", "SES", "IRIDIUM", "IRIDIUM_NEXT", "ORBCOMM",
            "GLOBALSTAR", "SWARM_TECHNOLOGIES", "AMATEUR", "CELESTRAK", "KUIPER",
        ]:
            aj_src = getattr(aj_celestrak.SupGPSource, name)
            bh_src = getattr(brahe_celestrak.SupGPSource, name)
            assert str(aj_src) == str(bh_src), f"Mismatch for {name}"

    def test_output_format_is_json(self):
        # brahe's Rust bindings may not expose is_json(); test our own behavior
        assert aj_celestrak.CelestrakOutputFormat.JSON.is_json() is True
        assert aj_celestrak.CelestrakOutputFormat.CSV.is_json() is False


class TestQueryBuilderParity:
    """Verify query builder URL construction matches brahe."""

    def test_gp_by_group(self):
        aj = aj_celestrak.CelestrakQuery.gp.group("stations").build_url()
        bh = brahe_celestrak.CelestrakQuery.gp.group("stations").build_url()
        assert aj == bh

    def test_gp_by_catnr(self):
        aj = aj_celestrak.CelestrakQuery.gp.catnr(25544).build_url()
        bh = brahe_celestrak.CelestrakQuery.gp.catnr(25544).build_url()
        assert aj == bh

    def test_gp_by_intdes(self):
        aj = aj_celestrak.CelestrakQuery.gp.intdes("1998-067A").build_url()
        bh = brahe_celestrak.CelestrakQuery.gp.intdes("1998-067A").build_url()
        assert aj == bh

    def test_gp_by_name(self):
        aj = aj_celestrak.CelestrakQuery.gp.name_search("ISS").build_url()
        bh = brahe_celestrak.CelestrakQuery.gp.name_search("ISS").build_url()
        assert aj == bh

    def test_gp_with_format(self):
        aj = aj_celestrak.CelestrakQuery.gp.group("stations").format(
            aj_celestrak.CelestrakOutputFormat.THREE_LE
        ).build_url()
        bh = brahe_celestrak.CelestrakQuery.gp.group("stations").format(
            brahe_celestrak.CelestrakOutputFormat.THREE_LE
        ).build_url()
        assert aj == bh

    def test_sup_gp_by_source(self):
        aj = aj_celestrak.CelestrakQuery.sup_gp.source(aj_celestrak.SupGPSource.STARLINK).build_url()
        bh = brahe_celestrak.CelestrakQuery.sup_gp.source(brahe_celestrak.SupGPSource.STARLINK).build_url()
        assert aj == bh

    def test_satcat_active(self):
        aj = aj_celestrak.CelestrakQuery.satcat.active(True).build_url()
        bh = brahe_celestrak.CelestrakQuery.satcat.active(True).build_url()
        assert aj == bh

    def test_satcat_multiple_flags(self):
        aj = aj_celestrak.CelestrakQuery.satcat.active(True).payloads(True).on_orbit(True).build_url()
        bh = brahe_celestrak.CelestrakQuery.satcat.active(True).payloads(True).on_orbit(True).build_url()
        assert aj == bh

    def test_empty_query(self):
        aj = aj_celestrak.CelestrakQuery.gp.build_url()
        bh = brahe_celestrak.CelestrakQuery.gp.build_url()
        assert aj == bh

    def test_satcat_false_flags(self):
        aj = aj_celestrak.CelestrakQuery.satcat.active(False).build_url()
        bh = brahe_celestrak.CelestrakQuery.satcat.active(False).build_url()
        assert aj == bh


class TestSATCATRecordParity:
    """Verify CelestrakSATCATRecord type exists in brahe."""

    def test_type_exists(self):
        """Both libraries should expose CelestrakSATCATRecord."""
        assert hasattr(brahe_celestrak, "CelestrakSATCATRecord")
        assert hasattr(aj_celestrak, "CelestrakSATCATRecord")

    def test_astrojax_from_json_dict(self):
        """Verify our from_json_dict produces correct fields."""
        d = {
            "OBJECT_NAME": "ISS (ZARYA)",
            "NORAD_CAT_ID": 25544,
            "INCLINATION": "51.64",
        }
        rec = aj_celestrak.CelestrakSATCATRecord.from_json_dict(d)
        assert rec.object_name == "ISS (ZARYA)"
        assert rec.norad_cat_id == 25544
        assert rec.inclination == "51.64"
