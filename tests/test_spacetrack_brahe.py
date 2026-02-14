"""Comparison tests between astrojax.spacetrack and brahe.spacetrack.

Validates that astrojax spacetrack implementation produces identical
results to the brahe reference implementation for query building,
enum behavior, operator functions, and type construction.
"""

import pytest

brahe = pytest.importorskip("brahe")
brahe_spacetrack = pytest.importorskip("brahe.spacetrack")

from brahe.spacetrack import operators as bh_op  # noqa: E402

import astrojax.spacetrack as aj_spacetrack  # noqa: E402
from astrojax.spacetrack import operators as aj_op  # noqa: E402


class TestEnumParity:
    """Verify enum variants and string representations match brahe."""

    def test_request_controller_variants(self):
        for name in ["BASIC_SPACE_DATA", "EXPANDED_SPACE_DATA", "FILE_SHARE", "SP_EPHEMERIS", "PUBLIC_FILES"]:
            aj = getattr(aj_spacetrack.RequestController, name)
            bh = getattr(brahe.RequestController, name)
            assert str(aj) == str(bh), f"RequestController.{name} mismatch"

    def test_request_class_variants(self):
        for name in [
            "GP", "GP_HISTORY", "SATCAT", "SATCAT_CHANGE", "SATCAT_DEBUT",
            "DECAY", "TIP", "CDM_PUBLIC", "BOXSCORE", "ANNOUNCEMENT", "LAUNCH_SITE",
        ]:
            aj = getattr(aj_spacetrack.RequestClass, name)
            bh = getattr(brahe.RequestClass, name)
            assert str(aj) == str(bh), f"RequestClass.{name} mismatch"

    def test_sort_order_variants(self):
        for name in ["ASC", "DESC"]:
            aj = getattr(aj_spacetrack.SortOrder, name)
            bh = getattr(brahe.SortOrder, name)
            assert str(aj) == str(bh), f"SortOrder.{name} mismatch"

    def test_output_format_variants(self):
        for name in ["JSON", "XML", "HTML", "CSV", "TLE", "THREE_LE", "KVN"]:
            aj = getattr(aj_spacetrack.OutputFormat, name)
            bh = getattr(brahe.OutputFormat, name)
            assert str(aj) == str(bh), f"OutputFormat.{name} mismatch"

    def test_output_format_is_json(self):
        # brahe's Rust bindings may not expose is_json(); test our own behavior
        assert aj_spacetrack.OutputFormat.JSON.is_json() is True
        assert aj_spacetrack.OutputFormat.CSV.is_json() is False


class TestOperatorParity:
    """Verify operator functions produce identical output."""

    def test_greater_than(self):
        assert aj_op.greater_than("25544") == bh_op.greater_than("25544")

    def test_less_than(self):
        assert aj_op.less_than("0.01") == bh_op.less_than("0.01")

    def test_not_equal(self):
        assert aj_op.not_equal("DEBRIS") == bh_op.not_equal("DEBRIS")

    def test_inclusive_range(self):
        assert aj_op.inclusive_range("1", "100") == bh_op.inclusive_range("1", "100")

    def test_like(self):
        assert aj_op.like("ISS") == bh_op.like("ISS")

    def test_startswith(self):
        assert aj_op.startswith("2024") == bh_op.startswith("2024")

    def test_now(self):
        assert aj_op.now() == bh_op.now()

    def test_now_offset_negative(self):
        assert aj_op.now_offset(-7) == bh_op.now_offset(-7)

    def test_now_offset_positive(self):
        assert aj_op.now_offset(14) == bh_op.now_offset(14)

    def test_null_val(self):
        assert aj_op.null_val() == bh_op.null_val()

    def test_or_list(self):
        assert aj_op.or_list(["a", "b", "c"]) == bh_op.or_list(["a", "b", "c"])


class TestQueryBuilderParity:
    """Verify query builder URL construction matches brahe."""

    def test_basic_gp_query(self):
        aj = aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.GP).build()
        bh = brahe.SpaceTrackQuery(brahe.RequestClass.GP).build()
        assert aj == bh

    def test_filter(self):
        aj = aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.GP).filter("NORAD_CAT_ID", "25544").build()
        bh = brahe.SpaceTrackQuery(brahe.RequestClass.GP).filter("NORAD_CAT_ID", "25544").build()
        assert aj == bh

    def test_multiple_filters(self):
        aj = (
            aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.GP)
            .filter("NORAD_CAT_ID", "25544")
            .filter("EPOCH", ">2024-01-01")
            .build()
        )
        bh = (
            brahe.SpaceTrackQuery(brahe.RequestClass.GP)
            .filter("NORAD_CAT_ID", "25544")
            .filter("EPOCH", ">2024-01-01")
            .build()
        )
        assert aj == bh

    def test_order_by_desc(self):
        aj = aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.GP).order_by("EPOCH", aj_spacetrack.SortOrder.DESC).build()
        bh = brahe.SpaceTrackQuery(brahe.RequestClass.GP).order_by("EPOCH", brahe.SortOrder.DESC).build()
        assert aj == bh

    def test_limit(self):
        aj = aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.GP).limit(10).build()
        bh = brahe.SpaceTrackQuery(brahe.RequestClass.GP).limit(10).build()
        assert aj == bh

    def test_limit_offset(self):
        aj = aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.GP).limit_offset(10, 20).build()
        bh = brahe.SpaceTrackQuery(brahe.RequestClass.GP).limit_offset(10, 20).build()
        assert aj == bh

    def test_predicates(self):
        aj = aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.GP).predicates_filter(["NORAD_CAT_ID", "EPOCH"]).build()
        bh = brahe.SpaceTrackQuery(brahe.RequestClass.GP).predicates_filter(["NORAD_CAT_ID", "EPOCH"]).build()
        assert aj == bh

    def test_metadata(self):
        aj = aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.GP).metadata(True).build()
        bh = brahe.SpaceTrackQuery(brahe.RequestClass.GP).metadata(True).build()
        assert aj == bh

    def test_distinct(self):
        aj = aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.GP).distinct(True).build()
        bh = brahe.SpaceTrackQuery(brahe.RequestClass.GP).distinct(True).build()
        assert aj == bh

    def test_full_query(self):
        aj = (
            aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.GP)
            .filter("NORAD_CAT_ID", "25544")
            .order_by("EPOCH", aj_spacetrack.SortOrder.DESC)
            .limit(1)
            .build()
        )
        bh = (
            brahe.SpaceTrackQuery(brahe.RequestClass.GP)
            .filter("NORAD_CAT_ID", "25544")
            .order_by("EPOCH", brahe.SortOrder.DESC)
            .limit(1)
            .build()
        )
        assert aj == bh

    def test_query_with_operators(self):
        aj = (
            aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.GP)
            .filter("EPOCH", aj_op.greater_than(aj_op.now_offset(-7)))
            .filter("ECCENTRICITY", aj_op.less_than("0.01"))
            .build()
        )
        bh = (
            brahe.SpaceTrackQuery(brahe.RequestClass.GP)
            .filter("EPOCH", bh_op.greater_than(bh_op.now_offset(-7)))
            .filter("ECCENTRICITY", bh_op.less_than("0.01"))
            .build()
        )
        assert aj == bh

    def test_controller_override(self):
        aj = (
            aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.GP)
            .controller(aj_spacetrack.RequestController.EXPANDED_SPACE_DATA)
            .build()
        )
        bh = (
            brahe.SpaceTrackQuery(brahe.RequestClass.GP)
            .controller(brahe.RequestController.EXPANDED_SPACE_DATA)
            .build()
        )
        assert aj == bh


class TestCDMPublicParity:
    """Verify CDM_PUBLIC query construction matches brahe."""

    def test_cdm_public_default_controller(self):
        aj = aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.CDM_PUBLIC).build()
        bh = brahe.SpaceTrackQuery(brahe.RequestClass.CDM_PUBLIC).build()
        assert aj == bh

    def test_cdm_public_with_filters(self):
        aj = (
            aj_spacetrack.SpaceTrackQuery(aj_spacetrack.RequestClass.CDM_PUBLIC)
            .filter("CREATION_DATE", aj_op.greater_than(aj_op.now_offset(-7)))
            .order_by("TCA", aj_spacetrack.SortOrder.DESC)
            .build()
        )
        bh = (
            brahe.SpaceTrackQuery(brahe.RequestClass.CDM_PUBLIC)
            .filter("CREATION_DATE", bh_op.greater_than(bh_op.now_offset(-7)))
            .order_by("TCA", brahe.SortOrder.DESC)
            .build()
        )
        assert aj == bh


class TestRateLimitConfigParity:
    """Verify RateLimitConfig matches brahe."""

    def test_default_values(self):
        aj = aj_spacetrack.RateLimitConfig()
        bh = brahe.RateLimitConfig()
        assert aj.max_per_minute == bh.max_per_minute
        assert aj.max_per_hour == bh.max_per_hour

    def test_disabled_values(self):
        aj = aj_spacetrack.RateLimitConfig.disabled()
        bh = brahe.RateLimitConfig.disabled()
        assert aj.max_per_minute == bh.max_per_minute
        assert aj.max_per_hour == bh.max_per_hour


class TestSATCATRecordParity:
    """Verify SATCATRecord type exists in brahe."""

    def test_type_exists(self):
        """Both libraries should expose SATCATRecord."""
        assert hasattr(brahe_spacetrack, "SATCATRecord")
        assert hasattr(aj_spacetrack, "SATCATRecord")

    def test_astrojax_from_json_dict(self):
        """Verify our from_json_dict produces correct fields."""
        d = {
            "INTLDES": "98067A",
            "NORAD_CAT_ID": "25544",
            "SATNAME": "ISS (ZARYA)",
            "COUNTRY": "ISS",
        }
        rec = aj_spacetrack.SATCATRecord.from_json_dict(d)
        assert rec.intldes == "98067A"
        assert rec.norad_cat_id == 25544
        assert rec.satname == "ISS (ZARYA)"
        assert rec.country == "ISS"
