"""Tests for SpaceTrackQuery builder."""

from astrojax.spacetrack import (
    OutputFormat,
    RequestClass,
    RequestController,
    SortOrder,
    SpaceTrackQuery,
)
from astrojax.spacetrack import (
    operators as op,
)


class TestSpaceTrackQuery:
    """Tests for SpaceTrackQuery builder."""

    def test_basic_query(self):
        query = SpaceTrackQuery(RequestClass.GP)
        url = query.build()
        assert "/basicspacedata/" in url
        assert "/class/gp/" in url
        assert "format/json" in url

    def test_filter(self):
        query = SpaceTrackQuery(RequestClass.GP).filter("NORAD_CAT_ID", "25544")
        url = query.build()
        assert "NORAD_CAT_ID/25544" in url

    def test_multiple_filters(self):
        query = (
            SpaceTrackQuery(RequestClass.GP)
            .filter("NORAD_CAT_ID", "25544")
            .filter("EPOCH", ">2024-01-01")
        )
        url = query.build()
        assert "NORAD_CAT_ID/25544" in url
        assert "EPOCH/%3E2024-01-01" in url

    def test_order_by(self):
        query = SpaceTrackQuery(RequestClass.GP).order_by("EPOCH", SortOrder.DESC)
        url = query.build()
        assert "orderby/EPOCH%20desc" in url

    def test_order_by_asc(self):
        query = SpaceTrackQuery(RequestClass.GP).order_by("EPOCH", SortOrder.ASC)
        url = query.build()
        assert "orderby/EPOCH%20asc" in url

    def test_limit(self):
        query = SpaceTrackQuery(RequestClass.GP).limit(10)
        url = query.build()
        assert "limit/10" in url

    def test_limit_offset(self):
        query = SpaceTrackQuery(RequestClass.GP).limit_offset(10, 20)
        url = query.build()
        assert "limit/10,20" in url

    def test_format_tle(self):
        query = SpaceTrackQuery(RequestClass.GP).format(OutputFormat.TLE)
        url = query.build()
        assert "format/tle" in url

    def test_format_csv(self):
        query = SpaceTrackQuery(RequestClass.GP).format(OutputFormat.CSV)
        url = query.build()
        assert "format/csv" in url

    def test_predicates_filter(self):
        query = SpaceTrackQuery(RequestClass.GP).predicates_filter(
            ["NORAD_CAT_ID", "OBJECT_NAME", "EPOCH"]
        )
        url = query.build()
        assert "predicates/NORAD_CAT_ID,OBJECT_NAME,EPOCH" in url

    def test_metadata(self):
        query = SpaceTrackQuery(RequestClass.GP).metadata(True)
        url = query.build()
        assert "metadata/true" in url

    def test_distinct(self):
        query = SpaceTrackQuery(RequestClass.GP).distinct(True)
        url = query.build()
        assert "distinct/true" in url

    def test_empty_result(self):
        query = SpaceTrackQuery(RequestClass.GP).empty_result(True)
        url = query.build()
        assert "emptyresult/show" in url

    def test_favorites(self):
        query = SpaceTrackQuery(RequestClass.GP).favorites("my_faves")
        url = query.build()
        assert "favorites/my_faves" in url

    def test_controller_override(self):
        query = SpaceTrackQuery(RequestClass.GP).controller(
            RequestController.EXPANDED_SPACE_DATA
        )
        url = query.build()
        assert "/expandedspacedata/" in url

    def test_satcat_query(self):
        query = SpaceTrackQuery(RequestClass.SATCAT)
        url = query.build()
        assert "/class/satcat/" in url

    def test_full_query_chain(self):
        """Test a complete query with multiple builder methods."""
        query = (
            SpaceTrackQuery(RequestClass.GP)
            .filter("NORAD_CAT_ID", "25544")
            .order_by("EPOCH", SortOrder.DESC)
            .limit(1)
        )
        url = query.build()
        assert "NORAD_CAT_ID/25544" in url
        assert "orderby/EPOCH%20desc" in url
        assert "limit/1" in url
        assert "format/json" in url

    def test_query_with_operators(self):
        """Test query using operator functions for filter values."""
        query = (
            SpaceTrackQuery(RequestClass.GP)
            .filter("EPOCH", op.greater_than(op.now_offset(-7)))
            .filter("ECCENTRICITY", op.less_than("0.01"))
            .filter("NORAD_CAT_ID", op.inclusive_range("25544", "25600"))
        )
        url = query.build()
        assert "EPOCH/%3Enow-7" in url
        assert "ECCENTRICITY/%3C0.01" in url
        assert "NORAD_CAT_ID/25544--25600" in url

    def test_query_str(self):
        query = SpaceTrackQuery(RequestClass.GP)
        assert str(query) == query.build()

    def test_query_repr(self):
        query = SpaceTrackQuery(RequestClass.GP)
        assert "SpaceTrackQuery" in repr(query)
        assert query.build() in repr(query)

    def test_query_with_all_formats(self):
        """Test all 7 output formats produce correct URL segments."""
        formats = [
            (OutputFormat.JSON, "format/json"),
            (OutputFormat.XML, "format/xml"),
            (OutputFormat.HTML, "format/html"),
            (OutputFormat.CSV, "format/csv"),
            (OutputFormat.TLE, "format/tle"),
            (OutputFormat.THREE_LE, "format/3le"),
            (OutputFormat.KVN, "format/kvn"),
        ]
        for fmt, expected in formats:
            query = SpaceTrackQuery(RequestClass.GP).format(fmt)
            url = query.build()
            assert expected in url, f"Expected '{expected}' in URL, got: {url}"

    def test_query_with_all_controllers(self):
        """Test all 5 controllers produce correct URL segments."""
        controllers = [
            (RequestController.BASIC_SPACE_DATA, "/basicspacedata/"),
            (RequestController.EXPANDED_SPACE_DATA, "/expandedspacedata/"),
            (RequestController.FILE_SHARE, "/fileshare/"),
            (RequestController.SP_EPHEMERIS, "/spephemeris/"),
            (RequestController.PUBLIC_FILES, "/publicfiles/"),
        ]
        for ctrl, expected in controllers:
            query = SpaceTrackQuery(RequestClass.GP).controller(ctrl)
            url = query.build()
            assert expected in url, f"Expected '{expected}' in URL, got: {url}"

    def test_query_immutability(self):
        """Test that builder methods return new instances."""
        base = SpaceTrackQuery(RequestClass.GP)
        with_filter = base.filter("NORAD_CAT_ID", "25544")
        with_limit = base.limit(10)

        base_url = base.build()
        filter_url = with_filter.build()
        limit_url = with_limit.build()

        assert "NORAD_CAT_ID" not in base_url
        assert "limit" not in base_url
        assert "NORAD_CAT_ID/25544" in filter_url
        assert "limit/10" in limit_url


class TestCDMPublicQuery:
    """Tests for CDM_PUBLIC query construction."""

    def test_cdm_public_default_controller(self):
        """CDM_PUBLIC uses basicspacedata controller."""
        query = SpaceTrackQuery(RequestClass.CDM_PUBLIC)
        url = query.build()
        assert url.startswith("/basicspacedata/")
        assert "/class/cdm_public/" in url

    def test_cdm_public_with_filters(self):
        """CDM_PUBLIC query with typical CDM filters."""
        query = (
            SpaceTrackQuery(RequestClass.CDM_PUBLIC)
            .filter("CREATION_DATE", op.greater_than(op.now_offset(-7)))
            .order_by("TCA", SortOrder.DESC)
        )
        url = query.build()
        assert "/basicspacedata/" in url
        assert "/class/cdm_public/" in url
        assert "CREATION_DATE/%3Enow-7" in url
        assert "orderby/TCA%20desc" in url
        assert "format/json" in url

    def test_cdm_public_full_query(self):
        """CDM_PUBLIC query with multiple filters and limit."""
        query = (
            SpaceTrackQuery(RequestClass.CDM_PUBLIC)
            .filter("PC", op.greater_than("1.0e-3"))
            .order_by("TCA", SortOrder.DESC)
            .limit(25)
        )
        url = query.build()
        expected = "/basicspacedata/query/class/cdm_public/PC/%3E1.0e-3/orderby/TCA%20desc/limit/25/format/json"
        assert url == expected

    def test_cdm_public_sat_filter(self):
        """CDM_PUBLIC query filtering by satellite ID."""
        query = (
            SpaceTrackQuery(RequestClass.CDM_PUBLIC)
            .filter("SAT_1_ID", "25544")
            .order_by("TCA", SortOrder.DESC)
            .limit(10)
        )
        url = query.build()
        expected = "/basicspacedata/query/class/cdm_public/SAT_1_ID/25544/orderby/TCA%20desc/limit/10/format/json"
        assert url == expected


class TestSpaceTrackQueryAccessors:
    """Tests for query accessor methods."""

    def test_output_format_default(self):
        query = SpaceTrackQuery(RequestClass.GP)
        assert query.output_format() == OutputFormat.JSON

    def test_output_format_set(self):
        query = SpaceTrackQuery(RequestClass.GP).format(OutputFormat.TLE)
        assert query.output_format() == OutputFormat.TLE

    def test_request_class(self):
        query = SpaceTrackQuery(RequestClass.GP)
        assert query.request_class() == RequestClass.GP

    def test_request_class_satcat(self):
        query = SpaceTrackQuery(RequestClass.SATCAT)
        assert query.request_class() == RequestClass.SATCAT
