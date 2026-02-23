"""Tests for SpaceTrack enum types."""

from astrojax.spacetrack import (
    OutputFormat,
    RequestClass,
    RequestController,
    SortOrder,
)


class TestRequestController:
    """Tests for RequestController enum."""

    def test_variants(self):
        assert str(RequestController.BASIC_SPACE_DATA) == "BasicSpaceData"
        assert str(RequestController.EXPANDED_SPACE_DATA) == "ExpandedSpaceData"
        assert str(RequestController.FILE_SHARE) == "FileShare"
        assert str(RequestController.SP_EPHEMERIS) == "SPEphemeris"
        assert str(RequestController.PUBLIC_FILES) == "PublicFiles"

    def test_as_str(self):
        assert RequestController.BASIC_SPACE_DATA.as_str() == "basicspacedata"
        assert RequestController.EXPANDED_SPACE_DATA.as_str() == "expandedspacedata"
        assert RequestController.FILE_SHARE.as_str() == "fileshare"
        assert RequestController.SP_EPHEMERIS.as_str() == "spephemeris"
        assert RequestController.PUBLIC_FILES.as_str() == "publicfiles"

    def test_repr(self):
        assert "RequestController" in repr(RequestController.BASIC_SPACE_DATA)


class TestRequestClass:
    """Tests for RequestClass enum."""

    def test_variants(self):
        assert str(RequestClass.GP) == "GP"
        assert str(RequestClass.GP_HISTORY) == "GPHistory"
        assert str(RequestClass.SATCAT) == "SATCAT"
        assert str(RequestClass.SATCAT_CHANGE) == "SATCATChange"
        assert str(RequestClass.SATCAT_DEBUT) == "SATCATDebut"
        assert str(RequestClass.DECAY) == "Decay"
        assert str(RequestClass.TIP) == "TIP"
        assert str(RequestClass.CDM_PUBLIC) == "CDMPublic"
        assert str(RequestClass.BOXSCORE) == "Boxscore"
        assert str(RequestClass.ANNOUNCEMENT) == "Announcement"
        assert str(RequestClass.LAUNCH_SITE) == "LaunchSite"

    def test_default_controller(self):
        assert RequestClass.GP.default_controller() == RequestController.BASIC_SPACE_DATA
        assert RequestClass.SATCAT.default_controller() == RequestController.BASIC_SPACE_DATA

    def test_as_str(self):
        assert RequestClass.GP.as_str() == "gp"
        assert RequestClass.SATCAT.as_str() == "satcat"

    def test_equality(self):
        assert RequestClass.GP == RequestClass.GP
        assert RequestClass.GP != RequestClass.SATCAT

    def test_repr(self):
        assert "RequestClass" in repr(RequestClass.GP)


class TestSortOrder:
    """Tests for SortOrder enum."""

    def test_variants(self):
        assert str(SortOrder.ASC) == "Asc"
        assert str(SortOrder.DESC) == "Desc"

    def test_as_str(self):
        assert SortOrder.ASC.as_str() == "asc"
        assert SortOrder.DESC.as_str() == "desc"

    def test_equality(self):
        assert SortOrder.ASC == SortOrder.ASC
        assert SortOrder.ASC != SortOrder.DESC

    def test_repr(self):
        assert "SortOrder" in repr(SortOrder.ASC)


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_variants(self):
        assert str(OutputFormat.JSON) == "JSON"
        assert str(OutputFormat.XML) == "XML"
        assert str(OutputFormat.HTML) == "HTML"
        assert str(OutputFormat.CSV) == "CSV"
        assert str(OutputFormat.TLE) == "TLE"
        assert str(OutputFormat.THREE_LE) == "3LE"
        assert str(OutputFormat.KVN) == "KVN"

    def test_is_json(self):
        assert OutputFormat.JSON.is_json() is True
        assert OutputFormat.XML.is_json() is False
        assert OutputFormat.TLE.is_json() is False

    def test_as_str(self):
        assert OutputFormat.JSON.as_str() == "json"
        assert OutputFormat.TLE.as_str() == "tle"

    def test_equality(self):
        assert OutputFormat.JSON == OutputFormat.JSON
        assert OutputFormat.JSON != OutputFormat.TLE

    def test_repr(self):
        assert "OutputFormat" in repr(OutputFormat.JSON)
