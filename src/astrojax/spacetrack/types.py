"""SpaceTrack enum types.

Defines RequestController, RequestClass, SortOrder, and OutputFormat enums
for constructing SpaceTrack API queries.
"""

from enum import Enum


class RequestController(Enum):
    """SpaceTrack API request controller.

    Controllers determine which API endpoint is used. Most queries use
    ``BASIC_SPACE_DATA``.
    """

    BASIC_SPACE_DATA = "basicspacedata"
    EXPANDED_SPACE_DATA = "expandedspacedata"
    FILE_SHARE = "fileshare"
    SP_EPHEMERIS = "spephemeris"
    PUBLIC_FILES = "publicfiles"

    def as_str(self) -> str:
        """Return the URL path segment for this controller."""
        return self.value

    def __str__(self) -> str:
        return _CONTROLLER_DISPLAY[self]

    def __repr__(self) -> str:
        return f"RequestController.{_CONTROLLER_DISPLAY[self]}"


_CONTROLLER_DISPLAY = {
    RequestController.BASIC_SPACE_DATA: "BasicSpaceData",
    RequestController.EXPANDED_SPACE_DATA: "ExpandedSpaceData",
    RequestController.FILE_SHARE: "FileShare",
    RequestController.SP_EPHEMERIS: "SPEphemeris",
    RequestController.PUBLIC_FILES: "PublicFiles",
}


class RequestClass(Enum):
    """SpaceTrack API request class.

    Each request class corresponds to a specific type of data available
    from Space-Track.org.
    """

    GP = "gp"
    GP_HISTORY = "gp_history"
    SATCAT = "satcat"
    SATCAT_CHANGE = "satcat_change"
    SATCAT_DEBUT = "satcat_debut"
    DECAY = "decay"
    TIP = "tip"
    CDM_PUBLIC = "cdm_public"
    BOXSCORE = "boxscore"
    ANNOUNCEMENT = "announcement"
    LAUNCH_SITE = "launch_site"

    def as_str(self) -> str:
        """Return the URL path segment for this request class."""
        return self.value

    def default_controller(self) -> RequestController:
        """Return the default controller for this request class."""
        return RequestController.BASIC_SPACE_DATA

    def __str__(self) -> str:
        return _CLASS_DISPLAY[self]

    def __repr__(self) -> str:
        return f"RequestClass.{_CLASS_DISPLAY[self]}"


_CLASS_DISPLAY = {
    RequestClass.GP: "GP",
    RequestClass.GP_HISTORY: "GPHistory",
    RequestClass.SATCAT: "SATCAT",
    RequestClass.SATCAT_CHANGE: "SATCATChange",
    RequestClass.SATCAT_DEBUT: "SATCATDebut",
    RequestClass.DECAY: "Decay",
    RequestClass.TIP: "TIP",
    RequestClass.CDM_PUBLIC: "CDMPublic",
    RequestClass.BOXSCORE: "Boxscore",
    RequestClass.ANNOUNCEMENT: "Announcement",
    RequestClass.LAUNCH_SITE: "LaunchSite",
}


class SortOrder(Enum):
    """Sort order for query results."""

    ASC = "asc"
    DESC = "desc"

    def as_str(self) -> str:
        """Return the URL path segment for this sort order."""
        return self.value

    def __str__(self) -> str:
        return _SORT_DISPLAY[self]

    def __repr__(self) -> str:
        return f"SortOrder.{_SORT_DISPLAY[self]}"


_SORT_DISPLAY = {
    SortOrder.ASC: "Asc",
    SortOrder.DESC: "Desc",
}


class OutputFormat(Enum):
    """Output format for query results."""

    JSON = "json"
    XML = "xml"
    HTML = "html"
    CSV = "csv"
    TLE = "tle"
    THREE_LE = "3le"
    KVN = "kvn"

    def as_str(self) -> str:
        """Return the URL path segment for this output format."""
        return self.value

    def is_json(self) -> bool:
        """Return True if this format produces JSON output."""
        return self is OutputFormat.JSON

    def __str__(self) -> str:
        return _FORMAT_DISPLAY[self]

    def __repr__(self) -> str:
        return f"OutputFormat.{_FORMAT_DISPLAY[self]}"


_FORMAT_DISPLAY = {
    OutputFormat.JSON: "JSON",
    OutputFormat.XML: "XML",
    OutputFormat.HTML: "HTML",
    OutputFormat.CSV: "CSV",
    OutputFormat.TLE: "TLE",
    OutputFormat.THREE_LE: "3LE",
    OutputFormat.KVN: "KVN",
}
