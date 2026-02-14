"""Celestrak type enums.

Provides enums for query types, output formats, and supplemental GP sources.
"""

from __future__ import annotations

from enum import Enum

_QUERY_TYPE_DISPLAY: dict[str, str] = {
    "gp": "GP",
    "sup_gp": "SupGP",
    "satcat": "SATCAT",
}

_OUTPUT_FORMAT_DISPLAY: dict[str, str] = {
    "TLE": "Tle",
    "2LE": "TwoLe",
    "3LE": "ThreeLe",
    "XML": "Xml",
    "KVN": "Kvn",
    "JSON": "Json",
    "JSON-PRETTY": "JsonPretty",
    "CSV": "Csv",
}

_SUP_GP_SOURCE_DISPLAY: dict[str, str] = {
    "spacex": "SpaceX",
    "spacex-sup": "SpaceXSup",
    "planet": "Planet",
    "oneweb": "OneWeb",
    "starlink": "Starlink",
    "starlink-sup": "StarlinkSup",
    "geo": "Geo",
    "gps": "Gps",
    "glonass": "Glonass",
    "meteosat": "Meteosat",
    "intelsat": "Intelsat",
    "ses": "Ses",
    "iridium": "Iridium",
    "iridium-next": "IridiumNext",
    "orbcomm": "Orbcomm",
    "globalstar": "Globalstar",
    "swarm": "SwarmTechnologies",
    "amateur": "Amateur",
    "celestrak": "CelesTrak",
    "kuiper": "Kuiper",
}


class CelestrakQueryType(Enum):
    """Celestrak query endpoint type."""

    GP = "gp"
    SUP_GP = "sup_gp"
    SATCAT = "satcat"

    def as_str(self) -> str:
        """Return the query parameter value."""
        return self.value

    def endpoint_path(self) -> str:
        """Return the API endpoint path for this query type."""
        _paths = {
            CelestrakQueryType.GP: "/NORAD/elements/gp.php",
            CelestrakQueryType.SUP_GP: "/NORAD/elements/supplemental/sup-gp.php",
            CelestrakQueryType.SATCAT: "/satcat/records.php",
        }
        return _paths[self]

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"CelestrakQueryType.{_QUERY_TYPE_DISPLAY[self.value]}"


class CelestrakOutputFormat(Enum):
    """Output format for Celestrak query results."""

    TLE = "TLE"
    TWO_LE = "2LE"
    THREE_LE = "3LE"
    XML = "XML"
    KVN = "KVN"
    JSON = "JSON"
    JSON_PRETTY = "JSON-PRETTY"
    CSV = "CSV"

    def as_str(self) -> str:
        """Return the query parameter value."""
        return self.value

    def is_json(self) -> bool:
        """Return True if this format produces JSON output."""
        return self in (CelestrakOutputFormat.JSON, CelestrakOutputFormat.JSON_PRETTY)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"CelestrakOutputFormat.{_OUTPUT_FORMAT_DISPLAY[self.value]}"


class SupGPSource(Enum):
    """Supplemental GP data source."""

    SPACEX = "spacex"
    SPACEX_SUP = "spacex-sup"
    PLANET = "planet"
    ONEWEB = "oneweb"
    STARLINK = "starlink"
    STARLINK_SUP = "starlink-sup"
    GEO = "geo"
    GPS = "gps"
    GLONASS = "glonass"
    METEOSAT = "meteosat"
    INTELSAT = "intelsat"
    SES = "ses"
    IRIDIUM = "iridium"
    IRIDIUM_NEXT = "iridium-next"
    ORBCOMM = "orbcomm"
    GLOBALSTAR = "globalstar"
    SWARM_TECHNOLOGIES = "swarm"
    AMATEUR = "amateur"
    CELESTRAK = "celestrak"
    KUIPER = "kuiper"

    def as_str(self) -> str:
        """Return the query parameter value."""
        return self.value

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"SupGPSource.{_SUP_GP_SOURCE_DISPLAY[self.value]}"
