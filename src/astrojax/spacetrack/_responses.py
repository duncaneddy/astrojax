"""Typed response classes for SpaceTrack API responses.

Provides a dataclass for commonly queried SpaceTrack SATCAT data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SATCATRecord:
    """Satellite Catalog (SATCAT) record from SpaceTrack.

    Contains metadata about a cataloged space object.
    """

    intldes: str | None = None
    norad_cat_id: int | None = None
    object_type: str | None = None
    satname: str | None = None
    country: str | None = None
    launch: str | None = None
    site: str | None = None
    decay: str | None = None
    period: str | None = None
    inclination: str | None = None
    apogee: str | None = None
    perigee: str | None = None
    comment: str | None = None
    commentcode: str | None = None
    rcsvalue: str | None = None
    rcs_size: str | None = None
    file: str | None = None
    launch_year: str | None = None
    launch_num: str | None = None
    launch_piece: str | None = None
    current: str | None = None
    object_name: str | None = None
    object_id: str | None = None
    object_number: str | None = None

    @classmethod
    def from_json_dict(cls, d: dict) -> SATCATRecord:
        """Create a SATCATRecord from a JSON dict with uppercase keys.

        Args:
            d: Dictionary with uppercase field names as keys.

        Returns:
            A new SATCATRecord instance.
        """
        norad_raw = d.get("NORAD_CAT_ID")
        norad_cat_id = None
        if norad_raw is not None:
            try:
                norad_cat_id = int(norad_raw)
            except (ValueError, TypeError):
                pass

        return cls(
            intldes=d.get("INTLDES"),
            norad_cat_id=norad_cat_id,
            object_type=d.get("OBJECT_TYPE"),
            satname=d.get("SATNAME"),
            country=d.get("COUNTRY"),
            launch=d.get("LAUNCH"),
            site=d.get("SITE"),
            decay=d.get("DECAY"),
            period=d.get("PERIOD"),
            inclination=d.get("INCLINATION"),
            apogee=d.get("APOGEE"),
            perigee=d.get("PERIGEE"),
            comment=d.get("COMMENT"),
            commentcode=d.get("COMMENTCODE"),
            rcsvalue=d.get("RCSVALUE"),
            rcs_size=d.get("RCS_SIZE"),
            file=d.get("FILE"),
            launch_year=d.get("LAUNCH_YEAR"),
            launch_num=d.get("LAUNCH_NUM"),
            launch_piece=d.get("LAUNCH_PIECE"),
            current=d.get("CURRENT"),
            object_name=d.get("OBJECT_NAME"),
            object_id=d.get("OBJECT_ID"),
            object_number=d.get("OBJECT_NUMBER"),
        )

    def __str__(self) -> str:
        return f"SATCATRecord(name={self.satname!r}, norad_id={self.norad_cat_id!r})"

    def __repr__(self) -> str:
        return self.__str__()
