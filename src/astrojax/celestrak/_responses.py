"""Typed response classes for Celestrak API responses.

Provides a dataclass for Celestrak's SATCAT endpoint.
GP query responses use GPRecord from astrojax._gp_record.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CelestrakSATCATRecord:
    """Celestrak Satellite Catalog (SATCAT) record.

    Contains metadata about a cataloged space object from Celestrak's
    SATCAT endpoint (``/satcat/records.php``).
    """

    object_name: str | None = None
    object_id: str | None = None
    norad_cat_id: int | None = None
    object_type: str | None = None
    ops_status_code: str | None = None
    owner: str | None = None
    launch_date: str | None = None
    launch_site: str | None = None
    decay_date: str | None = None
    period: str | None = None
    inclination: str | None = None
    apogee: str | None = None
    perigee: str | None = None
    rcs: str | None = None
    data_status_code: str | None = None
    orbit_center: str | None = None
    orbit_type: str | None = None

    @classmethod
    def from_json_dict(cls, d: dict) -> CelestrakSATCATRecord:
        """Create a CelestrakSATCATRecord from a JSON dict with uppercase keys.

        Args:
            d: Dictionary with uppercase field names as keys.

        Returns:
            A new CelestrakSATCATRecord instance.
        """
        norad_raw = d.get("NORAD_CAT_ID")
        norad_cat_id = None
        if norad_raw is not None:
            try:
                norad_cat_id = int(norad_raw)
            except (ValueError, TypeError):
                pass

        return cls(
            object_name=d.get("OBJECT_NAME"),
            object_id=d.get("OBJECT_ID"),
            norad_cat_id=norad_cat_id,
            object_type=d.get("OBJECT_TYPE"),
            ops_status_code=d.get("OPS_STATUS_CODE"),
            owner=d.get("OWNER"),
            launch_date=d.get("LAUNCH_DATE"),
            launch_site=d.get("LAUNCH_SITE"),
            decay_date=d.get("DECAY_DATE"),
            period=d.get("PERIOD"),
            inclination=d.get("INCLINATION"),
            apogee=d.get("APOGEE"),
            perigee=d.get("PERIGEE"),
            rcs=d.get("RCS"),
            data_status_code=d.get("DATA_STATUS_CODE"),
            orbit_center=d.get("ORBIT_CENTER"),
            orbit_type=d.get("ORBIT_TYPE"),
        )

    def get_field(self, name: str) -> str | None:
        """Get a field value by its uppercase API name.

        Used by the filter engine for client-side filtering.

        Args:
            name: Uppercase field name (e.g. ``"OBJECT_NAME"``).

        Returns:
            String representation of the field value, or None if unset.
        """
        _fields: dict[str, str | int | None] = {
            "OBJECT_NAME": self.object_name,
            "OBJECT_ID": self.object_id,
            "NORAD_CAT_ID": self.norad_cat_id,
            "OBJECT_TYPE": self.object_type,
            "OPS_STATUS_CODE": self.ops_status_code,
            "OWNER": self.owner,
            "LAUNCH_DATE": self.launch_date,
            "LAUNCH_SITE": self.launch_site,
            "DECAY_DATE": self.decay_date,
            "PERIOD": self.period,
            "INCLINATION": self.inclination,
            "APOGEE": self.apogee,
            "PERIGEE": self.perigee,
            "RCS": self.rcs,
            "DATA_STATUS_CODE": self.data_status_code,
            "ORBIT_CENTER": self.orbit_center,
            "ORBIT_TYPE": self.orbit_type,
        }
        val = _fields.get(name)
        if val is None:
            return None
        return str(val)

    def __str__(self) -> str:
        return (
            f"CelestrakSATCATRecord(name={self.object_name!r}, "
            f"norad_id={self.norad_cat_id!r})"
        )

    def __repr__(self) -> str:
        return self.__str__()
