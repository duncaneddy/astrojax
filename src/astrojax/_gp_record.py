"""General Perturbations (GP) record dataclass.

Shared data type for OMM/GP orbital element data from both Celestrak
and SpaceTrack APIs. All fields are optional since APIs return varying
subsets of the full OMM standard.
"""

from __future__ import annotations

from dataclasses import dataclass

from astrojax.sgp4._tle import parse_omm
from astrojax.sgp4._types import SGP4Elements


def _flex_float(val: object) -> float | None:
    """Convert a value to float, handling both numeric and string JSON values."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _flex_int(val: object) -> int | None:
    """Convert a value to int, handling both numeric and string JSON values."""
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


@dataclass
class GPRecord:
    """General Perturbations (GP) orbital element record.

    Contains orbital element data from Celestrak or SpaceTrack OMM/GP
    endpoints. All fields are optional since APIs return varying subsets.

    This is a pure-Python data container (not a JAX pytree).
    """

    object_name: str | None = None
    object_id: str | None = None
    norad_cat_id: int | None = None
    object_type: str | None = None
    classification_type: str | None = None
    intldes: str | None = None
    epoch: str | None = None
    mean_motion: float | None = None
    eccentricity: float | None = None
    inclination: float | None = None
    ra_of_asc_node: float | None = None
    arg_of_pericenter: float | None = None
    mean_anomaly: float | None = None
    ephemeris_type: int | None = None
    element_set_no: int | None = None
    rev_at_epoch: int | None = None
    bstar: float | None = None
    mean_motion_dot: float | None = None
    mean_motion_ddot: float | None = None
    semimajor_axis: float | None = None
    period: float | None = None
    apoapsis: float | None = None
    periapsis: float | None = None
    rcs_size: str | None = None
    country_code: str | None = None
    launch_date: str | None = None
    site: str | None = None
    decay_date: str | None = None
    file: int | None = None
    gp_id: int | None = None
    tle_line0: str | None = None
    tle_line1: str | None = None
    tle_line2: str | None = None

    @classmethod
    def from_json_dict(cls, d: dict) -> GPRecord:
        """Create a GPRecord from a JSON dict with uppercase keys.

        Handles both numeric JSON values (Celestrak) and string JSON
        values (SpaceTrack) via flexible type conversion.

        Args:
            d: Dictionary with uppercase OMM field names as keys.

        Returns:
            A new GPRecord instance.
        """
        return cls(
            object_name=d.get("OBJECT_NAME"),
            object_id=d.get("OBJECT_ID"),
            norad_cat_id=_flex_int(d.get("NORAD_CAT_ID")),
            object_type=d.get("OBJECT_TYPE"),
            classification_type=d.get("CLASSIFICATION_TYPE"),
            intldes=d.get("INTLDES"),
            epoch=d.get("EPOCH"),
            mean_motion=_flex_float(d.get("MEAN_MOTION")),
            eccentricity=_flex_float(d.get("ECCENTRICITY")),
            inclination=_flex_float(d.get("INCLINATION")),
            ra_of_asc_node=_flex_float(d.get("RA_OF_ASC_NODE")),
            arg_of_pericenter=_flex_float(d.get("ARG_OF_PERICENTER")),
            mean_anomaly=_flex_float(d.get("MEAN_ANOMALY")),
            ephemeris_type=_flex_int(d.get("EPHEMERIS_TYPE")),
            element_set_no=_flex_int(d.get("ELEMENT_SET_NO")),
            rev_at_epoch=_flex_int(d.get("REV_AT_EPOCH")),
            bstar=_flex_float(d.get("BSTAR")),
            mean_motion_dot=_flex_float(d.get("MEAN_MOTION_DOT")),
            mean_motion_ddot=_flex_float(d.get("MEAN_MOTION_DDOT")),
            semimajor_axis=_flex_float(d.get("SEMIMAJOR_AXIS")),
            period=_flex_float(d.get("PERIOD")),
            apoapsis=_flex_float(d.get("APOAPSIS")),
            periapsis=_flex_float(d.get("PERIAPSIS")),
            rcs_size=d.get("RCS_SIZE"),
            country_code=d.get("COUNTRY_CODE"),
            launch_date=d.get("LAUNCH_DATE"),
            site=d.get("SITE"),
            decay_date=d.get("DECAY_DATE"),
            file=_flex_int(d.get("FILE")),
            gp_id=_flex_int(d.get("GP_ID")),
            tle_line0=d.get("TLE_LINE0"),
            tle_line1=d.get("TLE_LINE1"),
            tle_line2=d.get("TLE_LINE2"),
        )

    def get_field(self, name: str) -> str | None:
        """Get a field value by its uppercase API name.

        Used by the filter engine for client-side filtering.

        Args:
            name: Uppercase field name (e.g. ``"OBJECT_NAME"``).

        Returns:
            String representation of the field value, or None if unset.
        """
        attr_name = name.lower()
        try:
            val = getattr(self, attr_name)
        except AttributeError:
            return None
        if val is None:
            return None
        return str(val)

    def to_sgp4_elements(self) -> SGP4Elements:
        """Convert this GP record to SGP4 orbital elements.

        Delegates to :func:`~astrojax.sgp4._tle.parse_omm` using the
        OMM field names expected by that function.

        Returns:
            Parsed orbital elements ready for SGP4 initialization.

        Raises:
            KeyError: If required OMM fields are missing.
            ValueError: If epoch cannot be parsed.
        """
        fields: dict[str, str] = {}

        if self.epoch is not None:
            fields["EPOCH"] = self.epoch
        if self.mean_motion is not None:
            fields["MEAN_MOTION"] = str(self.mean_motion)
        if self.eccentricity is not None:
            fields["ECCENTRICITY"] = str(self.eccentricity)
        if self.inclination is not None:
            fields["INCLINATION"] = str(self.inclination)
        if self.ra_of_asc_node is not None:
            fields["RA_OF_ASC_NODE"] = str(self.ra_of_asc_node)
        if self.arg_of_pericenter is not None:
            fields["ARG_OF_PERICENTER"] = str(self.arg_of_pericenter)
        if self.mean_anomaly is not None:
            fields["MEAN_ANOMALY"] = str(self.mean_anomaly)
        if self.norad_cat_id is not None:
            fields["NORAD_CAT_ID"] = str(self.norad_cat_id)
        if self.bstar is not None:
            fields["BSTAR"] = str(self.bstar)
        if self.classification_type is not None:
            fields["CLASSIFICATION_TYPE"] = self.classification_type
        if self.object_id is not None:
            fields["OBJECT_ID"] = self.object_id
        if self.ephemeris_type is not None:
            fields["EPHEMERIS_TYPE"] = str(self.ephemeris_type)
        if self.element_set_no is not None:
            fields["ELEMENT_SET_NO"] = str(self.element_set_no)
        if self.rev_at_epoch is not None:
            fields["REV_AT_EPOCH"] = str(self.rev_at_epoch)
        if self.mean_motion_dot is not None:
            fields["MEAN_MOTION_DOT"] = str(self.mean_motion_dot)
        if self.mean_motion_ddot is not None:
            fields["MEAN_MOTION_DDOT"] = str(self.mean_motion_ddot)

        return parse_omm(fields)

    def __str__(self) -> str:
        return f"GPRecord(name={self.object_name!r}, norad_id={self.norad_cat_id!r})"

    def __repr__(self) -> str:
        return self.__str__()
