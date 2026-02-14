"""Tests for Celestrak client-side filter engine."""

from astrojax._gp_record import GPRecord
from astrojax.celestrak._filter import (
    apply_filters,
    apply_limit,
    apply_order_by,
    get_field,
)
from astrojax.celestrak._responses import CelestrakSATCATRecord


class TestGetField:
    """Tests for get_field() function."""

    def test_gp_record(self):
        record = GPRecord(object_name="ISS", norad_cat_id=25544, inclination=51.6)
        assert get_field(record, "OBJECT_NAME") == "ISS"
        assert get_field(record, "NORAD_CAT_ID") == "25544"
        assert get_field(record, "INCLINATION") == "51.6"

    def test_gp_record_none_field(self):
        record = GPRecord(object_name="ISS")
        assert get_field(record, "NORAD_CAT_ID") is None

    def test_satcat_record(self):
        record = CelestrakSATCATRecord(
            object_name="ISS (ZARYA)",
            norad_cat_id=25544,
            inclination="51.6",
        )
        assert get_field(record, "OBJECT_NAME") == "ISS (ZARYA)"
        assert get_field(record, "NORAD_CAT_ID") == "25544"
        assert get_field(record, "INCLINATION") == "51.6"

    def test_nonexistent_field(self):
        record = GPRecord()
        assert get_field(record, "NONEXISTENT") is None


class TestFilterOperators:
    """Tests for filter operator parsing and matching."""

    def _make_records(self) -> list[GPRecord]:
        return [
            GPRecord(object_name="ISS", norad_cat_id=25544, inclination=51.6),
            GPRecord(object_name="HUBBLE", norad_cat_id=20580, inclination=28.5),
            GPRecord(object_name="DEBRIS A", norad_cat_id=99999, inclination=98.2),
        ]

    def test_greater_than(self):
        records = self._make_records()
        result = apply_filters(records, [("INCLINATION", ">50")])
        assert len(result) == 2  # ISS (51.6) and DEBRIS A (98.2)

    def test_less_than(self):
        records = self._make_records()
        result = apply_filters(records, [("INCLINATION", "<50")])
        assert len(result) == 1  # HUBBLE (28.5)

    def test_not_equal(self):
        records = self._make_records()
        result = apply_filters(records, [("OBJECT_NAME", "<>ISS")])
        assert len(result) == 2

    def test_range(self):
        records = self._make_records()
        result = apply_filters(records, [("INCLINATION", "28--52")])
        assert len(result) == 2  # HUBBLE (28.5) and ISS (51.6)

    def test_like(self):
        records = self._make_records()
        result = apply_filters(records, [("OBJECT_NAME", "~~debris")])
        assert len(result) == 1
        assert result[0].object_name == "DEBRIS A"

    def test_startswith(self):
        records = self._make_records()
        result = apply_filters(records, [("OBJECT_NAME", "^deb")])
        assert len(result) == 1
        assert result[0].object_name == "DEBRIS A"

    def test_exact_match(self):
        records = self._make_records()
        result = apply_filters(records, [("OBJECT_NAME", "ISS")])
        assert len(result) == 1
        assert result[0].object_name == "ISS"

    def test_and_logic(self):
        records = self._make_records()
        result = apply_filters(records, [
            ("INCLINATION", ">30"),
            ("INCLINATION", "<60"),
        ])
        assert len(result) == 1  # Only ISS (51.6)

    def test_empty_filters(self):
        records = self._make_records()
        result = apply_filters(records, [])
        assert len(result) == 3

    def test_no_match(self):
        records = self._make_records()
        result = apply_filters(records, [("OBJECT_NAME", "NONEXISTENT")])
        assert len(result) == 0

    def test_none_field_skipped(self):
        records = [GPRecord(object_name="ISS")]
        result = apply_filters(records, [("INCLINATION", ">50")])
        assert len(result) == 0


class TestOrderBy:
    """Tests for apply_order_by()."""

    def test_ascending(self):
        records = [
            GPRecord(norad_cat_id=99999, inclination=98.2),
            GPRecord(norad_cat_id=25544, inclination=51.6),
            GPRecord(norad_cat_id=20580, inclination=28.5),
        ]
        apply_order_by(records, [("INCLINATION", True)])
        assert [r.inclination for r in records] == [28.5, 51.6, 98.2]

    def test_descending(self):
        records = [
            GPRecord(norad_cat_id=20580, inclination=28.5),
            GPRecord(norad_cat_id=25544, inclination=51.6),
            GPRecord(norad_cat_id=99999, inclination=98.2),
        ]
        apply_order_by(records, [("INCLINATION", False)])
        assert [r.inclination for r in records] == [98.2, 51.6, 28.5]

    def test_empty_order(self):
        records = [
            GPRecord(norad_cat_id=99999),
            GPRecord(norad_cat_id=25544),
        ]
        apply_order_by(records, [])
        assert records[0].norad_cat_id == 99999

    def test_multi_field(self):
        records = [
            GPRecord(object_type="PAYLOAD", norad_cat_id=3),
            GPRecord(object_type="DEBRIS", norad_cat_id=1),
            GPRecord(object_type="PAYLOAD", norad_cat_id=2),
        ]
        apply_order_by(records, [("OBJECT_TYPE", True), ("NORAD_CAT_ID", True)])
        assert records[0].object_type == "DEBRIS"
        assert records[1].norad_cat_id == 2
        assert records[2].norad_cat_id == 3


class TestLimit:
    """Tests for apply_limit()."""

    def test_limit(self):
        records = [GPRecord(norad_cat_id=i) for i in range(10)]
        result = apply_limit(records, 3)
        assert len(result) == 3

    def test_limit_none(self):
        records = [GPRecord(norad_cat_id=i) for i in range(10)]
        result = apply_limit(records, None)
        assert len(result) == 10

    def test_limit_exceeds_count(self):
        records = [GPRecord(norad_cat_id=i) for i in range(3)]
        result = apply_limit(records, 10)
        assert len(result) == 3
