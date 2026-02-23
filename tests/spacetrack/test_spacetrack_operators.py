"""Tests for SpaceTrack operator functions."""

from astrojax.spacetrack import operators as op


class TestOperators:
    """Tests for SpaceTrack operator functions."""

    def test_greater_than(self):
        assert op.greater_than("25544") == ">25544"

    def test_less_than(self):
        assert op.less_than("0.01") == "<0.01"

    def test_not_equal(self):
        assert op.not_equal("DEBRIS") == "<>DEBRIS"

    def test_inclusive_range(self):
        assert op.inclusive_range("1", "100") == "1--100"

    def test_like(self):
        assert op.like("ISS") == "~~ISS"

    def test_startswith(self):
        assert op.startswith("2024") == "^2024"

    def test_now(self):
        assert op.now() == "now"

    def test_now_offset_negative(self):
        assert op.now_offset(-7) == "now-7"

    def test_now_offset_positive(self):
        assert op.now_offset(14) == "now+14"

    def test_now_offset_zero(self):
        assert op.now_offset(0) == "now+0"

    def test_null_val(self):
        assert op.null_val() == "null-val"

    def test_or_list(self):
        assert op.or_list(["25544", "25545", "25546"]) == "25544,25545,25546"

    def test_or_list_single(self):
        assert op.or_list(["25544"]) == "25544"

    def test_operator_composition(self):
        """Test operators can be composed as query filter values."""
        result = op.greater_than(op.now_offset(-7))
        assert result == ">now-7"


class TestOperatorsModule:
    """Tests for module-level operator function access."""

    def test_module_functions(self):
        """Operator functions are accessible as module-level functions."""
        from astrojax.spacetrack._operators import (
            greater_than,
            inclusive_range,
            less_than,
            like,
            not_equal,
            now,
            now_offset,
            null_val,
            or_list,
            startswith,
        )

        assert greater_than("x") == ">x"
        assert less_than("x") == "<x"
        assert not_equal("x") == "<>x"
        assert inclusive_range("a", "b") == "a--b"
        assert like("x") == "~~x"
        assert startswith("x") == "^x"
        assert now() == "now"
        assert now_offset(5) == "now+5"
        assert null_val() == "null-val"
        assert or_list(["a", "b"]) == "a,b"
