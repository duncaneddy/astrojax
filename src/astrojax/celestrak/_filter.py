"""Client-side filtering engine for Celestrak query results.

Parses SpaceTrack-compatible operator strings and applies them
as filters to downloaded records.

Supported operators:

- ``>value`` -- Greater than
- ``<value`` -- Less than
- ``<>value`` -- Not equal (case-insensitive)
- ``min--max`` -- Inclusive range
- ``~~pattern`` -- Case-insensitive substring match (like)
- ``^prefix`` -- Case-insensitive prefix match (starts with)
- ``value`` -- Exact string match
"""

from __future__ import annotations

import functools
from typing import Any, Protocol


class HasGetField(Protocol):
    """Protocol for records that support field access by name."""

    def get_field(self, name: str) -> str | None: ...


def get_field(record: Any, name: str) -> str | None:
    """Get a field value from any record type.

    Supports records with a ``get_field`` method (GPRecord,
    CelestrakSATCATRecord) or arbitrary attribute access.

    Args:
        record: A record object with field access.
        name: Uppercase field name (e.g. ``"OBJECT_NAME"``).

    Returns:
        String representation of the field value, or None if unset.
    """
    if hasattr(record, "get_field"):
        return record.get_field(name)
    attr_name = name.lower()
    try:
        val = getattr(record, attr_name)
    except AttributeError:
        return None
    if val is None:
        return None
    return str(val)


def _parse_filter_value(value: str) -> tuple[str, str | tuple[str, str]]:
    """Parse a filter value string into (operator, operand).

    Args:
        value: Filter value string (e.g. ``">50"``, ``"10--20"``).

    Returns:
        A tuple of (op_type, operand) where op_type is one of:
        ``"gt"``, ``"lt"``, ``"ne"``, ``"range"``, ``"like"``,
        ``"startswith"``, ``"exact"``.
        For ``"range"``, operand is a tuple ``(min, max)``.
    """
    if value.startswith("<>"):
        return ("ne", value[2:])
    if value.startswith("~~"):
        return ("like", value[2:])
    if value.startswith(">"):
        return ("gt", value[1:])
    if value.startswith("<"):
        return ("lt", value[1:])
    if value.startswith("^"):
        return ("startswith", value[1:])
    dash_pos = value.find("--")
    if dash_pos >= 0:
        return ("range", (value[:dash_pos], value[dash_pos + 2:]))
    return ("exact", value)


def _compare_values(a: str, b: str) -> int | None:
    """Compare two string values, attempting numeric comparison first.

    Args:
        a: First value.
        b: Second value.

    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b, None if incomparable.
    """
    try:
        a_num = float(a)
        b_num = float(b)
        if a_num < b_num:
            return -1
        elif a_num > b_num:
            return 1
        else:
            return 0
    except (ValueError, TypeError):
        pass
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


def _matches_filter(record: Any, field: str, value: str) -> bool:
    """Check if a single filter matches a record.

    Args:
        record: A record object with field access.
        field: Uppercase field name.
        value: Filter value string with optional operator prefix.

    Returns:
        True if the record matches the filter.
    """
    field_value = get_field(record, field)
    if field_value is None:
        return False

    op_type, operand = _parse_filter_value(value)

    if op_type == "gt":
        return _compare_values(field_value, operand) == 1
    elif op_type == "lt":
        return _compare_values(field_value, operand) == -1
    elif op_type == "ne":
        return field_value.lower() != operand.lower()
    elif op_type == "range":
        min_val, max_val = operand
        cmp_min = _compare_values(field_value, min_val)
        cmp_max = _compare_values(field_value, max_val)
        return (
            cmp_min is not None
            and cmp_max is not None
            and cmp_min >= 0
            and cmp_max <= 0
        )
    elif op_type == "like":
        return operand.lower() in field_value.lower()
    elif op_type == "startswith":
        return field_value.lower().startswith(operand.lower())
    else:  # exact
        return field_value == operand


def apply_filters(
    records: list[Any], filters: list[tuple[str, str]]
) -> list[Any]:
    """Apply client-side filters to a list of records.

    Records must match ALL filters (AND logic).

    Args:
        records: List of records to filter.
        filters: List of (field, value) tuples.

    Returns:
        Filtered list of records.
    """
    if not filters:
        return records
    return [
        r
        for r in records
        if all(_matches_filter(r, field, value) for field, value in filters)
    ]


def apply_order_by(
    records: list[Any], order_by: list[tuple[str, bool]]
) -> None:
    """Apply client-side ordering to a list of records (in-place).

    Args:
        records: List of records to sort.
        order_by: List of (field, ascending) tuples.
    """
    if not order_by:
        return

    def compare(a: Any, b: Any) -> int:
        for field, ascending in order_by:
            a_val = get_field(a, field)
            b_val = get_field(b, field)
            if a_val is not None and b_val is not None:
                cmp = _compare_values(a_val, b_val)
                if cmp is None:
                    cmp = 0
            elif a_val is not None:
                cmp = -1
            elif b_val is not None:
                cmp = 1
            else:
                cmp = 0
            if not ascending:
                cmp = -cmp
            if cmp != 0:
                return cmp
        return 0

    records.sort(key=functools.cmp_to_key(compare))


def apply_limit(records: list[Any], limit: int | None) -> list[Any]:
    """Apply a client-side limit to truncate results.

    Args:
        records: List of records.
        limit: Maximum number of records, or None for no limit.

    Returns:
        Truncated list of records.
    """
    if limit is None:
        return records
    return records[:limit]
