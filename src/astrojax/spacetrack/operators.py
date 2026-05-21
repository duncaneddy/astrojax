"""Query operator functions for building filter values.

These functions generate operator-prefixed strings for use in SpaceTrack
and Celestrak query filter values.
"""

from __future__ import annotations


def greater_than(value: str) -> str:
    """Greater-than operator: produces ``">value"``.

    Args:
        value: The comparison value.

    Returns:
        Operator-prefixed string.
    """
    return f">{value}"


def less_than(value: str) -> str:
    """Less-than operator: produces ``"<value"``.

    Args:
        value: The comparison value.

    Returns:
        Operator-prefixed string.
    """
    return f"<{value}"


def not_equal(value: str) -> str:
    """Not-equal operator: produces ``"<>value"``.

    Args:
        value: The comparison value.

    Returns:
        Operator-prefixed string.
    """
    return f"<>{value}"


def inclusive_range(left: str, right: str) -> str:
    """Inclusive range operator: produces ``"left--right"``.

    Args:
        left: Lower bound of the range.
        right: Upper bound of the range.

    Returns:
        Range operator string.
    """
    return f"{left}--{right}"


def like(value: str) -> str:
    """Like/contains operator: produces ``"~~value"``.

    Args:
        value: The substring to search for.

    Returns:
        Operator-prefixed string.
    """
    return f"~~{value}"


def startswith(value: str) -> str:
    """Starts-with operator: produces ``"^value"``.

    Args:
        value: The prefix to match.

    Returns:
        Operator-prefixed string.
    """
    return f"^{value}"


def now() -> str:
    """Current time reference: returns ``"now"``.

    Returns:
        The string ``"now"``.
    """
    return "now"


def now_offset(days: int) -> str:
    """Time offset from now: produces ``"now-N"`` or ``"now+N"``.

    Args:
        days: Number of days offset (negative for past, positive for future).

    Returns:
        Time offset string.
    """
    if days < 0:
        return f"now{days}"
    else:
        return f"now+{days}"


def null_val() -> str:
    """Null value reference: returns ``"null-val"``.

    Returns:
        The string ``"null-val"``.
    """
    return "null-val"


def or_list(values: list[str]) -> str:
    """OR list operator: produces ``"val1,val2,val3"``.

    Args:
        values: List of values to combine with OR.

    Returns:
        Comma-separated string.
    """
    return ",".join(str(v) for v in values)


class _OperatorsNamespace:
    """SpaceTrack query operator functions.

    Provides operator functions for constructing SpaceTrack query filters.

    Example::

        from astrojax.spacetrack import operators as op

        op.greater_than("25544")         # ">25544"
        op.less_than("0.01")             # "<0.01"
        op.inclusive_range("1", "100")    # "1--100"
        op.now_offset(-7)                # "now-7"
    """

    greater_than = staticmethod(greater_than)
    less_than = staticmethod(less_than)
    not_equal = staticmethod(not_equal)
    inclusive_range = staticmethod(inclusive_range)
    like = staticmethod(like)
    startswith = staticmethod(startswith)
    now = staticmethod(now)
    now_offset = staticmethod(now_offset)
    null_val = staticmethod(null_val)
    or_list = staticmethod(or_list)


operators = _OperatorsNamespace()
