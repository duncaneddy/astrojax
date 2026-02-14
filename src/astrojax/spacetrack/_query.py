"""SpaceTrack query builder.

Provides a fluent builder API for constructing Space-Track.org API queries.
"""

from __future__ import annotations

from astrojax.spacetrack._types import (
    OutputFormat,
    RequestClass,
    RequestController,
    SortOrder,
)


def _encode_path_value(s: str) -> str:
    """Percent-encode characters that are invalid in URI path segments.

    Args:
        s: Raw value string.

    Returns:
        Encoded string safe for use in URL paths.
    """
    return s.replace(">", "%3E").replace("<", "%3C").replace("^", "%5E")


class SpaceTrackQuery:
    """Builder for constructing SpaceTrack API queries.

    Uses an immutable fluent API where each method returns a new instance.

    Args:
        request_class: The type of data to query.
    """

    def __init__(self, request_class: RequestClass) -> None:
        self._controller = request_class.default_controller()
        self._class = request_class
        self._filters: list[tuple[str, str]] = []
        self._order_by: list[tuple[str, SortOrder]] = []
        self._limit_count: int | None = None
        self._limit_offset: int | None = None
        self._output_format: OutputFormat | None = None
        self._predicates: list[str] = []
        self._metadata: bool = False
        self._distinct: bool = False
        self._empty_result: bool = False
        self._favorites: str | None = None

    def _clone(self) -> SpaceTrackQuery:
        """Return a shallow copy of this query."""
        new = SpaceTrackQuery.__new__(SpaceTrackQuery)
        new._controller = self._controller
        new._class = self._class
        new._filters = list(self._filters)
        new._order_by = list(self._order_by)
        new._limit_count = self._limit_count
        new._limit_offset = self._limit_offset
        new._output_format = self._output_format
        new._predicates = list(self._predicates)
        new._metadata = self._metadata
        new._distinct = self._distinct
        new._empty_result = self._empty_result
        new._favorites = self._favorites
        return new

    def controller(self, controller: RequestController) -> SpaceTrackQuery:
        """Override the default controller for this query.

        Args:
            controller: The request controller to use.
        """
        new = self._clone()
        new._controller = controller
        return new

    def filter(self, field: str, value: str) -> SpaceTrackQuery:
        """Add a filter predicate to the query.

        Args:
            field: Uppercase field name (e.g. ``"NORAD_CAT_ID"``).
            value: Filter value with optional operator prefix.
        """
        new = self._clone()
        new._filters.append((field, value))
        return new

    def order_by(self, field: str, order: SortOrder) -> SpaceTrackQuery:
        """Add an ordering clause to the query.

        Args:
            field: Uppercase field name to sort by.
            order: Sort direction.
        """
        new = self._clone()
        new._order_by.append((field, order))
        return new

    def limit(self, count: int) -> SpaceTrackQuery:
        """Set the maximum number of results to return.

        Args:
            count: Maximum result count.
        """
        new = self._clone()
        new._limit_count = count
        return new

    def limit_offset(self, count: int, offset: int) -> SpaceTrackQuery:
        """Set the maximum number of results and an offset.

        Args:
            count: Maximum result count.
            offset: Number of results to skip.
        """
        new = self._clone()
        new._limit_count = count
        new._limit_offset = offset
        return new

    def format(self, fmt: OutputFormat) -> SpaceTrackQuery:
        """Set the output format for query results.

        Args:
            fmt: Output format.
        """
        new = self._clone()
        new._output_format = fmt
        return new

    def predicates_filter(self, fields: list[str]) -> SpaceTrackQuery:
        """Specify which fields to include in the response.

        Args:
            fields: List of uppercase field names.
        """
        new = self._clone()
        new._predicates = list(fields)
        return new

    def metadata(self, enabled: bool) -> SpaceTrackQuery:
        """Enable or disable metadata in the response.

        Args:
            enabled: Whether to include metadata.
        """
        new = self._clone()
        new._metadata = enabled
        return new

    def distinct(self, enabled: bool) -> SpaceTrackQuery:
        """Enable or disable distinct results.

        Args:
            enabled: Whether to return distinct results.
        """
        new = self._clone()
        new._distinct = enabled
        return new

    def empty_result(self, enabled: bool) -> SpaceTrackQuery:
        """Enable or disable empty result return.

        Args:
            enabled: Whether to show empty results.
        """
        new = self._clone()
        new._empty_result = enabled
        return new

    def favorites(self, favorites: str) -> SpaceTrackQuery:
        """Set a favorites filter for the query.

        Args:
            favorites: Favorites identifier.
        """
        new = self._clone()
        new._favorites = favorites
        return new

    def build(self) -> str:
        """Build the URL path string for this query.

        Returns:
            URL path string (e.g.
            ``"/basicspacedata/query/class/gp/NORAD_CAT_ID/25544/format/json"``).
        """
        parts: list[str] = []

        # Controller and query prefix
        parts.append(
            f"/{self._controller.as_str()}/query/class/{self._class.as_str()}"
        )

        # Filters
        for field, value in self._filters:
            parts.append(f"/{field}/{_encode_path_value(value)}")

        # Order by
        if self._order_by:
            order_str = ",".join(
                f"{field}%20{order.as_str()}" for field, order in self._order_by
            )
            parts.append(f"/orderby/{order_str}")

        # Limit
        if self._limit_count is not None:
            if self._limit_offset is not None:
                parts.append(f"/limit/{self._limit_count},{self._limit_offset}")
            else:
                parts.append(f"/limit/{self._limit_count}")

        # Predicates filter
        if self._predicates:
            parts.append(f"/predicates/{','.join(self._predicates)}")

        # Metadata
        if self._metadata:
            parts.append("/metadata/true")

        # Distinct
        if self._distinct:
            parts.append("/distinct/true")

        # Empty result
        if self._empty_result:
            parts.append("/emptyresult/show")

        # Favorites
        if self._favorites is not None:
            parts.append(f"/favorites/{self._favorites}")

        # Format (default to JSON)
        fmt = self._output_format if self._output_format is not None else OutputFormat.JSON
        parts.append(f"/format/{fmt.as_str()}")

        return "".join(parts)

    def output_format(self) -> OutputFormat:
        """Return the output format for this query.

        Returns:
            The output format (defaults to JSON if not explicitly set).
        """
        return self._output_format if self._output_format is not None else OutputFormat.JSON

    def request_class(self) -> RequestClass:
        """Return the request class for this query."""
        return self._class

    def __str__(self) -> str:
        return self.build()

    def __repr__(self) -> str:
        return f'SpaceTrackQuery("{self.build()}")'
