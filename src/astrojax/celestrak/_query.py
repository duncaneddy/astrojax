"""Celestrak query builder.

Provides a fluent builder API for constructing Celestrak API queries
with both server-side parameters and client-side post-processing options.
"""

from __future__ import annotations

from astrojax.celestrak._types import (
    CelestrakOutputFormat,
    CelestrakQueryType,
    SupGPSource,
)

_QUERY_TYPE_DEBUG: dict[CelestrakQueryType, str] = {
    CelestrakQueryType.GP: "GP",
    CelestrakQueryType.SUP_GP: "SupGP",
    CelestrakQueryType.SATCAT: "SATCAT",
}


class _QueryFactory:
    """Descriptor that creates a fresh CelestrakQuery on each access.

    Allows ``CelestrakQuery.gp`` to return a new query instance each time,
    enabling fluent builder patterns without shared mutable state.
    """

    def __init__(self, query_type: CelestrakQueryType) -> None:
        self._query_type = query_type

    def __get__(self, obj: object, objtype: type | None = None) -> CelestrakQuery:
        return CelestrakQuery(self._query_type)


class CelestrakQuery:
    """Builder for constructing Celestrak API queries.

    Uses a fluent API where each method returns a new instance.
    Parameters are divided into server-side (sent to Celestrak)
    and client-side (applied after download).

    Use the class properties ``gp``, ``sup_gp``, and ``satcat``
    to create queries targeting the respective endpoints.
    """

    def __init__(self, query_type: CelestrakQueryType) -> None:
        self._query_type = query_type
        # Server-side parameters
        self._group: str | None = None
        self._catnr: int | None = None
        self._intdes: str | None = None
        self._name: str | None = None
        self._special: str | None = None
        self._source: SupGPSource | None = None
        self._file: str | None = None
        self._payloads: bool | None = None
        self._on_orbit: bool | None = None
        self._active: bool | None = None
        self._max_results: int | None = None
        self._output_format: CelestrakOutputFormat | None = None
        # Client-side parameters
        self._filters: list[tuple[str, str]] = []
        self._order_by_clauses: list[tuple[str, bool]] = []
        self._limit_count: int | None = None

    def _clone(self) -> CelestrakQuery:
        """Return a shallow copy of this query."""
        new = CelestrakQuery.__new__(CelestrakQuery)
        new._query_type = self._query_type
        new._group = self._group
        new._catnr = self._catnr
        new._intdes = self._intdes
        new._name = self._name
        new._special = self._special
        new._source = self._source
        new._file = self._file
        new._payloads = self._payloads
        new._on_orbit = self._on_orbit
        new._active = self._active
        new._max_results = self._max_results
        new._output_format = self._output_format
        new._filters = list(self._filters)
        new._order_by_clauses = list(self._order_by_clauses)
        new._limit_count = self._limit_count
        return new

    gp = _QueryFactory(CelestrakQueryType.GP)
    sup_gp = _QueryFactory(CelestrakQueryType.SUP_GP)
    satcat = _QueryFactory(CelestrakQueryType.SATCAT)

    # -- Server-side parameters --

    def group(self, name: str) -> CelestrakQuery:
        """Set the satellite group for the query.

        Args:
            name: Satellite group name (e.g. ``"stations"``).
        """
        new = self._clone()
        new._group = name
        return new

    def catnr(self, norad_id: int) -> CelestrakQuery:
        """Set the NORAD catalog number filter.

        Args:
            norad_id: NORAD catalog number (e.g. ``25544``).
        """
        new = self._clone()
        new._catnr = norad_id
        return new

    def intdes(self, intdes: str) -> CelestrakQuery:
        """Set the international designator filter.

        Args:
            intdes: International designator (e.g. ``"1998-067A"``).
        """
        new = self._clone()
        new._intdes = intdes
        return new

    def name_search(self, name: str) -> CelestrakQuery:
        """Set the satellite name search filter.

        Args:
            name: Satellite name to search for.
        """
        new = self._clone()
        new._name = name
        return new

    def special(self, special: str) -> CelestrakQuery:
        """Set the special query parameter.

        Args:
            special: Special query parameter value.
        """
        new = self._clone()
        new._special = special
        return new

    def source(self, source: SupGPSource) -> CelestrakQuery:
        """Set the supplemental GP data source.

        Args:
            source: Supplemental GP data source.
        """
        new = self._clone()
        new._source = source
        return new

    def file(self, file: str) -> CelestrakQuery:
        """Set the file parameter for supplemental GP queries.

        Args:
            file: File parameter value.
        """
        new = self._clone()
        new._file = file
        return new

    def payloads(self, enabled: bool) -> CelestrakQuery:
        """Filter to payloads only.

        Args:
            enabled: Whether to filter to payloads.
        """
        new = self._clone()
        new._payloads = enabled
        return new

    def on_orbit(self, enabled: bool) -> CelestrakQuery:
        """Filter to on-orbit objects only.

        Args:
            enabled: Whether to filter to on-orbit objects.
        """
        new = self._clone()
        new._on_orbit = enabled
        return new

    def active(self, enabled: bool) -> CelestrakQuery:
        """Filter to active objects only.

        Args:
            enabled: Whether to filter to active objects.
        """
        new = self._clone()
        new._active = enabled
        return new

    def max(self, count: int) -> CelestrakQuery:
        """Set the maximum number of results returned by the server.

        Args:
            count: Maximum result count.
        """
        new = self._clone()
        new._max_results = count
        return new

    def format(self, fmt: CelestrakOutputFormat) -> CelestrakQuery:
        """Set the output format for query results.

        Args:
            fmt: Output format.
        """
        new = self._clone()
        new._output_format = fmt
        return new

    # -- Client-side parameters --

    def filter(self, field: str, value: str) -> CelestrakQuery:
        """Add a client-side filter predicate.

        Args:
            field: Uppercase field name (e.g. ``"INCLINATION"``).
            value: Filter value with optional operator prefix
                (e.g. ``">50"``, ``"~~ISS"``).
        """
        new = self._clone()
        new._filters.append((field, value))
        return new

    def order_by(self, field: str, ascending: bool) -> CelestrakQuery:
        """Add a client-side ordering clause.

        Args:
            field: Uppercase field name to sort by.
            ascending: True for ascending, False for descending.
        """
        new = self._clone()
        new._order_by_clauses.append((field, ascending))
        return new

    def limit(self, count: int) -> CelestrakQuery:
        """Set a client-side limit on the number of results.

        Args:
            count: Maximum number of results.
        """
        new = self._clone()
        new._limit_count = count
        return new

    # -- Accessors --

    def query_type(self) -> CelestrakQueryType:
        """Return the query type for this query."""
        return self._query_type

    def output_format(self) -> CelestrakOutputFormat | None:
        """Return the output format, or None if not explicitly set."""
        return self._output_format

    def has_client_side_processing(self) -> bool:
        """Return True if this query has client-side filters/ordering/limit."""
        return bool(self._filters) or bool(self._order_by_clauses) or self._limit_count is not None

    def client_side_filters(self) -> list[tuple[str, str]]:
        """Return the client-side filters."""
        return self._filters

    def client_side_order_by(self) -> list[tuple[str, bool]]:
        """Return the client-side ordering clauses."""
        return self._order_by_clauses

    def client_side_limit(self) -> int | None:
        """Return the client-side limit, if set."""
        return self._limit_count

    # -- URL building --

    def build_url(self) -> str:
        """Build the URL query string for this query.

        Only includes server-side parameters; client-side filters
        are not included in the URL.

        Returns:
            URL query string (e.g. ``"GROUP=stations&FORMAT=JSON"``).
        """
        params: list[str] = []

        if self._group is not None:
            params.append(f"GROUP={self._group}")

        if self._catnr is not None:
            params.append(f"CATNR={self._catnr}")

        if self._intdes is not None:
            params.append(f"INTDES={self._intdes}")

        if self._name is not None:
            params.append(f"NAME={self._name}")

        if self._special is not None:
            params.append(f"SPECIAL={self._special}")

        if self._source is not None:
            params.append(f"SOURCE={self._source.as_str()}")

        if self._file is not None:
            params.append(f"FILE={self._file}")

        # Boolean flags: only include when True
        if self._payloads is True:
            params.append("PAYLOADS=Y")

        if self._on_orbit is True:
            params.append("ONORBIT=Y")

        if self._active is True:
            params.append("ACTIVE=Y")

        if self._max_results is not None:
            params.append(f"MAX={self._max_results}")

        if self._output_format is not None:
            params.append(f"FORMAT={self._output_format.as_str()}")

        return "&".join(params)

    def __str__(self) -> str:
        return self.build_url()

    def __repr__(self) -> str:
        return f'CelestrakQuery({_QUERY_TYPE_DEBUG[self._query_type]}, "{self.build_url()}")'
