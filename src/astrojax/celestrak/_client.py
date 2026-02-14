"""Celestrak HTTP client with caching.

Provides access to Celestrak endpoints with file-based caching
and typed query execution. No authentication is required.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from astrojax._gp_record import GPRecord
from astrojax.celestrak._filter import apply_filters, apply_limit, apply_order_by
from astrojax.celestrak._query import CelestrakQuery
from astrojax.celestrak._responses import CelestrakSATCATRecord
from astrojax.celestrak._types import (
    CelestrakOutputFormat,
    CelestrakQueryType,
    SupGPSource,
)
from astrojax.utils.caching import get_cache_dir

_DEFAULT_BASE_URL = "https://celestrak.org"
_DEFAULT_MAX_CACHE_AGE = 21600.0  # 6 hours


def _get_celestrak_cache_dir() -> Path:
    """Return the Celestrak cache directory."""
    return get_cache_dir("celestrak")


class CelestrakClient:
    """Celestrak API client with caching.

    Provides typed query execution for GP, supplemental GP, and SATCAT
    data from Celestrak. No authentication is required. Responses are
    cached locally to reduce server load.

    Args:
        base_url: Custom base URL for testing.
        cache_max_age: Cache TTL in seconds. Default: 21600.0 (6 hours).
    """

    def __init__(
        self,
        base_url: str | None = None,
        cache_max_age: float | None = None,
    ) -> None:
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._cache_max_age = (
            cache_max_age if cache_max_age is not None else _DEFAULT_MAX_CACHE_AGE
        )
        self._client = httpx.Client(timeout=120.0, follow_redirects=True)

    # ========================================
    # Tier 1: Compact convenience methods
    # ========================================

    def get_gp(
        self,
        *,
        catnr: int | None = None,
        group: str | None = None,
        name: str | None = None,
        intdes: str | None = None,
    ) -> list[GPRecord]:
        """Look up GP records by exactly one identifier.

        Args:
            catnr: NORAD catalog number.
            group: Satellite group name.
            name: Satellite name to search for.
            intdes: International designator.

        Returns:
            List of matching GP records.

        Raises:
            ValueError: If zero or more than one identifier is provided.
        """
        count = sum(x is not None for x in (catnr, group, name, intdes))
        if count != 1:
            raise ValueError(
                "Provide exactly one of: catnr, group, name, intdes"
            )

        if catnr is not None:
            query = CelestrakQuery.gp.catnr(catnr)
        elif group is not None:
            query = CelestrakQuery.gp.group(group)
        elif name is not None:
            query = CelestrakQuery.gp.name_search(name)
        else:
            query = CelestrakQuery.gp.intdes(intdes)  # type: ignore[arg-type]

        return self._query_gp(query)

    def get_sup_gp(self, source: SupGPSource) -> list[GPRecord]:
        """Look up supplemental GP records by source.

        Args:
            source: Supplemental GP data source.

        Returns:
            List of matching GP records.
        """
        query = CelestrakQuery.sup_gp.source(source)
        return self._query_gp(query)

    def get_satcat(
        self,
        *,
        catnr: int | None = None,
        active: bool | None = None,
        payloads: bool | None = None,
        on_orbit: bool | None = None,
    ) -> list[CelestrakSATCATRecord]:
        """Look up SATCAT records.

        At least one parameter must be provided.

        Args:
            catnr: NORAD catalog number.
            active: Filter to active objects.
            payloads: Filter to payloads.
            on_orbit: Filter to on-orbit objects.

        Returns:
            List of matching SATCAT records.

        Raises:
            ValueError: If no parameters are provided.
        """
        if catnr is None and active is None and payloads is None and on_orbit is None:
            raise ValueError(
                "Provide at least one of: catnr, active, payloads, on_orbit"
            )

        query: Any = CelestrakQuery.satcat
        if catnr is not None:
            query = query.catnr(catnr)
        if active is not None:
            query = query.active(active)
        if payloads is not None:
            query = query.payloads(payloads)
        if on_orbit is not None:
            query = query.on_orbit(on_orbit)

        return self._query_satcat(query)

    # ========================================
    # Tier 2: Query builder methods
    # ========================================

    def query(
        self, query: CelestrakQuery
    ) -> list[GPRecord] | list[CelestrakSATCATRecord]:
        """Execute a query and return typed results.

        Dispatches to the appropriate handler based on query type:
        GP and SupGP return ``list[GPRecord]``, SATCAT returns
        ``list[CelestrakSATCATRecord]``.

        Args:
            query: A CelestrakQuery instance.

        Returns:
            List of typed records.
        """
        qt = query.query_type()
        if qt in (CelestrakQueryType.GP, CelestrakQueryType.SUP_GP):
            return self._query_gp(query)
        else:
            return self._query_satcat(query)

    def query_raw(self, query: CelestrakQuery) -> str:
        """Execute a query and return the raw response body.

        Args:
            query: A CelestrakQuery instance.

        Returns:
            Raw response body as a string.
        """
        url = self._build_full_url(query)
        return self._fetch_with_cache(url)

    def download(self, query: CelestrakQuery, filepath: str) -> None:
        """Execute a query and save the response to a file.

        Args:
            query: A CelestrakQuery instance.
            filepath: Path to save the response to.
        """
        body = self.query_raw(query)
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    # ========================================
    # Internal query helpers
    # ========================================

    def _query_gp(self, query: CelestrakQuery) -> list[GPRecord]:
        """Execute a GP query and return typed GP records."""
        fmt = query.output_format()
        if fmt is None or not fmt.is_json():
            query = query.format(CelestrakOutputFormat.JSON)

        body = self.query_raw(query)
        records_json = json.loads(body)
        records = [GPRecord.from_json_dict(r) for r in records_json]

        records = apply_filters(records, query.client_side_filters())
        apply_order_by(records, query.client_side_order_by())
        records = apply_limit(records, query.client_side_limit())

        return records

    def _query_satcat(self, query: CelestrakQuery) -> list[CelestrakSATCATRecord]:
        """Execute a SATCAT query and return typed SATCAT records."""
        fmt = query.output_format()
        if fmt is None or not fmt.is_json():
            query = query.format(CelestrakOutputFormat.JSON)

        body = self.query_raw(query)
        records_json = json.loads(body)
        records = [CelestrakSATCATRecord.from_json_dict(r) for r in records_json]

        records = apply_filters(records, query.client_side_filters())
        apply_order_by(records, query.client_side_order_by())
        records = apply_limit(records, query.client_side_limit())

        return records

    # ========================================
    # URL building
    # ========================================

    def _build_full_url(self, query: CelestrakQuery) -> str:
        """Build the full URL for a query."""
        endpoint = query.query_type().endpoint_path()
        params = query.build_url()
        if params:
            return f"{self._base_url}{endpoint}?{params}"
        return f"{self._base_url}{endpoint}"

    # ========================================
    # Caching
    # ========================================

    def _fetch_with_cache(self, url: str) -> str:
        """Fetch a URL with file-based caching."""
        cache_key = self._cache_key_for_url(url)

        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        body = self._execute_get(url)
        self._write_cache(cache_key, body)

        return body

    def _cache_key_for_url(self, url: str) -> str:
        """Generate a cache key from a URL."""
        return "".join(
            c if c.isalnum() or c == "." else "_" for c in url
        )

    def _read_cache(self, cache_key: str) -> str | None:
        """Read cached data if it exists and is fresh."""
        cache_dir = _get_celestrak_cache_dir()
        cache_path = Path(cache_dir) / cache_key

        if not cache_path.exists():
            return None

        try:
            mtime = cache_path.stat().st_mtime
            age = time.time() - mtime
            if age > self._cache_max_age:
                return None
        except OSError:
            return None

        try:
            return cache_path.read_text()
        except OSError:
            return None

    def _write_cache(self, cache_key: str, data: str) -> None:
        """Write data to the cache."""
        try:
            cache_dir = _get_celestrak_cache_dir()
            cache_path = Path(cache_dir) / cache_key
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(data)
        except OSError:
            pass  # Cache write failures are non-fatal

    def _execute_get(self, url: str) -> str:
        """Execute an HTTP GET request and return the response body."""
        response = self._client.get(url)
        response.raise_for_status()
        return response.text
