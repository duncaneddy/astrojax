"""SpaceTrack HTTP client with authentication and query execution.

Handles session management via cookie-based authentication against
the Space-Track.org API.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from astrojax._gp_record import GPRecord
from astrojax.spacetrack._query import SpaceTrackQuery
from astrojax.spacetrack._rate_limiter import RateLimitConfig, RateLimiter
from astrojax.spacetrack._responses import SATCATRecord

_DEFAULT_BASE_URL = "https://www.space-track.org"


class SpaceTrackClient:
    """SpaceTrack API client with session-based authentication.

    Lazily authenticates on first query and re-authenticates on session expiry.

    Args:
        identity: Space-Track.org login email.
        password: Space-Track.org password.
        base_url: Custom base URL for testing.
        rate_limit: Rate limit configuration.
    """

    def __init__(
        self,
        identity: str,
        password: str,
        base_url: str | None = None,
        *,
        rate_limit: RateLimitConfig | None = None,
    ) -> None:
        self._identity = identity
        self._password = password
        self._base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.Client(timeout=120.0, follow_redirects=True)
        self._authenticated = False
        config = rate_limit if rate_limit is not None else RateLimitConfig()
        self._rate_limiter = RateLimiter(config)

    def _wait_for_rate_limit(self) -> None:
        """Wait for rate limit clearance before making a request."""
        wait = self._rate_limiter.acquire()
        if wait > 0:
            time.sleep(wait)

    def authenticate(self) -> None:
        """Explicitly authenticate with Space-Track.org.

        Called automatically on first query. Call explicitly to verify
        credentials early.

        Raises:
            RuntimeError: If authentication fails.
        """
        self._wait_for_rate_limit()

        url = f"{self._base_url}/ajaxauth/login"
        response = self._client.post(
            url,
            data={"identity": self._identity, "password": self._password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()

        body = response.text
        if '"Login":"Failed"' in body or '"Login": "Failed"' in body:
            raise RuntimeError(
                "SpaceTrack authentication failed: invalid credentials"
            )

        self._authenticated = True

    def _ensure_authenticated(self) -> None:
        """Ensure the client is authenticated, authenticating if necessary."""
        if not self._authenticated:
            self.authenticate()

    def _execute_get(self, url: str) -> str:
        """Execute an HTTP GET request and return the response body as text."""
        self._wait_for_rate_limit()
        response = self._client.get(url)
        response.raise_for_status()
        return response.text

    def _authenticated_get_string(self, url: str) -> str:
        """Execute an authenticated GET, re-authenticating on 401."""
        self._ensure_authenticated()
        try:
            return self._execute_get(url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                self.authenticate()
                return self._execute_get(url)
            raise

    # ========================================
    # Query methods
    # ========================================

    def query_raw(self, query: SpaceTrackQuery) -> str:
        """Execute a query and return the raw response body as a string.

        Args:
            query: A SpaceTrackQuery instance.

        Returns:
            Raw response body.
        """
        url = f"{self._base_url}{query.build()}"
        return self._authenticated_get_string(url)

    def query_json(self, query: SpaceTrackQuery) -> list[dict[str, Any]]:
        """Execute a query and return parsed JSON values.

        Args:
            query: A SpaceTrackQuery instance.

        Returns:
            List of JSON dictionaries.

        Raises:
            RuntimeError: If the query format is not JSON.
        """
        if not query.output_format().is_json():
            raise RuntimeError("query_json requires JSON output format")
        body = self.query_raw(query)
        return json.loads(body)

    def query_gp(self, query: SpaceTrackQuery) -> list[GPRecord]:
        """Execute a GP query and return typed GP records.

        Args:
            query: A SpaceTrackQuery instance.

        Returns:
            List of GP records.

        Raises:
            RuntimeError: If the query format is not JSON.
        """
        if not query.output_format().is_json():
            raise RuntimeError("query_gp requires JSON output format")
        body = self.query_raw(query)
        records_json = json.loads(body)
        return [GPRecord.from_json_dict(r) for r in records_json]

    def query_satcat(self, query: SpaceTrackQuery) -> list[SATCATRecord]:
        """Execute a SATCAT query and return typed SATCAT records.

        Args:
            query: A SpaceTrackQuery instance.

        Returns:
            List of SATCAT records.

        Raises:
            RuntimeError: If the query format is not JSON.
        """
        if not query.output_format().is_json():
            raise RuntimeError("query_satcat requires JSON output format")
        body = self.query_raw(query)
        records_json = json.loads(body)
        return [SATCATRecord.from_json_dict(r) for r in records_json]
