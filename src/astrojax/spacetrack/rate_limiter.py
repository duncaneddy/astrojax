"""Sliding-window rate limiter for SpaceTrack API requests.

Space-Track.org enforces rate limits of 30 requests per minute and
300 requests per hour. Default limits are set conservatively at ~83%
(25/min, 250/hour).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

_U32_MAX = 2**32 - 1


@dataclass
class RateLimitConfig:
    """Configuration for SpaceTrack API rate limiting.

    Args:
        max_per_minute: Maximum requests per rolling 60-second window.
        max_per_hour: Maximum requests per rolling 3600-second window.
    """

    max_per_minute: int = 25
    max_per_hour: int = 250

    @classmethod
    def disabled(cls) -> RateLimitConfig:
        """Create a configuration that disables rate limiting.

        Returns:
            A RateLimitConfig with effectively unlimited limits.
        """
        return cls(max_per_minute=_U32_MAX, max_per_hour=_U32_MAX)

    def __str__(self) -> str:
        return (
            f"RateLimitConfig(max_per_minute={self.max_per_minute}, "
            f"max_per_hour={self.max_per_hour})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class RateLimiter:
    """Sliding-window rate limiter that tracks request timestamps.

    Thread-safe via internal lock.

    Args:
        config: Rate limit configuration.
    """

    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        self._minute_window: deque[float] = deque()
        self._hour_window: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Acquire permission to make a request.

        Returns the number of seconds the caller must sleep before
        proceeding. Zero means proceed immediately.

        Returns:
            Wait time in seconds (0.0 means proceed immediately).
        """
        with self._lock:
            now = time.monotonic()

            # Prune expired entries
            minute_cutoff = now - 60.0
            while self._minute_window and self._minute_window[0] < minute_cutoff:
                self._minute_window.popleft()

            hour_cutoff = now - 3600.0
            while self._hour_window and self._hour_window[0] < hour_cutoff:
                self._hour_window.popleft()

            # Calculate required wait
            wait = 0.0

            if len(self._minute_window) >= self._config.max_per_minute:
                oldest = self._minute_window[0]
                minute_wait = (oldest + 60.0) - now
                if minute_wait > wait:
                    wait = minute_wait

            if len(self._hour_window) >= self._config.max_per_hour:
                oldest = self._hour_window[0]
                hour_wait = (oldest + 3600.0) - now
                if hour_wait > wait:
                    wait = hour_wait

            # Clamp negative waits to zero
            if wait < 0:
                wait = 0.0

            # Record future timestamp
            request_time = now + wait
            self._minute_window.append(request_time)
            self._hour_window.append(request_time)

            return wait
