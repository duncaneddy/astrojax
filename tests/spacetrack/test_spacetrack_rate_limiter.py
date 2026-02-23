"""Tests for SpaceTrack rate limiter."""

from astrojax.spacetrack._rate_limiter import RateLimitConfig, RateLimiter


class TestRateLimitConfig:
    """Tests for RateLimitConfig configuration object."""

    def test_default_values(self):
        config = RateLimitConfig()
        assert config.max_per_minute == 25
        assert config.max_per_hour == 250

    def test_custom_values(self):
        config = RateLimitConfig(max_per_minute=10, max_per_hour=100)
        assert config.max_per_minute == 10
        assert config.max_per_hour == 100

    def test_keyword_arguments(self):
        config = RateLimitConfig(max_per_hour=50, max_per_minute=5)
        assert config.max_per_minute == 5
        assert config.max_per_hour == 50

    def test_disabled(self):
        config = RateLimitConfig.disabled()
        assert config.max_per_minute == 2**32 - 1
        assert config.max_per_hour == 2**32 - 1

    def test_str(self):
        config = RateLimitConfig()
        s = str(config)
        assert "25" in s
        assert "250" in s

    def test_repr(self):
        config = RateLimitConfig()
        r = repr(config)
        assert "RateLimitConfig" in r
        assert "25" in r
        assert "250" in r

    def test_equality(self):
        a = RateLimitConfig()
        b = RateLimitConfig()
        c = RateLimitConfig(max_per_minute=10, max_per_hour=100)
        assert a == b
        assert a != c

    def test_equality_disabled(self):
        a = RateLimitConfig.disabled()
        b = RateLimitConfig.disabled()
        assert a == b
        assert a != RateLimitConfig()


class TestRateLimiter:
    """Tests for RateLimiter sliding window."""

    def test_first_acquire_returns_zero(self):
        limiter = RateLimiter(RateLimitConfig())
        wait = limiter.acquire()
        assert wait == 0.0

    def test_under_limit_returns_zero(self):
        limiter = RateLimiter(RateLimitConfig(max_per_minute=5, max_per_hour=100))
        for _ in range(4):
            wait = limiter.acquire()
            assert wait == 0.0

    def test_at_minute_limit_returns_positive(self):
        limiter = RateLimiter(RateLimitConfig(max_per_minute=3, max_per_hour=1000))
        for _ in range(3):
            limiter.acquire()
        wait = limiter.acquire()
        assert wait > 0.0

    def test_at_hour_limit_returns_positive(self):
        limiter = RateLimiter(RateLimitConfig(max_per_minute=1000, max_per_hour=3))
        for _ in range(3):
            limiter.acquire()
        wait = limiter.acquire()
        assert wait > 0.0

    def test_disabled_always_returns_zero(self):
        limiter = RateLimiter(RateLimitConfig.disabled())
        for _ in range(100):
            wait = limiter.acquire()
            assert wait == 0.0
