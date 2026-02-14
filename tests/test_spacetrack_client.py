"""Tests for SpaceTrackClient construction (no network)."""

import pytest

from astrojax.spacetrack import (
    OutputFormat,
    RateLimitConfig,
    RequestClass,
    SATCATRecord,
    SpaceTrackClient,
    SpaceTrackQuery,
)


class TestSpaceTrackClient:
    """Tests for SpaceTrackClient construction."""

    def test_client_creation(self):
        client = SpaceTrackClient("user@example.com", "password")
        assert client is not None

    def test_client_with_base_url(self):
        client = SpaceTrackClient(
            "user@example.com", "password", "https://test.space-track.org"
        )
        assert client is not None

    def test_client_with_rate_limit(self):
        config = RateLimitConfig(max_per_minute=10, max_per_hour=100)
        client = SpaceTrackClient("user@example.com", "password", rate_limit=config)
        assert client is not None

    def test_client_with_base_url_and_rate_limit(self):
        config = RateLimitConfig(max_per_minute=10, max_per_hour=100)
        client = SpaceTrackClient(
            "user@example.com",
            "password",
            "https://test.space-track.org",
            rate_limit=config,
        )
        assert client is not None

    def test_client_with_disabled_rate_limit(self):
        config = RateLimitConfig.disabled()
        client = SpaceTrackClient("user@example.com", "password", rate_limit=config)
        assert client is not None

    def test_client_default_rate_limit(self):
        """Client created without rate_limit uses defaults."""
        config = RateLimitConfig()
        client = SpaceTrackClient("user@example.com", "password", rate_limit=config)
        assert client is not None


class TestSpaceTrackClientMethods:
    """Tests for SpaceTrack client method availability."""

    def test_query_methods_exist(self):
        client = SpaceTrackClient("user@example.com", "password")
        assert hasattr(client, "authenticate")
        assert hasattr(client, "query_raw")
        assert hasattr(client, "query_json")
        assert hasattr(client, "query_gp")
        assert hasattr(client, "query_satcat")

    def test_query_json_requires_json_format(self):
        """query_json should raise if format is not JSON."""
        client = SpaceTrackClient("user@example.com", "password")
        query = SpaceTrackQuery(RequestClass.GP).format(OutputFormat.TLE)
        with pytest.raises(RuntimeError, match="JSON"):
            client.query_json(query)

    def test_query_gp_requires_json_format(self):
        """query_gp should raise if format is not JSON."""
        client = SpaceTrackClient("user@example.com", "password")
        query = SpaceTrackQuery(RequestClass.GP).format(OutputFormat.CSV)
        with pytest.raises(RuntimeError, match="JSON"):
            client.query_gp(query)

    def test_query_satcat_requires_json_format(self):
        """query_satcat should raise if format is not JSON."""
        client = SpaceTrackClient("user@example.com", "password")
        query = SpaceTrackQuery(RequestClass.SATCAT).format(OutputFormat.TLE)
        with pytest.raises(RuntimeError, match="JSON"):
            client.query_satcat(query)


class TestSATCATRecord:
    """Tests for SATCATRecord response type."""

    def test_from_json_dict(self):
        d = {
            "INTLDES": "98067A",
            "NORAD_CAT_ID": "25544",
            "OBJECT_TYPE": "PAY",
            "SATNAME": "ISS (ZARYA)",
            "COUNTRY": "ISS",
            "LAUNCH": "1998-11-20",
            "SITE": "TYMSC",
            "DECAY": None,
            "PERIOD": "92.87",
            "INCLINATION": "51.64",
            "APOGEE": "420",
            "PERIGEE": "418",
        }
        record = SATCATRecord.from_json_dict(d)
        assert record.intldes == "98067A"
        assert record.norad_cat_id == 25544
        assert record.satname == "ISS (ZARYA)"
        assert record.inclination == "51.64"

    def test_str_repr(self):
        record = SATCATRecord(satname="ISS", norad_cat_id=25544)
        s = str(record)
        assert "ISS" in s
        assert "25544" in s
