"""Tests for EOP caching and download functionality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from astrojax.eop import (
    EOPData,
    download_standard_eop_file,
    get_ut1_utc,
    load_cached_eop,
)
from astrojax.eop._download import IERS_STANDARD_URL

# ---------------------------------------------------------------------------
# download_standard_eop_file tests
# ---------------------------------------------------------------------------


class TestDownloadStandardEOPFile:
    """Tests for download_standard_eop_file."""

    @pytest.mark.ci
    def test_download_success(self, tmp_path: Path) -> None:
        """Actual download from IERS produces a non-empty file."""
        dest = tmp_path / "finals.all.iau2000.txt"
        result = download_standard_eop_file(dest)
        assert result.exists()
        assert result.stat().st_size > 0

    @pytest.mark.ci
    def test_download_content_looks_like_eop(self, tmp_path: Path) -> None:
        """Downloaded content contains lines that look like EOP data."""
        dest = tmp_path / "finals.all.iau2000.txt"
        download_standard_eop_file(dest)
        text = dest.read_text(encoding="utf-8")
        lines = text.strip().splitlines()
        assert len(lines) > 100
        # First data line should have an MJD field around column 7-15
        first_line = lines[0]
        assert len(first_line) >= 80

    def test_download_bad_url_raises(self, tmp_path: Path) -> None:
        """A bad URL raises an httpx error."""
        import httpx

        dest = tmp_path / "bad.txt"
        with pytest.raises((httpx.HTTPStatusError, httpx.TransportError)):
            download_standard_eop_file(
                dest,
                url="https://datacenter.iers.org/data/nonexistent_file.txt",
                timeout=10.0,
            )

    def test_download_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they do not exist."""
        dest = tmp_path / "deep" / "nested" / "dir" / "finals.txt"
        assert not dest.parent.exists()

        # Use mock to avoid actual network call -- we only test dir creation
        with patch("astrojax.eop._download.httpx.Client") as mock_client_cls:
            mock_response = mock_client_cls.return_value.__enter__.return_value.get.return_value
            mock_response.text = "mock eop data\n"
            mock_response.raise_for_status.return_value = None

            download_standard_eop_file(dest)

        assert dest.parent.exists()
        assert dest.exists()

    def test_download_default_url(self) -> None:
        """Default URL points to IERS data centre."""
        assert "iers.org" in IERS_STANDARD_URL
        assert "finals.all.iau2000.txt" in IERS_STANDARD_URL


# ---------------------------------------------------------------------------
# load_cached_eop tests
# ---------------------------------------------------------------------------


class TestLoadCachedEOP:
    """Tests for load_cached_eop."""

    def test_fresh_file_reused(self, tmp_path: Path) -> None:
        """A fresh cached file is loaded without downloading."""
        import importlib.resources

        # Copy bundled file to tmp as a "cached" file
        data_pkg = importlib.resources.files("astrojax.data.eop")
        resource = data_pkg.joinpath("finals.all.iau2000.txt")
        with importlib.resources.as_file(resource) as src:
            dest = tmp_path / "finals.all.iau2000.txt"
            dest.write_bytes(src.read_bytes())

        with patch("astrojax.eop._providers.download_standard_eop_file") as mock_dl:
            eop = load_cached_eop(dest, max_age_days=7.0)
            mock_dl.assert_not_called()

        assert isinstance(eop, EOPData)
        assert eop.mjd.shape[0] > 1000

    def test_stale_file_triggers_download(self, tmp_path: Path) -> None:
        """A stale file triggers a download attempt."""
        import importlib.resources
        import os

        # Copy bundled file with an old modification time
        data_pkg = importlib.resources.files("astrojax.data.eop")
        resource = data_pkg.joinpath("finals.all.iau2000.txt")
        with importlib.resources.as_file(resource) as src:
            dest = tmp_path / "finals.all.iau2000.txt"
            dest.write_bytes(src.read_bytes())

        # Set mtime to 30 days ago
        old_time = dest.stat().st_mtime - 30 * 86400
        os.utime(dest, (old_time, old_time))

        with patch("astrojax.eop._providers.download_standard_eop_file") as mock_dl:
            # Make the download a no-op (file is already valid from copy)
            mock_dl.return_value = dest
            eop = load_cached_eop(dest, max_age_days=7.0)
            mock_dl.assert_called_once_with(dest)

        assert isinstance(eop, EOPData)

    def test_missing_file_triggers_download(self, tmp_path: Path) -> None:
        """A missing file triggers a download attempt."""
        dest = tmp_path / "finals.all.iau2000.txt"
        assert not dest.exists()

        with patch("astrojax.eop._providers.download_standard_eop_file") as mock_dl:
            # Simulate download creating a valid file
            import importlib.resources

            data_pkg = importlib.resources.files("astrojax.data.eop")
            resource = data_pkg.joinpath("finals.all.iau2000.txt")

            def fake_download(fp: Path) -> Path:
                with importlib.resources.as_file(resource) as src:
                    fp.write_bytes(src.read_bytes())
                return fp

            mock_dl.side_effect = fake_download
            eop = load_cached_eop(dest)
            mock_dl.assert_called_once()

        assert isinstance(eop, EOPData)
        assert eop.mjd.shape[0] > 1000

    def test_download_failure_falls_back_to_bundled(self, tmp_path: Path) -> None:
        """If download fails, bundled default data is returned."""
        dest = tmp_path / "finals.all.iau2000.txt"

        with patch(
            "astrojax.eop._providers.download_standard_eop_file",
            side_effect=RuntimeError("network error"),
        ):
            eop = load_cached_eop(dest)

        assert isinstance(eop, EOPData)
        assert eop.mjd.shape[0] > 1000

    def test_corrupt_file_falls_back_to_bundled(self, tmp_path: Path) -> None:
        """A corrupt cached file falls back to bundled data."""
        dest = tmp_path / "finals.all.iau2000.txt"
        dest.write_text("this is not valid EOP data\n" * 10, encoding="utf-8")

        with patch("astrojax.eop._providers.download_standard_eop_file") as mock_dl:
            # Download "succeeds" but file is still corrupt
            mock_dl.return_value = dest
            eop = load_cached_eop(dest, max_age_days=0.0)

        assert isinstance(eop, EOPData)
        # Should be bundled data
        assert eop.mjd.shape[0] > 1000

    def test_default_filepath(self) -> None:
        """When filepath is None, the default cache location is used."""
        from astrojax.utils.caching import get_eop_cache_dir

        expected_dir = get_eop_cache_dir()
        expected_file = expected_dir / "finals.all.iau2000.txt"

        with (
            patch("astrojax.eop._providers.is_file_stale", return_value=False) as mock_stale,
            patch("astrojax.eop._providers.load_eop_from_file") as mock_load,
            patch("astrojax.eop._providers.get_eop_cache_dir", return_value=expected_dir),
        ):
            from astrojax.eop._types import EOPData as _EOPData

            mock_load.return_value = _EOPData(
                mjd=None, pm_x=None, pm_y=None, ut1_utc=None,
                dX=None, dY=None, lod=None,
                mjd_min=None, mjd_max=None,
                mjd_last_lod=None, mjd_last_dxdy=None,
            )

            load_cached_eop()

            # Verify the correct path was checked for staleness
            mock_stale.assert_called_once()
            actual_path = mock_stale.call_args[0][0]
            assert Path(actual_path) == expected_file

    def test_returns_valid_eop_data(self, tmp_path: Path) -> None:
        """Returned EOPData is queryable."""
        import importlib.resources

        data_pkg = importlib.resources.files("astrojax.data.eop")
        resource = data_pkg.joinpath("finals.all.iau2000.txt")
        with importlib.resources.as_file(resource) as src:
            dest = tmp_path / "finals.all.iau2000.txt"
            dest.write_bytes(src.read_bytes())

        eop = load_cached_eop(dest)
        val = get_ut1_utc(eop, 59569.0)
        assert float(val) != 0.0  # Should have real EOP data

    def test_custom_max_age(self, tmp_path: Path) -> None:
        """Custom max_age_days is respected."""
        import importlib.resources
        import os

        data_pkg = importlib.resources.files("astrojax.data.eop")
        resource = data_pkg.joinpath("finals.all.iau2000.txt")
        with importlib.resources.as_file(resource) as src:
            dest = tmp_path / "finals.all.iau2000.txt"
            dest.write_bytes(src.read_bytes())

        # Set mtime to 2 days ago
        old_time = dest.stat().st_mtime - 2 * 86400
        os.utime(dest, (old_time, old_time))

        # With max_age_days=3, file should be fresh (no download)
        with patch("astrojax.eop._providers.download_standard_eop_file") as mock_dl:
            eop = load_cached_eop(dest, max_age_days=3.0)
            mock_dl.assert_not_called()
        assert isinstance(eop, EOPData)

        # With max_age_days=1, file should be stale (triggers download)
        with patch("astrojax.eop._providers.download_standard_eop_file") as mock_dl:
            mock_dl.return_value = dest
            eop = load_cached_eop(dest, max_age_days=1.0)
            mock_dl.assert_called_once()
        assert isinstance(eop, EOPData)
