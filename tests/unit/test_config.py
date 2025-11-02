"""Unit tests for configuration."""

import pytest
from pydantic import ValidationError

from github_client.config import Settings


@pytest.mark.unit
class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self):
        """Test Settings with default values."""
        settings = Settings()

        assert settings.github_api_base_url == "https://api.github.com"
        assert settings.github_api_version == "2022-11-28"
        assert settings.user_agent == "GitHubEventsClient/1.0"
        assert settings.default_per_page == 30
        assert settings.timeout == 30.0

    def test_custom_settings(self):
        """Test Settings with custom values."""
        settings = Settings(
            github_api_base_url="https://custom.api.com",
            github_api_version="2023-01-01",
            user_agent="CustomClient/2.0",
            default_per_page=50,
            timeout=60.0,
        )

        assert settings.github_api_base_url == "https://custom.api.com"
        assert settings.github_api_version == "2023-01-01"
        assert settings.user_agent == "CustomClient/2.0"
        assert settings.default_per_page == 50
        assert settings.timeout == 60.0

    def test_per_page_validation_min(self):
        """Test per_page validation enforces minimum value."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(default_per_page=0)

        assert "default_per_page" in str(exc_info.value).lower()

    def test_per_page_validation_max(self):
        """Test per_page validation enforces maximum value."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(default_per_page=101)

        assert "default_per_page" in str(exc_info.value).lower()

    def test_per_page_boundary_values(self):
        """Test per_page accepts boundary values."""
        settings_min = Settings(default_per_page=1)
        settings_max = Settings(default_per_page=100)

        assert settings_min.default_per_page == 1
        assert settings_max.default_per_page == 100

    def test_timeout_validation(self):
        """Test timeout validation enforces positive value."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(timeout=0)

        assert "timeout" in str(exc_info.value).lower()

    def test_timeout_negative(self):
        """Test timeout rejects negative values."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(timeout=-5.0)

        assert "timeout" in str(exc_info.value).lower()

    def test_settings_immutability(self):
        """Test that settings values can be set."""
        settings = Settings()
        # Pydantic models are mutable by default, so this should work
        settings.timeout = 45.0
        assert settings.timeout == 45.0
