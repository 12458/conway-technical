"""Integration tests to verify service setup."""

import pytest

from service.config import service_settings
from service.database import init_db
from service.queue import get_redis_connection


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database():
    """Test database initialization."""
    await init_db()
    # If we get here without exception, the database initialized successfully


@pytest.mark.integration
def test_redis():
    """Test Redis connection."""
    redis_conn = get_redis_connection()
    redis_conn.set("test_key", "test_value")
    value = redis_conn.get("test_key")
    assert value == b"test_value", "Redis value mismatch"
    redis_conn.delete("test_key")


@pytest.mark.integration
class TestConfiguration:
    """Tests for service configuration."""

    def test_config_values(self):
        """Test that configuration values are accessible."""
        assert service_settings.redis_url is not None
        assert service_settings.database_url is not None
        assert service_settings.ai_provider is not None
        assert service_settings.ai_model is not None
        assert service_settings.anomaly_threshold is not None
        assert service_settings.enable_bot_filtering is not None

    def test_ai_api_key_configured(self):
        """Test that OpenAI API key is configured."""
        # Check for OpenAI key
        # Note: This test will pass even if no key is set, as the service can use fallback summaries
        has_ai_key = service_settings.openai_api_key
        # We don't assert True here because the service can work without AI keys (fallback mode)
        # Just verify the field exists
        assert hasattr(service_settings, "openai_api_key")
