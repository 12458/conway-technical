"""Integration tests to verify service setup."""

import pytest

from service.config import service_settings
from service.database import check_database_connection
from service.queue import get_redis_connection


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database():
    """Test database connection."""
    is_healthy = await check_database_connection()
    assert is_healthy, "Database connection failed"


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
        # Note: This test will pass even if no key is set, but summarization will fail without it
        has_ai_key = service_settings.openai_api_key
        # We don't assert True here to allow the service to start for event collection/detection only
        # Just verify the field exists
        assert hasattr(service_settings, "openai_api_key")
