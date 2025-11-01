"""Quick test script to verify service setup."""

import asyncio

from service.config import service_settings
from service.database import init_db
from service.queue import get_redis_connection


async def test_database():
    """Test database initialization."""
    print("\n=== Testing Database ===")
    try:
        await init_db()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


def test_redis():
    """Test Redis connection."""
    print("\n=== Testing Redis ===")
    try:
        redis_conn = get_redis_connection()
        redis_conn.set("test_key", "test_value")
        value = redis_conn.get("test_key")
        assert value == b"test_value", "Redis value mismatch"
        redis_conn.delete("test_key")
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False


def test_config():
    """Test configuration."""
    print("\n=== Testing Configuration ===")
    print(f"Redis URL: {service_settings.redis_url}")
    print(f"Database URL: {service_settings.database_url}")
    print(f"AI Provider: {service_settings.ai_provider}")
    print(f"AI Model: {service_settings.ai_model}")
    print(f"GitHub Token: {'✅ Set' if service_settings.github_token else '❌ Not set'}")
    print(f"Anthropic Key: {'✅ Set' if service_settings.anthropic_api_key else '❌ Not set'}")
    print(f"OpenAI Key: {'✅ Set' if service_settings.openai_api_key else '❌ Not set'}")
    print(f"Anomaly Threshold: {service_settings.anomaly_threshold}")
    print(f"Bot Filtering: {service_settings.enable_bot_filtering}")

    # Check for at least one AI key
    if service_settings.anthropic_api_key or service_settings.openai_api_key:
        print("✅ AI API key configured")
        return True
    else:
        print("⚠️  No AI API key set - will use fallback summaries")
        return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("GitHub Anomaly Monitoring Service - Setup Test")
    print("=" * 60)

    results = []

    # Test configuration
    results.append(test_config())

    # Test Redis
    results.append(test_redis())

    # Test database
    results.append(await test_database())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    if all(results):
        print("✅ All tests passed!")
        print("\nYou can now run the service with:")
        print("  python run_service.py")
    else:
        print("❌ Some tests failed. Please check the output above.")
        print("\nMake sure you have:")
        print("  1. Created a .env file (copy from .env.example)")
        print("  2. Set SERVICE_REDIS_URL in .env")
        print("  3. Set AI API key (SERVICE_ANTHROPIC_API_KEY or SERVICE_OPENAI_API_KEY)")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
