"""Integration tests for GitHub Events API client hitting real API.

These tests make real HTTP requests to GitHub's public API.
Mark them as slow and integration for selective test running.
"""

import asyncio

import pytest

from github_client import GitHubEventsClient
from github_client.models import Event


@pytest.mark.integration
@pytest.mark.slow
class TestRealGitHubAPI:
    """Integration tests with real GitHub API."""

    @pytest.mark.asyncio
    async def test_fetch_public_events(self):
        """Test fetching public events from real GitHub API."""
        async with GitHubEventsClient() as client:
            response = await client.list_public_events(per_page=10)

            # Should get some events (GitHub is very active)
            assert len(response.events) > 0
            assert len(response.events) <= 10

            # Verify events are properly validated Event objects
            for event in response.events:
                assert isinstance(event, Event)
                assert event.id is not None
                assert event.type is not None
                assert event.actor.login is not None
                assert event.repo.name is not None

    @pytest.mark.asyncio
    async def test_etag_caching_real_api(self):
        """Test ETag caching with real GitHub API."""
        async with GitHubEventsClient() as client:
            # First request
            response1 = await client.list_public_events(per_page=5)
            etag = response1.etag

            assert etag is not None
            assert len(response1.events) > 0

            # Second request with ETag - might get 304 or new events
            response2 = await client.list_public_events(per_page=5, etag=etag)

            # Either no new events (304) or new events with different ETag
            if len(response2.events) == 0:
                # 304 Not Modified
                assert response2.etag == etag
            else:
                # New events
                assert response2.etag is not None

    @pytest.mark.asyncio
    async def test_poll_interval_real_api(self):
        """Test poll interval parsing from real GitHub API."""
        async with GitHubEventsClient() as client:
            response = await client.list_public_events(per_page=5)

            # GitHub typically returns 60 seconds poll interval
            assert response.poll_interval is not None
            assert response.poll_interval > 0
            # Usually 60 seconds
            assert response.poll_interval == 60

    @pytest.mark.asyncio
    async def test_pagination_real_api(self):
        """Test pagination with real GitHub API."""
        async with GitHubEventsClient() as client:
            # Fetch first page
            page1 = await client.list_public_events(per_page=10, page=1)

            # Fetch second page
            page2 = await client.list_public_events(per_page=10, page=2)

            assert len(page1.events) > 0
            assert len(page2.events) > 0

            # Pages should have different events (usually)
            # Check by comparing first event IDs
            if len(page1.events) > 0 and len(page2.events) > 0:
                # Event IDs should be different between pages
                page1_ids = {e.id for e in page1.events}
                page2_ids = {e.id for e in page2.events}
                # Pages might overlap due to timing, but usually won't
                assert (
                    page1_ids != page2_ids
                    or len(page1_ids.intersection(page2_ids)) < 10
                )

    @pytest.mark.asyncio
    async def test_different_per_page_values(self):
        """Test different per_page parameter values."""
        async with GitHubEventsClient() as client:
            # Test small page
            response_small = await client.list_public_events(per_page=5)
            assert len(response_small.events) <= 5

            # Test medium page
            response_medium = await client.list_public_events(per_page=15)
            assert len(response_medium.events) <= 15

            # Test large page
            response_large = await client.list_public_events(per_page=30)
            assert len(response_large.events) <= 30

    @pytest.mark.asyncio
    async def test_event_types_from_real_api(self):
        """Test that real API returns various event types."""
        async with GitHubEventsClient() as client:
            response = await client.list_public_events(per_page=30)

            # Collect event types
            event_types = {event.type for event in response.events}

            # Should have at least a few different types
            assert len(event_types) > 1

            # Common event types that should appear
            common_types = {
                "PushEvent",
                "CreateEvent",
                "WatchEvent",
                "IssuesEvent",
                "PullRequestEvent",
            }

            # At least one common type should be present
            assert len(event_types.intersection(common_types)) > 0

    @pytest.mark.asyncio
    async def test_response_model_validation_real_data(self):
        """Test that real API data validates against our models."""
        async with GitHubEventsClient() as client:
            response = await client.list_public_events(per_page=30)

            for event in response.events:
                # All events should have required fields
                assert event.id
                assert event.type
                assert event.public is not None
                assert event.created_at

                # Actor fields
                assert event.actor.id
                assert event.actor.login
                assert event.actor.url
                assert event.actor.avatar_url

                # Repo fields
                assert event.repo.id
                assert event.repo.name
                assert event.repo.url

                # Payload should be a dict
                assert isinstance(event.payload, dict)

    @pytest.mark.asyncio
    async def test_client_reuse(self):
        """Test reusing client for multiple requests."""
        async with GitHubEventsClient() as client:
            # Make multiple requests with same client
            responses = []
            for _ in range(3):
                response = await client.list_public_events(per_page=5)
                responses.append(response)
                # Small delay to avoid rate limiting
                await asyncio.sleep(1)

            # All requests should succeed
            assert len(responses) == 3
            for response in responses:
                assert len(response.events) > 0

    @pytest.mark.asyncio
    async def test_max_per_page_limit(self):
        """Test requesting maximum per_page value."""
        async with GitHubEventsClient() as client:
            response = await client.list_public_events(per_page=100)

            # Should not exceed 100 events
            assert len(response.events) <= 100
            # GitHub is very active, should get close to 100
            assert len(response.events) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestRealAPIEdgeCases:
    """Edge case integration tests with real API."""

    @pytest.mark.asyncio
    async def test_rapid_polling_respects_poll_interval(self):
        """Test that we can handle rapid polling (respects rate limits)."""
        async with GitHubEventsClient() as client:
            etag = None

            # Poll a few times
            for i in range(3):
                response = await client.list_public_events(etag=etag)

                if response.poll_interval:
                    # In real usage, you should wait this long
                    # For testing, we just verify it's returned
                    assert response.poll_interval > 0

                etag = response.etag

                # Small delay to be respectful
                await asyncio.sleep(2)

    @pytest.mark.asyncio
    async def test_default_per_page(self):
        """Test using default per_page value."""
        async with GitHubEventsClient() as client:
            response = await client.list_public_events()

            # Should use default (30 from config)
            assert len(response.events) <= 30
            assert len(response.events) > 0
