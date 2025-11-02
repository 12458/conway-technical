"""Integration test to verify event parsing from GitHub API."""

import pytest

from github_client import GitHubEventsClient
from github_client.models import Event
from service.database import GitHubEvent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parse_public_events():
    """Test parsing current GitHub events from the API."""
    async with GitHubEventsClient() as client:
        response = await client.list_public_events(per_page=5)

        # Verify we got events
        assert len(response.events) > 0, "Should have received at least one event"
        assert len(response.events) <= 5, "Should not exceed requested page size"

        for event in response.events:
            # Verify event has required fields
            assert event.id is not None, "Event should have an ID"
            assert event.actor is not None, "Event should have an actor"
            assert event.actor.id is not None, "Actor should have an ID"
            assert event.repo is not None, "Event should have a repo"
            assert event.repo.id is not None, "Repo should have an ID"
            assert event.created_at is not None, "Event should have a created_at timestamp"

            # Verify we can create a database record from the event
            db_event = GitHubEvent(
                id=event.id,
                event_type=event.type,
                actor_login=event.actor.login,
                actor_id=event.actor.id,
                repo_name=event.repo.name,
                repo_id=event.repo.id,
                org_login=event.org.login if event.org else None,
                created_at=event.created_at,
                payload=event.payload,
                processed=True,
                is_anomaly=False,
                anomaly_score=0.0,
            )

            # Verify the database event was created with correct types
            assert db_event.id == event.id
            assert db_event.event_type == event.type
            assert db_event.actor_id == event.actor.id
            assert db_event.repo_id == event.repo.id
