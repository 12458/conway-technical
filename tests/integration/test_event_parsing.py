#!/usr/bin/env python3
"""Test script to reproduce the event parsing error."""

import asyncio
import json
from github_client import GitHubEventsClient
from github_client.models import Event

async def test_parse():
    """Test parsing current GitHub events."""
    async with GitHubEventsClient() as client:
        response = await client.list_public_events(per_page=5)

        print(f"Fetched {len(response.events)} events\n")

        for event in response.events:
            print(f"Event ID: {event.id} (type: {type(event.id).__name__})")
            print(f"  Actor ID: {event.actor.id} (type: {type(event.actor.id).__name__})")
            print(f"  Repo ID: {event.repo.id} (type: {type(event.repo.id).__name__})")
            print(f"  Created at: {event.created_at} (type: {type(event.created_at).__name__})")
            print()

            # Try to create a database record to see if error occurs
            try:
                from service.database import GitHubEvent
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
                print(f"✓ Successfully created GitHubEvent for {event.id}")
            except Exception as e:
                print(f"✗ Error creating GitHubEvent for {event.id}: {e}")
                print(f"  Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
            print()

if __name__ == "__main__":
    asyncio.run(test_parse())
