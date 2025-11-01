"""Background polling service for GitHub Events API."""

import asyncio
import logging
import random
from datetime import datetime

from sqlalchemy import select

from github_client import GitHubEventsClient
from github_client.exceptions import ForbiddenError, GitHubAPIError, ServiceUnavailableError
from service.anomaly_detector import detector
from service.config import service_settings
from service.database import AsyncSessionLocal, GitHubEvent
from service.queue import enqueue_anomaly_summarization

logger = logging.getLogger(__name__)


class GitHubEventsPoller:
    """Background service that polls GitHub Events API."""

    def __init__(self):
        """Initialize the poller."""
        self.running = False
        self.etag: str | None = None
        self.current_backoff = 0
        self.poll_count = 0
        self.anomaly_count = 0
        self.event_count = 0

    async def start(self) -> None:
        """Start the polling service."""
        logger.info("Starting GitHub Events polling service")
        self.running = True

        # Build headers with GitHub token if available
        headers = {}
        if service_settings.github_token:
            headers["Authorization"] = f"token {service_settings.github_token}"
            logger.info("Using GitHub token for authentication")

        async with GitHubEventsClient(headers=headers) as client:
            while self.running:
                try:
                    await self._poll_iteration(client)

                except ForbiddenError as e:
                    # Rate limit hit - exponential backoff
                    await self._handle_rate_limit()

                except ServiceUnavailableError as e:
                    # GitHub API unavailable - backoff and retry
                    logger.warning(f"GitHub API unavailable: {e.message}")
                    await self._backoff(60)

                except GitHubAPIError as e:
                    # Other API error - log and continue with backoff
                    logger.error(f"GitHub API error: {e.message} (status: {e.status_code})")
                    await self._backoff(30)

                except Exception as e:
                    # Unexpected error - log and continue
                    logger.exception(f"Unexpected error in polling loop: {e}")
                    await asyncio.sleep(10)

        logger.info("Polling service stopped")

    async def stop(self) -> None:
        """Stop the polling service."""
        logger.info("Stopping polling service")
        self.running = False

    async def _poll_iteration(self, client: GitHubEventsClient) -> None:
        """Execute one polling iteration.

        Args:
            client: GitHub Events client
        """
        logger.debug(f"Polling GitHub Events API (iteration {self.poll_count + 1})")

        # Fetch events with ETag caching
        response = await client.list_public_events(
            per_page=service_settings.max_events_per_fetch,
            etag=self.etag,
        )

        self.poll_count += 1

        # Check if we got new events
        if not response.events:
            logger.debug("No new events (304 Not Modified)")
            wait_time = response.poll_interval or service_settings.polling_interval
            await asyncio.sleep(wait_time)
            return

        # Update ETag for next request
        self.etag = response.etag

        logger.info(f"Fetched {len(response.events)} new events")

        # Process events
        await self._process_events(response.events)

        # Reset backoff on success
        self.current_backoff = 0

        # Wait for recommended interval
        wait_time = response.poll_interval or service_settings.polling_interval
        logger.debug(f"Waiting {wait_time}s before next poll")
        await asyncio.sleep(wait_time)

    async def _process_events(self, events: list) -> None:
        """Process fetched events through anomaly detection.

        Args:
            events: List of Event objects
        """
        async with AsyncSessionLocal() as session:
            for event in events:
                try:
                    # Check if event already exists
                    result = await session.execute(
                        select(GitHubEvent).where(GitHubEvent.id == event.id)
                    )
                    existing = result.scalar_one_or_none()

                    if existing:
                        logger.debug(f"Event {event.id} already processed, skipping")
                        continue

                    self.event_count += 1

                    # Run anomaly detection
                    score, patterns, features, velocity_score, is_inhuman_speed, velocity_reason = (
                        detector.process_event(event)
                    )

                    # Skip if event was filtered (e.g., bot)
                    if score is None:
                        continue

                    # Determine if anomaly (includes velocity-based detection)
                    is_anomaly = detector.is_anomaly(score, patterns, is_inhuman_speed)

                    # Create database record
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
                        is_anomaly=is_anomaly,
                        anomaly_score=float(score),
                    )

                    session.add(db_event)
                    await session.commit()

                    # If anomaly detected, enqueue for summarization
                    if is_anomaly:
                        self.anomaly_count += 1

                        logger.info(
                            f"🚨 ANOMALY DETECTED: {event.type} by {event.actor.login} "
                            f"on {event.repo.name} (score: {score:.2f}, patterns: {len(patterns)}, "
                            f"velocity: {velocity_score:.1f} events/min, inhuman: {is_inhuman_speed})"
                        )

                        if is_inhuman_speed:
                            logger.info(f"   ⚡ Velocity anomaly: {velocity_reason}")

                        # Enqueue job for AI summarization
                        # Use to_event_dict() to get Event-compatible format for enrichment
                        enqueue_anomaly_summarization(
                            event_id=event.id,
                            event_data=db_event.to_event_dict(),
                            anomaly_score=float(score),
                            suspicious_patterns=patterns,
                        )

                except Exception as e:
                    logger.error(f"Error processing event {event.id}: {e}")
                    await session.rollback()
                    continue

    async def _handle_rate_limit(self) -> None:
        """Handle rate limit with exponential backoff."""
        self.current_backoff = min(
            service_settings.max_backoff_seconds,
            max(60, self.current_backoff * service_settings.backoff_multiplier)
            if self.current_backoff > 0
            else 60,
        )

        # Add jitter to avoid thundering herd
        jitter = random.uniform(0, 0.1 * self.current_backoff)
        wait_time = self.current_backoff + jitter

        logger.warning(
            f"Rate limit exceeded. Backing off for {wait_time:.1f}s "
            f"(base: {self.current_backoff}s)"
        )

        await asyncio.sleep(wait_time)

    async def _backoff(self, base_seconds: int) -> None:
        """Execute backoff with jitter.

        Args:
            base_seconds: Base backoff time in seconds
        """
        jitter = random.uniform(0, 0.1 * base_seconds)
        wait_time = base_seconds + jitter
        logger.debug(f"Backing off for {wait_time:.1f}s")
        await asyncio.sleep(wait_time)

    def get_stats(self) -> dict:
        """Get poller statistics.

        Returns:
            Dictionary with poller stats
        """
        return {
            "running": self.running,
            "poll_count": self.poll_count,
            "events_processed": self.event_count,
            "anomalies_detected": self.anomaly_count,
            "current_backoff_seconds": self.current_backoff,
            "etag": self.etag,
        }


# Global poller instance
poller = GitHubEventsPoller()
