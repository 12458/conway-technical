"""Enrichment service orchestrator for detected anomalies."""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from github_client.enriched_models import (
    ActorProfile,
    CommitVerification,
    EnrichedEvent,
    RepositoryContext,
    WorkflowStatus,
)
from github_client.enrichment_cache import EnrichmentCacheManager
from github_client.graphql_client import GitHubGraphQLClient
from github_client.models import Event

logger = logging.getLogger(__name__)


class EnrichmentService:
    """Service for enriching detected anomalies with GraphQL data."""

    def __init__(
        self,
        graphql_client: GitHubGraphQLClient,
        db_session: AsyncSession,
        enabled: bool = True,
    ):
        """Initialize enrichment service.

        Args:
            graphql_client: GitHub GraphQL API client
            db_session: Database session for caching
            enabled: Whether enrichment is enabled (feature flag)
        """
        self.graphql_client = graphql_client
        self.cache_manager = EnrichmentCacheManager(db_session)
        self.enabled = enabled
        self._enrichment_count = 0
        self._error_count = 0

    async def enrich_anomaly(
        self, event: Event, anomaly_score: float, suspicious_patterns: list[str]
    ) -> EnrichedEvent:
        """Enrich a detected anomaly with GraphQL data.

        Args:
            event: GitHub event that was flagged as anomaly
            anomaly_score: Anomaly detection score
            suspicious_patterns: List of detected suspicious patterns

        Returns:
            EnrichedEvent with all available enrichment data
        """
        if not self.enabled:
            logger.debug("Enrichment disabled, returning base event")
            return EnrichedEvent(
                event=event,
                anomaly_score=anomaly_score,
                suspicious_patterns=suspicious_patterns,
            )

        logger.info(
            f"Enriching anomaly: {event.type} by {event.actor.login} "
            f"on {event.repo.name} (score: {anomaly_score:.2f})"
        )

        enriched = EnrichedEvent(
            event=event,
            anomaly_score=anomaly_score,
            suspicious_patterns=suspicious_patterns,
        )

        # Enrich actor profile (always relevant)
        try:
            enriched.actor_profile = await self._enrich_actor_profile(event.actor.login)
        except Exception as e:
            logger.error(f"Failed to enrich actor profile: {e}")
            self._error_count += 1

        # Enrich repository context (always relevant)
        try:
            owner, name = event.repo.name.split("/", 1)
            enriched.repository_context = await self._enrich_repository_context(
                owner, name
            )
        except Exception as e:
            logger.error(f"Failed to enrich repository context: {e}")
            self._error_count += 1

        # Event-specific enrichment
        if event.type == "PushEvent":
            try:
                await self._enrich_push_event(event, enriched)
            except Exception as e:
                logger.error(f"Failed to enrich PushEvent: {e}")
                self._error_count += 1

        elif event.type == "PullRequestEvent":
            try:
                await self._enrich_pull_request_event(event, enriched)
            except Exception as e:
                logger.error(f"Failed to enrich PullRequestEvent: {e}")
                self._error_count += 1

        self._enrichment_count += 1
        logger.debug(
            f"Enrichment complete for {event.id}, score: {enriched.anomaly_score:.2f}"
        )

        return enriched

    async def _enrich_actor_profile(self, login: str) -> ActorProfile | None:
        """Enrich actor profile with cache-first strategy.

        Args:
            login: GitHub username

        Returns:
            ActorProfile or None on error
        """
        # Check cache first
        cached = await self.cache_manager.get_actor_profile(login)
        if cached:
            logger.debug(f"Actor profile cache hit for {login}")
            return cached

        # Cache miss, fetch from GraphQL
        logger.debug(f"Actor profile cache miss for {login}, fetching from GraphQL")
        profile = await self.graphql_client.get_actor_profile(login)

        # Store in cache if successful
        if profile:
            await self.cache_manager.set_actor_profile(profile)

        return profile

    async def _enrich_repository_context(
        self, owner: str, name: str
    ) -> RepositoryContext | None:
        """Enrich repository context with cache-first strategy.

        Args:
            owner: Repository owner
            name: Repository name

        Returns:
            RepositoryContext or None on error
        """
        # Check cache first
        cached = await self.cache_manager.get_repository_context(owner, name)
        if cached:
            logger.debug(f"Repository context cache hit for {owner}/{name}")
            return cached

        # Cache miss, fetch from GraphQL
        logger.debug(
            f"Repository context cache miss for {owner}/{name}, fetching from GraphQL"
        )
        context = await self.graphql_client.get_repository_context(owner, name)

        # Store in cache if successful
        if context:
            await self.cache_manager.set_repository_context(context)

        return context

    async def _enrich_push_event(self, event: Event, enriched: EnrichedEvent) -> None:
        """Enrich PushEvent with workflow status and commit verification.

        Args:
            event: GitHub PushEvent
            enriched: EnrichedEvent to populate
        """
        payload = event.payload
        owner, name = event.repo.name.split("/", 1)

        # Get the head commit SHA
        head_sha = payload.get("head")
        if not head_sha:
            logger.warning(f"No head SHA in PushEvent {event.id}")
            return

        # Enrich workflow status
        try:
            enriched.workflow_status = await self._enrich_workflow_status(
                owner, name, head_sha
            )
        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")

        # Enrich commit verification (for head commit)
        try:
            enriched.commit_verification = await self._enrich_commit_verification(
                owner, name, head_sha
            )
        except Exception as e:
            logger.error(f"Failed to get commit verification: {e}")

    async def _enrich_pull_request_event(
        self, event: Event, enriched: EnrichedEvent
    ) -> None:
        """Enrich PullRequestEvent with workflow status.

        Args:
            event: GitHub PullRequestEvent
            enriched: EnrichedEvent to populate
        """
        payload = event.payload
        owner, name = event.repo.name.split("/", 1)

        # Get pull request head SHA
        pr = payload.get("pull_request", {})
        head_sha = pr.get("head", {}).get("sha")
        if not head_sha:
            logger.warning(f"No head SHA in PullRequestEvent {event.id}")
            return

        # Enrich workflow status
        try:
            enriched.workflow_status = await self._enrich_workflow_status(
                owner, name, head_sha
            )
        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")

    async def _enrich_workflow_status(
        self, owner: str, name: str, commit_sha: str
    ) -> WorkflowStatus | None:
        """Enrich workflow/CI status with cache-first strategy.

        Args:
            owner: Repository owner
            name: Repository name
            commit_sha: Commit SHA

        Returns:
            WorkflowStatus or None on error
        """
        repository = f"{owner}/{name}"

        # Check cache first
        cached = await self.cache_manager.get_workflow_status(repository, commit_sha)
        if cached:
            logger.debug(f"Workflow status cache hit for {repository}@{commit_sha}")
            return cached

        # Cache miss, fetch from GraphQL
        logger.debug(
            f"Workflow status cache miss for {repository}@{commit_sha}, "
            f"fetching from GraphQL"
        )
        status = await self.graphql_client.get_workflow_status(owner, name, commit_sha)

        # Store in cache if successful
        if status:
            await self.cache_manager.set_workflow_status(status)

        return status

    async def _enrich_commit_verification(
        self, owner: str, name: str, sha: str
    ) -> CommitVerification | None:
        """Enrich commit verification with cache-first strategy.

        Args:
            owner: Repository owner
            name: Repository name
            sha: Commit SHA

        Returns:
            CommitVerification or None on error
        """
        repository = f"{owner}/{name}"

        # Check cache first (commits are immutable, so cached forever)
        cached = await self.cache_manager.get_commit_verification(repository, sha)
        if cached:
            logger.debug(f"Commit verification cache hit for {repository}@{sha}")
            return cached

        # Cache miss, fetch from GraphQL
        logger.debug(
            f"Commit verification cache miss for {repository}@{sha}, "
            f"fetching from GraphQL"
        )
        verification = await self.graphql_client.get_commit_verification(
            owner, name, sha
        )

        # Store in cache if successful
        if verification:
            await self.cache_manager.set_commit_verification(verification)

        return verification

    async def cleanup_cache(self) -> int:
        """Clean up expired cache entries.

        Returns:
            Number of entries removed
        """
        logger.info("Running cache cleanup")
        count = await self.cache_manager.cleanup_expired()
        logger.info(f"Cache cleanup complete, removed {count} entries")
        return count

    @property
    def stats(self) -> dict[str, Any]:
        """Get enrichment service statistics."""
        cache_stats = self.cache_manager.cache_stats
        return {
            "enabled": self.enabled,
            "enrichment_count": self._enrichment_count,
            "error_count": self._error_count,
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
            "cache_hit_rate": cache_stats["hit_rate"],
            "graphql_rate_limit": self.graphql_client.rate_limit_info,
        }
