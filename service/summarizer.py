"""AI-powered summarization worker for anomalies with mandatory enrichment."""

import logging
from typing import Any

from github_client.graphql_client import GitHubGraphQLClient
from github_client.models import Event
from service.config import service_settings
from service.database import AsyncSessionLocal
from service.enrichment_service import EnrichmentService
from service.enhanced_summarizer import summarize_enriched_anomaly

logger = logging.getLogger(__name__)


async def summarize_anomaly_job(
    event_id: str,
    event_data: dict[str, Any],
    anomaly_score: float,
    suspicious_patterns: list[str],
) -> dict[str, Any]:
    """RQ job to generate AI summary for an anomaly with GraphQL enrichment.

    This worker function always uses enrichment with GraphQL data.
    Requires SERVICE_GITHUB_GRAPHQL_TOKEN to be configured.

    Args:
        event_id: Database event ID
        event_data: Full event data dictionary
        anomaly_score: CoDisp anomaly score
        suspicious_patterns: List of detected patterns

    Returns:
        Summary data dictionary

    Raises:
        ValueError: If GraphQL token is not configured
    """
    logger.info(f"Starting enriched summarization for event {event_id}")

    # Validate GraphQL token is configured
    if not service_settings.github_graphql_token:
        error_msg = (
            "GraphQL token is required for summarization. "
            "Set SERVICE_GITHUB_GRAPHQL_TOKEN environment variable."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Convert event_data dict to Event model
        event = Event(**event_data)

        # Initialize GraphQL client and enrichment service
        async with AsyncSessionLocal() as db_session:
            async with GitHubGraphQLClient(
                token=service_settings.github_graphql_token,
                base_url=service_settings.graphql_api_url,
            ) as graphql_client:
                enrichment_service = EnrichmentService(
                    graphql_client=graphql_client,
                    db_session=db_session,
                    enabled=True,
                )

                # Enrich the anomaly
                enriched_event = await enrichment_service.enrich_anomaly(
                    event=event,
                    anomaly_score=anomaly_score,
                    suspicious_patterns=suspicious_patterns,
                )

                logger.info(f"Enrichment complete for {event_id}")

                # Use enhanced summarization
                result = await summarize_enriched_anomaly(enriched_event)

                logger.info(f"Enrichment stats: {enrichment_service.stats}")

                return result

    except Exception as e:
        logger.exception(f"Error summarizing event {event_id}: {e}")
        raise
