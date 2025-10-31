"""AI-powered summarization worker for anomalies."""

import json
import logging
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from service.config import service_settings
from service.database import AsyncSessionLocal, AnomalySummary

logger = logging.getLogger(__name__)


class SummaryResponse(BaseModel):
    """Pydantic model for structured AI summary output."""

    title: str
    severity: str
    root_cause: list[str]
    impact: list[str]
    next_steps: list[str]
    tags: list[str]


async def summarize_anomaly_job(
    event_id: str,
    event_data: dict[str, Any],
    anomaly_score: float,
    suspicious_patterns: list[str],
) -> dict[str, Any]:
    """RQ job to generate AI summary for an anomaly.

    This is the worker function executed by RQ workers.

    Args:
        event_id: Database event ID
        event_data: Full event data dictionary
        anomaly_score: CoDisp anomaly score
        suspicious_patterns: List of detected patterns

    Returns:
        Summary data dictionary
    """
    logger.info(f"Starting summarization for event {event_id}")

    try:
        # Generate AI summary
        summary_data = await generate_summary(
            event_data=event_data,
            anomaly_score=anomaly_score,
            suspicious_patterns=suspicious_patterns,
        )

        # Save to database
        async with AsyncSessionLocal() as session:
            db_summary = AnomalySummary(
                event_id=event_id,
                title=summary_data["title"],
                severity=summary_data["severity"],
                root_cause=summary_data["root_cause"],
                impact=summary_data["impact"],
                next_steps=summary_data["next_steps"],
                suspicious_patterns=suspicious_patterns,
                anomaly_score=anomaly_score,
                event_type=event_data["event_type"],
                actor_login=event_data["actor_login"],
                repo_name=event_data["repo_name"],
                raw_event=event_data,
                event_timestamp=datetime.fromisoformat(event_data["created_at"]),
                tags=summary_data.get("tags", []),
            )

            session.add(db_summary)
            await session.commit()
            await session.refresh(db_summary)

            result = db_summary.to_dict()

        logger.info(
            f"✅ Summarization complete for event {event_id}: "
            f"{summary_data['title']} (severity: {summary_data['severity']})"
        )

        return result

    except Exception as e:
        logger.exception(f"Error summarizing event {event_id}: {e}")
        raise


async def generate_summary(
    event_data: dict[str, Any],
    anomaly_score: float,
    suspicious_patterns: list[str],
) -> dict[str, Any]:
    """Generate AI-powered summary using OpenAI.

    Args:
        event_data: Event data dictionary
        anomaly_score: Anomaly score
        suspicious_patterns: Detected patterns

    Returns:
        Summary dictionary with title, severity, root_cause, impact, next_steps
    """
    # Build context for AI
    context = _build_summary_context(event_data, anomaly_score, suspicious_patterns)

    # Call OpenAI API
    if service_settings.ai_provider == "openai":
        return await _generate_with_openai(context)
    else:
        # Fallback to rule-based summary
        logger.warning(f"Unsupported AI provider: {service_settings.ai_provider}, using fallback")
        raise Exception("Unsupported AI provider")


def _build_summary_context(
    event_data: dict[str, Any],
    anomaly_score: float,
    suspicious_patterns: list[str],
) -> str:
    """Build context string for AI summarization.

    Args:
        event_data: Event data
        anomaly_score: Anomaly score
        suspicious_patterns: Detected patterns

    Returns:
        Formatted context string
    """
    payload = event_data.get("payload", {})

    context = f"""Analyze this GitHub security incident and provide a structured assessment.

Event Type: {event_data['event_type']}
Actor: {event_data['actor_login']}
Repository: {event_data['repo_name']}
Timestamp: {event_data['created_at']}
Anomaly Score: {anomaly_score:.2f} (threshold: {service_settings.anomaly_threshold})

Suspicious Patterns Detected:
{chr(10).join(f"- {pattern}" for pattern in suspicious_patterns) if suspicious_patterns else "- None"}

Event Payload:
{json.dumps(payload, indent=2, default=str)[:2000]}

Provide:
- title: A concise incident description (max 200 chars)
- severity: One of: low, medium, high, or critical
- root_cause: 3-5 bullet points explaining what happened
- impact: 3-5 bullet points on potential consequences
- next_steps: 3-5 actionable remediation steps
- tags: Relevant security/incident classification tags"""

    return context


async def _generate_with_openai(context: str) -> dict[str, Any]:
    """Generate summary using OpenAI with structured outputs.

    Args:
        context: Summary context

    Returns:
        Summary dictionary
    """
    if not service_settings.openai_api_key:
        logger.warning("OpenAI API key not set, using fallback")
        raise Exception("OpenAI API key not configured")

    # Initialize OpenAI async client
    client = AsyncOpenAI(api_key=service_settings.openai_api_key)

    # Create structured output request
    response = await client.responses.parse(
        model=service_settings.ai_model,
        input=[
            {
                "role": "user",
                "content": context,
            }
        ],
        text_format=SummaryResponse,
    )

    # Extract parsed response
    summary = response.output_parsed

    # Validate and return
    return {
        "title": summary.title[:200],
        "severity": summary.severity,
        "root_cause": summary.root_cause[:5],
        "impact": summary.impact[:5],
        "next_steps": summary.next_steps[:5],
        "tags": summary.tags[:10],
    }