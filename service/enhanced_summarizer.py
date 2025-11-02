"""Enhanced AI-powered summarization with GraphQL enrichment data."""

import json
import logging
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from github_client.enriched_models import EnrichedEvent
from service.config import service_settings
from service.database import AsyncSessionLocal, AnomalySummary
from service.sse_models import Severity

logger = logging.getLogger(__name__)


class EnhancedSummaryResponse(BaseModel):
    """Pydantic model for enhanced structured AI summary output."""

    title: str
    severity: Severity
    severity_reasoning: str
    root_cause: list[str]
    impact: list[str]
    next_steps: list[str]
    tags: list[str]


async def summarize_enriched_anomaly(enriched_event: EnrichedEvent) -> dict[str, Any]:
    """Generate AI summary for an enriched anomaly.

    Args:
        enriched_event: EnrichedEvent with GraphQL enrichment data

    Returns:
        Summary data dictionary
    """
    event = enriched_event.event
    logger.info(f"Starting enhanced summarization for event {event.id}")

    try:
        # Generate AI summary with enriched context
        summary_data = await generate_enhanced_summary(enriched_event)

        # Save to database
        async with AsyncSessionLocal() as session:
            db_summary = AnomalySummary(
                event_id=event.id,
                title=summary_data["title"],
                severity=summary_data["severity"],
                severity_reasoning=summary_data["severity_reasoning"],
                root_cause=summary_data["root_cause"],
                impact=summary_data["impact"],
                next_steps=summary_data["next_steps"],
                suspicious_patterns=enriched_event.suspicious_patterns,
                anomaly_score=enriched_event.anomaly_score,
                event_type=event.type,
                actor_login=event.actor.login,
                repo_name=event.repo.name,
                raw_event=event.model_dump(mode="json"),
                event_timestamp=event.created_at,
                tags=summary_data.get("tags", []),
            )

            session.add(db_summary)
            await session.commit()
            await session.refresh(db_summary)

            result = db_summary.to_dict()

        logger.info(
            f"Enhanced summarization complete for event {event.id}: "
            f"{summary_data['title']} (severity: {summary_data['severity']})"
        )

        return result

    except Exception as e:
        logger.exception(f"Error summarizing enriched event {event.id}: {e}")
        raise


async def generate_enhanced_summary(enriched_event: EnrichedEvent) -> dict[str, Any]:
    """Generate AI-powered summary using OpenAI with enrichment data.

    Args:
        enriched_event: EnrichedEvent with all enrichment data

    Returns:
        Summary dictionary with title, severity, root_cause, impact, next_steps
    """
    # Build enhanced context for AI
    context = _build_enhanced_context(enriched_event)

    # Call OpenAI API
    if service_settings.ai_provider == "openai":
        return await _generate_with_openai(context)
    else:
        logger.warning(
            f"Unsupported AI provider: {service_settings.ai_provider}, using fallback"
        )
        raise Exception("Unsupported AI provider")


def _build_enhanced_context(enriched_event: EnrichedEvent) -> str:
    """Build enhanced context string with GraphQL enrichment data.

    Args:
        enriched_event: EnrichedEvent with enrichment data

    Returns:
        Formatted context string with enrichment
    """
    event = enriched_event.event
    payload = event.payload

    # Base event information
    context_parts = [
        "Analyze this GitHub security incident with the following enriched context.\n",
        f"Event Type: {event.type}",
        f"Actor: {event.actor.login}",
        f"Repository: {event.repo.name}",
        f"Timestamp: {event.created_at}",
        f"Anomaly Score: {enriched_event.anomaly_score:.2f} (threshold: {service_settings.anomaly_threshold})\n",
    ]

    # Suspicious patterns
    if enriched_event.suspicious_patterns:
        context_parts.append("Suspicious Patterns Detected:")
        for pattern in enriched_event.suspicious_patterns:
            context_parts.append(f"- {pattern}")
        context_parts.append("")

    # Actor profile enrichment
    if enriched_event.actor_profile:
        profile = enriched_event.actor_profile
        context_parts.extend(
            [
                "ACTOR PROFILE:",
                f"- Account Age: {profile.account_age_days} days "
                f"({'NEW ACCOUNT' if profile.is_new_account else 'Established'})",
                f"- Total Contributions: {profile.total_contributions} "
                f"({'Active' if profile.is_active_contributor else 'Low activity'})",
                f"- Followers: {profile.follower_count}",
                f"- Repositories: {profile.repository_count}",
            ]
        )
        if profile.organizations:
            context_parts.append(
                f"- Organizations: {', '.join(profile.organizations[:5])}"
            )
        if profile.is_site_admin:
            context_parts.append("- SITE ADMINISTRATOR")
        if profile.company:
            context_parts.append(f"- Company: {profile.company}")
        context_parts.append("")

    # Repository context enrichment
    if enriched_event.repository_context:
        repo = enriched_event.repository_context
        context_parts.extend(
            [
                "REPOSITORY CONTEXT:",
                f"- Popularity: {repo.stargazer_count} stars, {repo.fork_count} forks "
                f"({'CRITICAL' if repo.is_critical else 'Popular' if repo.is_popular else 'Standard'})",
                "- Visibility: Public (monitoring public events only)",
                f"- Security Score: {repo.security_score}/1",
            ]
        )
        if repo.has_security_policy:
            context_parts.append("  Security policy present")
        if repo.is_archived:
            context_parts.append("  Repository is ARCHIVED")
        if repo.primary_language:
            context_parts.append(f"- Primary Language: {repo.primary_language}")
        if repo.topics:
            context_parts.append(f"- Topics: {', '.join(repo.topics[:5])}")
        context_parts.append("")

    # Workflow status enrichment
    if enriched_event.workflow_status:
        workflow = enriched_event.workflow_status
        context_parts.extend(
            [
                "CI/CD WORKFLOW STATUS:",
                f"- Total Check Suites: {workflow.total_check_suites}",
                f"- Status: {workflow.overall_conclusion or 'Unknown'}",
            ]
        )
        if workflow.has_failures:
            context_parts.append(
                f"  FAILURES DETECTED: {workflow.failed_suites} failed suites"
            )
            if workflow.failed_check_names:
                context_parts.append(
                    f"  Failed checks: {', '.join(workflow.failed_check_names[:5])}"
                )
        elif workflow.all_passed:
            context_parts.append("  All checks passed")
        elif workflow.pending_suites > 0:
            context_parts.append(f"  {workflow.pending_suites} suites pending")
        context_parts.append("")

    # Commit verification enrichment
    if enriched_event.commit_verification:
        commit = enriched_event.commit_verification
        context_parts.extend(
            [
                "COMMIT VERIFICATION:",
                f"- Signature: {'Signed' if commit.is_verified else 'NOT SIGNED'}",
            ]
        )
        if commit.is_signed and commit.signer_login:
            context_parts.append(f"  Signer: {commit.signer_login}")
        context_parts.extend(
            [
                f"- Commit Size: {commit.commit_size} lines changed "
                f"(+{commit.additions} -{commit.deletions}, {commit.changed_files} files) "
                f"{'[LARGE COMMIT]' if commit.is_large_commit else ''}",
            ]
        )
        # Add entropy information
        if commit.commit_entropy is not None:
            entropy_warning = ""
            if commit.has_high_entropy:
                entropy_warning = " SUSPICIOUS - POSSIBLE OBFUSCATED CODE"
            context_parts.append(
                f"- Code Entropy: {commit.commit_entropy:.2f} "
                f"({commit.entropy_level}){entropy_warning}"
            )
        else:
            context_parts.append("- Code Entropy: Unable to calculate")
        if commit.author_name:
            context_parts.append(
                f"- Author: {commit.author_name} <{commit.author_email}>"
            )
        context_parts.append("")

    # Event payload (truncated)
    context_parts.extend(
        [
            "Event Payload (sample):",
            json.dumps(payload, indent=2, default=str)[:1500],
            "\n",
        ]
    )

    # Instructions for AI with detailed severity criteria
    context_parts.extend(
        [
            "Based on the above enriched context, provide:\n",
            "- title: A concise incident description (max 200 chars)",
            "",
            "- severity: Choose ONE of: low, medium, high, or critical based on these criteria:",
            "  * CRITICAL: Destructive actions (force push, branch deletion), privilege escalations, ",
            "    verified security incidents, malicious code injection, compromised credentials,",
            "    HIGH ENTROPY commits (>7.0) indicating obfuscated/malicious code",
            "  * HIGH: Suspicious new accounts (<7 days) performing sensitive actions on critical repos,",
            "    unsigned commits to protected branches, failed CI/CD with security implications,",
            "    unusual permission changes, elevated entropy (6.0-7.0) with other suspicious indicators,",
            "    very large commits (>1000 lines) from untrusted sources",
            "  * MEDIUM: Unusual patterns from established accounts, policy violations,",
            "    unsigned commits on standard repos, minor workflow failures, inactive accounts with activity,",
            "    large commits (>500 lines) without other red flags, spammy usernames or repo names",
            "  * LOW: Benign unusual activity, legitimate owner/maintainer maintenance actions,",
            "    administrative tasks by trusted users, normal entropy (<6.0)",
            "",
            "- severity_reasoning: 1-2 sentences explaining WHY you chose this severity level",
            "  (reference specific enrichment data: actor profile, repo criticality, patterns detected,",
            "  commit entropy level, and commit size)",
            "",
            "- root_cause: 3-5 bullet points explaining what happened (use enrichment data)",
            "- impact: 3-5 bullet points on potential consequences (assess based on repo criticality, actor trust, etc.)",
            "- next_steps: 3-5 actionable remediation steps (prioritized by severity)",
            "- tags: Relevant security/incident classification tags",
        ]
    )

    return "\n".join(context_parts)


async def _generate_with_openai(context: str) -> dict[str, Any]:
    """Generate summary using OpenAI with structured outputs.

    Args:
        context: Enhanced summary context

    Returns:
        Summary dictionary
    """
    if not service_settings.openai_api_key:
        logger.warning("OpenAI API key not set")
        raise Exception("OpenAI API key not configured")

    # Initialize OpenAI async client
    client = AsyncOpenAI(api_key=service_settings.openai_api_key)

    # Create structured output request
    response = await client.responses.parse(
        model=service_settings.ai_model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a GitHub security analyst. Analyze incidents with "
                    "enriched context (actor profiles, repository metadata, CI/CD status, "
                    "commit verification) to provide accurate risk assessments."
                ),
            },
            {
                "role": "user",
                "content": context,
            },
        ],
        text_format=EnhancedSummaryResponse,
    )

    # Extract parsed response
    summary = response.output_parsed

    # Validate and return
    return {
        "title": summary.title[:200],
        "severity": summary.severity.value,  # Convert enum to string
        "severity_reasoning": summary.severity_reasoning[:500],
        "root_cause": summary.root_cause[:5],
        "impact": summary.impact[:5],
        "next_steps": summary.next_steps[:5],
        "tags": summary.tags[:10],
    }
