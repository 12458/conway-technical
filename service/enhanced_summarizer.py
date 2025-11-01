"""Enhanced AI-powered summarization with GraphQL enrichment data."""

import json
import logging
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from github_client.enriched_models import EnrichedEvent
from service.config import service_settings
from service.database import AsyncSessionLocal, AnomalySummary

logger = logging.getLogger(__name__)


class EnhancedSummaryResponse(BaseModel):
    """Pydantic model for enhanced structured AI summary output."""

    title: str
    severity: str
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
    logger.info(
        f"Starting enhanced summarization for event {event.id} "
        f"(risk level: {enriched_event.risk_level})"
    )

    try:
        # Generate AI summary with enriched context
        summary_data = await generate_enhanced_summary(enriched_event)

        # Override severity based on enrichment if needed
        summary_data["severity"] = _compute_final_severity(
            summary_data["severity"], enriched_event
        )

        # Save to database
        async with AsyncSessionLocal() as session:
            db_summary = AnomalySummary(
                event_id=event.id,
                title=summary_data["title"],
                severity=summary_data["severity"],
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
                tags=summary_data.get("tags", []) + [f"risk:{enriched_event.risk_level.lower()}"],
            )

            session.add(db_summary)
            await session.commit()
            await session.refresh(db_summary)

            result = db_summary.to_dict()

        logger.info(
            f"✅ Enhanced summarization complete for event {event.id}: "
            f"{summary_data['title']} (severity: {summary_data['severity']}, "
            f"risk: {enriched_event.risk_level})"
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
        f"Anomaly Score: {enriched_event.anomaly_score:.2f} (threshold: {service_settings.anomaly_threshold})",
        f"Computed Risk Level: {enriched_event.risk_level}\n",
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
            context_parts.append("- ⚠️ SITE ADMINISTRATOR")
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
                f"- Visibility: Public (monitoring public events only)",
                f"- Security Score: {repo.security_score}/1",
            ]
        )
        if repo.has_security_policy:
            context_parts.append("  ✓ Security policy present")
        if repo.is_archived:
            context_parts.append("  ⚠️ Repository is ARCHIVED")
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
                f"  ⚠️ FAILURES DETECTED: {workflow.failed_suites} failed suites"
            )
            if workflow.failed_check_names:
                context_parts.append(
                    f"  Failed checks: {', '.join(workflow.failed_check_names[:5])}"
                )
        elif workflow.all_passed:
            context_parts.append("  ✓ All checks passed")
        elif workflow.pending_suites > 0:
            context_parts.append(f"  ⏳ {workflow.pending_suites} suites pending")
        context_parts.append("")

    # Commit verification enrichment
    if enriched_event.commit_verification:
        commit = enriched_event.commit_verification
        context_parts.extend(
            [
                "COMMIT VERIFICATION:",
                f"- Signature: {'✓ Signed' if commit.is_verified else '⚠️ NOT SIGNED'}",
            ]
        )
        if commit.is_signed and commit.signer_login:
            context_parts.append(f"  Signer: {commit.signer_login}")
        context_parts.extend(
            [
                f"- Changes: +{commit.additions} -{commit.deletions} "
                f"({commit.changed_files} files) "
                f"{'[LARGE COMMIT]' if commit.is_large_commit else ''}",
            ]
        )
        if commit.author_name:
            context_parts.append(f"- Author: {commit.author_name} <{commit.author_email}>")
        context_parts.append("")

    # Event payload (truncated)
    context_parts.extend(
        [
            "Event Payload (sample):",
            json.dumps(payload, indent=2, default=str)[:1500],
            "\n",
        ]
    )

    # Instructions for AI
    context_parts.extend(
        [
            "Based on the above enriched context, provide:",
            "- title: A concise incident description (max 200 chars)",
            "- severity: One of: low, medium, high, or critical (consider enrichment context)",
            "- root_cause: 3-5 bullet points explaining what happened (use enrichment data)",
            "- impact: 3-5 bullet points on potential consequences (assess based on repo criticality, actor trust, etc.)",
            "- next_steps: 3-5 actionable remediation steps (prioritized by risk)",
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
        "severity": summary.severity,
        "root_cause": summary.root_cause[:5],
        "impact": summary.impact[:5],
        "next_steps": summary.next_steps[:5],
        "tags": summary.tags[:10],
    }


def _is_owner_or_maintainer_action(enriched_event: EnrichedEvent) -> bool:
    """Check if the event is an owner/maintainer performing administrative actions.

    Args:
        enriched_event: EnrichedEvent with actor and repo data

    Returns:
        True if this is likely an owner/maintainer administrative action
    """
    event = enriched_event.event
    actor_login = event.actor.login
    repo_name = event.repo.name

    # Check if actor is the repo owner (owner/repo-name pattern)
    repo_owner = repo_name.split("/")[0] if "/" in repo_name else ""
    is_repo_owner = actor_login == repo_owner

    # Check if actor is in the repo's organization (if enrichment available)
    is_org_member = False
    if enriched_event.actor_profile and enriched_event.actor_profile.organizations:
        # Extract org from repo name
        org_name = repo_owner  # For org repos, owner == org
        is_org_member = org_name in enriched_event.actor_profile.organizations

    # Define low-risk administrative event types
    low_risk_admin_types = {
        "IssuesEvent",  # Issue assignment, labels, etc.
        "IssueCommentEvent",  # Comments (unless on external PRs)
    }

    # Check for self-authored PR comments
    is_self_authored = False
    if event.type == "IssueCommentEvent":
        issue = event.payload.get("issue", {})
        issue_user = issue.get("user", {})
        if issue_user.get("login") == actor_login:
            is_self_authored = True

    return (is_repo_owner or is_org_member) and (
        event.type in low_risk_admin_types or is_self_authored
    )


def _compute_final_severity(ai_severity: str, enriched_event: EnrichedEvent) -> str:
    """Compute final severity considering both AI assessment and enrichment data.

    Args:
        ai_severity: Severity suggested by AI
        enriched_event: EnrichedEvent with risk level

    Returns:
        Final severity level (low, medium, high, critical)
    """
    # Map AI severity and risk level to numeric scores
    severity_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    risk_scores = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

    ai_score = severity_scores.get(ai_severity.lower(), 2)
    risk_score = risk_scores.get(enriched_event.risk_level, 2)

    # Take the maximum (most severe)
    final_score = max(ai_score, risk_score)

    # Downweight owner/maintainer administrative actions
    if _is_owner_or_maintainer_action(enriched_event):
        # Cap severity at MEDIUM for benign admin actions
        final_score = min(final_score, 2)

    # Anomaly score correlation (prevent miscalibration)
    # Low anomaly scores should not result in HIGH/CRITICAL severity
    if enriched_event.anomaly_score < 50:
        final_score = min(final_score, 2)  # Cap at MEDIUM
    # Very high anomaly scores should be at least MEDIUM
    if enriched_event.anomaly_score > 100:
        final_score = max(final_score, 2)  # Floor at MEDIUM

    # Additional escalation rules (removed site_admin auto-escalation)
    # Site admin actions are not inherently risky - rely on other signals

    if enriched_event.repository_context and enriched_event.repository_context.is_critical:
        # Critical repos warrant higher scrutiny, but not automatic escalation
        # Only escalate if there are other risk signals
        if len(enriched_event.suspicious_patterns) > 0:
            final_score = max(final_score, 3)  # At least HIGH for critical repos with patterns

    if (
        enriched_event.commit_verification
        and not enriched_event.commit_verification.is_verified
        and enriched_event.repository_context
    ):
        final_score = max(final_score, 2)  # At least MEDIUM for unsigned commits

    # Map back to severity string
    score_to_severity = {1: "low", 2: "medium", 3: "high", 4: "critical"}
    return score_to_severity[final_score]
