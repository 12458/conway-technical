"""Pydantic models for enriched GitHub data from GraphQL API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from github_client.models import Event


class ActorProfile(BaseModel):
    """Enriched actor/user profile data from GraphQL."""

    login: str = Field(description="Username")
    account_created_at: datetime = Field(description="Account creation timestamp")
    account_age_days: int = Field(description="Age of account in days")
    follower_count: int = Field(description="Number of followers")
    following_count: int = Field(description="Number of users following")
    repository_count: int = Field(description="Number of repositories owned")
    total_commit_contributions: int = Field(description="Total commit contributions")
    total_pr_contributions: int = Field(description="Total PR contributions")
    total_issue_contributions: int = Field(description="Total issue contributions")
    organizations: list[str] = Field(
        default_factory=list, description="Organization memberships"
    )
    is_site_admin: bool | None = Field(None, description="Whether user is site admin")
    company: str | None = Field(None, description="Company affiliation")
    location: str | None = Field(None, description="User location")
    bio: str | None = Field(None, description="User bio")
    cached_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this data was cached"
    )

    @property
    def total_contributions(self) -> int:
        """Total contributions across all types."""
        return (
            self.total_commit_contributions
            + self.total_pr_contributions
            + self.total_issue_contributions
        )

    @property
    def is_new_account(self) -> bool:
        """Whether account is less than 30 days old."""
        return self.account_age_days < 30

    @property
    def is_active_contributor(self) -> bool:
        """Whether user has significant contribution history."""
        return self.total_contributions > 100


class RepositoryContext(BaseModel):
    """Enriched repository context data from GraphQL."""

    owner: str = Field(description="Repository owner")
    name: str = Field(description="Repository name")
    full_name: str = Field(description="Full repository name (owner/name)")
    stargazer_count: int = Field(description="Number of stars")
    fork_count: int = Field(description="Number of forks")
    watcher_count: int = Field(description="Number of watchers")
    primary_language: str | None = Field(None, description="Primary programming language")
    default_branch: str | None = Field(None, description="Default branch name")
    has_security_policy: bool = Field(description="Whether repo has security policy")
    is_fork: bool = Field(description="Whether this is a fork")
    is_archived: bool = Field(description="Whether repository is archived")
    topics: list[str] = Field(default_factory=list, description="Repository topics")
    license_name: str | None = Field(None, description="License type")
    cached_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this data was cached"
    )

    @property
    def is_popular(self) -> bool:
        """Whether repository is popular (>1000 stars)."""
        return self.stargazer_count > 1000

    @property
    def is_critical(self) -> bool:
        """Whether repository is critical (>5000 stars or has security policy)."""
        return self.stargazer_count > 5000 or self.has_security_policy

    @property
    def security_score(self) -> int:
        """Simple security score (0-1) based on security policy presence."""
        return 1 if self.has_security_policy else 0


class WorkflowStatus(BaseModel):
    """Enriched workflow/CI status from GraphQL."""

    repository: str = Field(description="Repository name (owner/repo)")
    commit_sha: str = Field(description="Commit SHA")
    total_check_suites: int = Field(description="Number of check suites")
    successful_suites: int = Field(description="Number of successful check suites")
    failed_suites: int = Field(description="Number of failed check suites")
    pending_suites: int = Field(description="Number of pending check suites")
    check_runs: list[dict[str, Any]] = Field(
        default_factory=list, description="Individual check run details"
    )
    overall_conclusion: str | None = Field(
        None, description="Overall conclusion (SUCCESS, FAILURE, etc.)"
    )
    cached_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this data was cached"
    )

    @property
    def has_failures(self) -> bool:
        """Whether any check suite failed."""
        return self.failed_suites > 0

    @property
    def all_passed(self) -> bool:
        """Whether all check suites passed."""
        return self.failed_suites == 0 and self.pending_suites == 0

    @property
    def failed_check_names(self) -> list[str]:
        """List of failed check run names."""
        return [
            run["name"]
            for run in self.check_runs
            if run.get("conclusion") == "FAILURE"
        ]


class CommitVerification(BaseModel):
    """Enriched commit verification data from GraphQL."""

    repository: str = Field(description="Repository name (owner/repo)")
    sha: str = Field(description="Commit SHA")
    is_signed: bool = Field(description="Whether commit has valid GPG signature")
    signer_login: str | None = Field(None, description="Login of the signer")
    signature_valid: bool = Field(
        description="Whether signature is cryptographically valid"
    )
    additions: int = Field(description="Number of lines added")
    deletions: int = Field(description="Number of lines deleted")
    changed_files: int = Field(description="Number of files changed")
    message: str | None = Field(None, description="Commit message")
    author_name: str | None = Field(None, description="Commit author name")
    author_email: str | None = Field(None, description="Commit author email")
    cached_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this data was cached"
    )

    @property
    def total_changes(self) -> int:
        """Total lines changed."""
        return self.additions + self.deletions

    @property
    def is_large_commit(self) -> bool:
        """Whether commit changes >500 lines."""
        return self.total_changes > 500

    @property
    def is_verified(self) -> bool:
        """Whether commit is signed and signature is valid."""
        return self.is_signed and self.signature_valid


class EnrichedEvent(BaseModel):
    """Event with enriched GraphQL data."""

    event: Event = Field(description="Original GitHub event")
    anomaly_score: float = Field(description="Anomaly detection score")
    suspicious_patterns: list[str] = Field(
        default_factory=list, description="Detected suspicious patterns"
    )
    actor_profile: ActorProfile | None = Field(
        None, description="Enriched actor profile"
    )
    repository_context: RepositoryContext | None = Field(
        None, description="Enriched repository context"
    )
    workflow_status: WorkflowStatus | None = Field(
        None, description="Enriched workflow status"
    )
    commit_verification: CommitVerification | None = Field(
        None, description="Enriched commit verification"
    )
    enrichment_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When enrichment occurred"
    )

    @property
    def risk_level(self) -> str:
        """Compute overall risk level based on enriched data."""
        risk_score = 0

        # Anomaly score contributes to risk
        if self.anomaly_score > 80:
            risk_score += 3
        elif self.anomaly_score > 60:
            risk_score += 2
        elif self.anomaly_score > 40:
            risk_score += 1

        # Actor profile risk factors
        if self.actor_profile:
            if self.actor_profile.is_new_account:
                risk_score += 2
            if not self.actor_profile.is_active_contributor:
                risk_score += 1
            if self.actor_profile.is_site_admin:
                risk_score += 2  # Admin actions are higher risk

        # Repository context risk factors
        if self.repository_context:
            if self.repository_context.is_critical:
                risk_score += 2

        # Workflow status risk factors
        if self.workflow_status and self.workflow_status.has_failures:
            risk_score += 2

        # Commit verification risk factors
        if self.commit_verification and not self.commit_verification.is_verified:
            risk_score += 1
        if self.commit_verification and self.commit_verification.is_large_commit:
            risk_score += 1

        # Pattern risk
        risk_score += len(self.suspicious_patterns)

        # Convert score to risk level
        if risk_score >= 10:
            return "CRITICAL"
        elif risk_score >= 7:
            return "HIGH"
        elif risk_score >= 4:
            return "MEDIUM"
        else:
            return "LOW"

    @property
    def enrichment_summary(self) -> dict[str, Any]:
        """Summary of enrichment data for logging/alerting."""
        return {
            "event_id": self.event.id,
            "event_type": self.event.type,
            "actor": self.event.actor.login,
            "repo": self.event.repo.name,
            "anomaly_score": self.anomaly_score,
            "risk_level": self.risk_level,
            "patterns": self.suspicious_patterns,
            "actor_age_days": (
                self.actor_profile.account_age_days if self.actor_profile else None
            ),
            "repo_stars": (
                self.repository_context.stargazer_count
                if self.repository_context
                else None
            ),
            "commit_signed": (
                self.commit_verification.is_signed if self.commit_verification else None
            ),
            "workflow_failed": (
                self.workflow_status.has_failures if self.workflow_status else None
            ),
        }
