"""SQLite cache layer for GraphQL enrichment data."""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from github_client.enriched_models import (
    ActorProfile,
    CommitVerification,
    RepositoryContext,
    WorkflowStatus,
)
from service.database import Base

logger = logging.getLogger(__name__)


class ActorProfileCache(Base):
    """Cache for actor/user profile data from GraphQL."""

    __tablename__ = "actor_profile_cache"

    # Primary key
    login: Mapped[str] = mapped_column(String(100), primary_key=True)

    # Profile data
    account_created_at: Mapped[datetime] = mapped_column(DateTime)
    account_age_days: Mapped[int] = mapped_column(Integer)
    follower_count: Mapped[int] = mapped_column(Integer)
    following_count: Mapped[int] = mapped_column(Integer)
    repository_count: Mapped[int] = mapped_column(Integer)
    total_commit_contributions: Mapped[int] = mapped_column(Integer)
    total_pr_contributions: Mapped[int] = mapped_column(Integer)
    total_issue_contributions: Mapped[int] = mapped_column(Integer)
    organizations: Mapped[str] = mapped_column(Text)  # JSON array as string
    is_site_admin: Mapped[bool | None] = mapped_column(Integer, nullable=True)
    company: Mapped[str | None] = mapped_column(String(200), nullable=True)
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Cache metadata
    cached_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    def to_model(self) -> ActorProfile:
        """Convert to ActorProfile model."""
        return ActorProfile(
            login=self.login,
            account_created_at=self.account_created_at,
            account_age_days=self.account_age_days,
            follower_count=self.follower_count,
            following_count=self.following_count,
            repository_count=self.repository_count,
            total_commit_contributions=self.total_commit_contributions,
            total_pr_contributions=self.total_pr_contributions,
            total_issue_contributions=self.total_issue_contributions,
            organizations=json.loads(self.organizations),
            is_site_admin=bool(self.is_site_admin) if self.is_site_admin is not None else None,
            company=self.company,
            location=self.location,
            bio=self.bio,
            cached_at=self.cached_at,
        )

    @classmethod
    def from_model(cls, profile: ActorProfile) -> "ActorProfileCache":
        """Create from ActorProfile model."""
        return cls(
            login=profile.login,
            account_created_at=profile.account_created_at,
            account_age_days=profile.account_age_days,
            follower_count=profile.follower_count,
            following_count=profile.following_count,
            repository_count=profile.repository_count,
            total_commit_contributions=profile.total_commit_contributions,
            total_pr_contributions=profile.total_pr_contributions,
            total_issue_contributions=profile.total_issue_contributions,
            organizations=json.dumps(profile.organizations),
            is_site_admin=int(profile.is_site_admin) if profile.is_site_admin is not None else None,
            company=profile.company,
            location=profile.location,
            bio=profile.bio,
            cached_at=profile.cached_at,
        )


class RepositoryContextCache(Base):
    """Cache for repository context data from GraphQL."""

    __tablename__ = "repository_context_cache"

    # Composite primary key
    owner: Mapped[str] = mapped_column(String(100), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), primary_key=True)

    # Repository data
    full_name: Mapped[str] = mapped_column(String(300), index=True)
    stargazer_count: Mapped[int] = mapped_column(Integer)
    fork_count: Mapped[int] = mapped_column(Integer)
    watcher_count: Mapped[int] = mapped_column(Integer)
    primary_language: Mapped[str | None] = mapped_column(String(50), nullable=True)
    default_branch: Mapped[str | None] = mapped_column(String(100), nullable=True)
    has_security_policy: Mapped[bool] = mapped_column(Integer)
    is_fork: Mapped[bool] = mapped_column(Integer)
    is_archived: Mapped[bool] = mapped_column(Integer)
    topics: Mapped[str] = mapped_column(Text)  # JSON array as string
    license_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Cache metadata
    cached_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    def to_model(self) -> RepositoryContext:
        """Convert to RepositoryContext model."""
        return RepositoryContext(
            owner=self.owner,
            name=self.name,
            full_name=self.full_name,
            stargazer_count=self.stargazer_count,
            fork_count=self.fork_count,
            watcher_count=self.watcher_count,
            primary_language=self.primary_language,
            default_branch=self.default_branch,
            has_security_policy=bool(self.has_security_policy),
            is_fork=bool(self.is_fork),
            is_archived=bool(self.is_archived),
            topics=json.loads(self.topics),
            license_name=self.license_name,
            cached_at=self.cached_at,
        )

    @classmethod
    def from_model(cls, context: RepositoryContext) -> "RepositoryContextCache":
        """Create from RepositoryContext model."""
        return cls(
            owner=context.owner,
            name=context.name,
            full_name=context.full_name,
            stargazer_count=context.stargazer_count,
            fork_count=context.fork_count,
            watcher_count=context.watcher_count,
            primary_language=context.primary_language,
            default_branch=context.default_branch,
            has_security_policy=int(context.has_security_policy),
            is_fork=int(context.is_fork),
            is_archived=int(context.is_archived),
            topics=json.dumps(context.topics),
            license_name=context.license_name,
            cached_at=context.cached_at,
        )


class WorkflowStatusCache(Base):
    """Cache for workflow/CI status data from GraphQL."""

    __tablename__ = "workflow_status_cache"

    # Composite primary key
    repository: Mapped[str] = mapped_column(String(300), primary_key=True)
    commit_sha: Mapped[str] = mapped_column(String(40), primary_key=True)

    # Workflow data
    total_check_suites: Mapped[int] = mapped_column(Integer)
    successful_suites: Mapped[int] = mapped_column(Integer)
    failed_suites: Mapped[int] = mapped_column(Integer)
    pending_suites: Mapped[int] = mapped_column(Integer)
    check_runs: Mapped[str] = mapped_column(Text)  # JSON array as string
    overall_conclusion: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Cache metadata
    cached_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    def to_model(self) -> WorkflowStatus:
        """Convert to WorkflowStatus model."""
        return WorkflowStatus(
            repository=self.repository,
            commit_sha=self.commit_sha,
            total_check_suites=self.total_check_suites,
            successful_suites=self.successful_suites,
            failed_suites=self.failed_suites,
            pending_suites=self.pending_suites,
            check_runs=json.loads(self.check_runs),
            overall_conclusion=self.overall_conclusion,
            cached_at=self.cached_at,
        )

    @classmethod
    def from_model(cls, status: WorkflowStatus) -> "WorkflowStatusCache":
        """Create from WorkflowStatus model."""
        return cls(
            repository=status.repository,
            commit_sha=status.commit_sha,
            total_check_suites=status.total_check_suites,
            successful_suites=status.successful_suites,
            failed_suites=status.failed_suites,
            pending_suites=status.pending_suites,
            check_runs=json.dumps(status.check_runs),
            overall_conclusion=status.overall_conclusion,
            cached_at=status.cached_at,
        )


class CommitVerificationCache(Base):
    """Cache for commit verification data from GraphQL."""

    __tablename__ = "commit_verification_cache"

    # Composite primary key
    repository: Mapped[str] = mapped_column(String(300), primary_key=True)
    sha: Mapped[str] = mapped_column(String(40), primary_key=True)

    # Commit data
    is_signed: Mapped[bool] = mapped_column(Integer)
    signer_login: Mapped[str | None] = mapped_column(String(100), nullable=True)
    signature_valid: Mapped[bool] = mapped_column(Integer)
    additions: Mapped[int] = mapped_column(Integer)
    deletions: Mapped[int] = mapped_column(Integer)
    changed_files: Mapped[int] = mapped_column(Integer)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    author_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    author_email: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # Cache metadata (commits are immutable, so cache never expires)
    cached_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    def to_model(self) -> CommitVerification:
        """Convert to CommitVerification model."""
        return CommitVerification(
            repository=self.repository,
            sha=self.sha,
            is_signed=bool(self.is_signed),
            signer_login=self.signer_login,
            signature_valid=bool(self.signature_valid),
            additions=self.additions,
            deletions=self.deletions,
            changed_files=self.changed_files,
            message=self.message,
            author_name=self.author_name,
            author_email=self.author_email,
            cached_at=self.cached_at,
        )

    @classmethod
    def from_model(cls, verification: CommitVerification) -> "CommitVerificationCache":
        """Create from CommitVerification model."""
        return cls(
            repository=verification.repository,
            sha=verification.sha,
            is_signed=int(verification.is_signed),
            signer_login=verification.signer_login,
            signature_valid=int(verification.signature_valid),
            additions=verification.additions,
            deletions=verification.deletions,
            changed_files=verification.changed_files,
            message=verification.message,
            author_name=verification.author_name,
            author_email=verification.author_email,
            cached_at=verification.cached_at,
        )


class EnrichmentCacheManager:
    """Manager for enrichment cache operations with TTL support."""

    # TTL configurations
    ACTOR_PROFILE_TTL = timedelta(hours=24)
    REPOSITORY_CONTEXT_TTL = timedelta(hours=1)
    WORKFLOW_STATUS_TTL = timedelta(minutes=5)
    # Commits are immutable, no TTL

    def __init__(self, session: AsyncSession):
        """Initialize cache manager.

        Args:
            session: SQLAlchemy async session
        """
        self.session = session
        self._cache_hits = 0
        self._cache_misses = 0

    async def get_actor_profile(self, login: str) -> ActorProfile | None:
        """Get actor profile from cache if not expired.

        Args:
            login: GitHub username

        Returns:
            ActorProfile or None if not cached or expired
        """
        stmt = select(ActorProfileCache).where(ActorProfileCache.login == login)
        result = await self.session.execute(stmt)
        cached = result.scalar_one_or_none()

        if not cached:
            self._cache_misses += 1
            return None

        # Check TTL
        if datetime.utcnow() - cached.cached_at > self.ACTOR_PROFILE_TTL:
            logger.debug(f"Actor profile cache expired for {login}")
            self._cache_misses += 1
            return None

        self._cache_hits += 1
        return cached.to_model()

    async def set_actor_profile(self, profile: ActorProfile) -> None:
        """Store actor profile in cache.

        Args:
            profile: ActorProfile to cache
        """
        cached = ActorProfileCache.from_model(profile)
        await self.session.merge(cached)
        await self.session.commit()
        logger.debug(f"Cached actor profile for {profile.login}")

    async def get_repository_context(
        self, owner: str, name: str
    ) -> RepositoryContext | None:
        """Get repository context from cache if not expired.

        Args:
            owner: Repository owner
            name: Repository name

        Returns:
            RepositoryContext or None if not cached or expired
        """
        stmt = select(RepositoryContextCache).where(
            RepositoryContextCache.owner == owner,
            RepositoryContextCache.name == name,
        )
        result = await self.session.execute(stmt)
        cached = result.scalar_one_or_none()

        if not cached:
            self._cache_misses += 1
            return None

        # Check TTL
        if datetime.utcnow() - cached.cached_at > self.REPOSITORY_CONTEXT_TTL:
            logger.debug(f"Repository context cache expired for {owner}/{name}")
            self._cache_misses += 1
            return None

        self._cache_hits += 1
        return cached.to_model()

    async def set_repository_context(self, context: RepositoryContext) -> None:
        """Store repository context in cache.

        Args:
            context: RepositoryContext to cache
        """
        cached = RepositoryContextCache.from_model(context)
        await self.session.merge(cached)
        await self.session.commit()
        logger.debug(f"Cached repository context for {context.full_name}")

    async def get_workflow_status(
        self, repository: str, commit_sha: str
    ) -> WorkflowStatus | None:
        """Get workflow status from cache if not expired.

        Args:
            repository: Repository name (owner/repo)
            commit_sha: Commit SHA

        Returns:
            WorkflowStatus or None if not cached or expired
        """
        stmt = select(WorkflowStatusCache).where(
            WorkflowStatusCache.repository == repository,
            WorkflowStatusCache.commit_sha == commit_sha,
        )
        result = await self.session.execute(stmt)
        cached = result.scalar_one_or_none()

        if not cached:
            self._cache_misses += 1
            return None

        # Check TTL
        if datetime.utcnow() - cached.cached_at > self.WORKFLOW_STATUS_TTL:
            logger.debug(f"Workflow status cache expired for {repository}@{commit_sha}")
            self._cache_misses += 1
            return None

        self._cache_hits += 1
        return cached.to_model()

    async def set_workflow_status(self, status: WorkflowStatus) -> None:
        """Store workflow status in cache.

        Args:
            status: WorkflowStatus to cache
        """
        cached = WorkflowStatusCache.from_model(status)
        await self.session.merge(cached)
        await self.session.commit()
        logger.debug(f"Cached workflow status for {status.repository}@{status.commit_sha}")

    async def get_commit_verification(
        self, repository: str, sha: str
    ) -> CommitVerification | None:
        """Get commit verification from cache (no expiration - commits are immutable).

        Args:
            repository: Repository name (owner/repo)
            sha: Commit SHA

        Returns:
            CommitVerification or None if not cached
        """
        stmt = select(CommitVerificationCache).where(
            CommitVerificationCache.repository == repository,
            CommitVerificationCache.sha == sha,
        )
        result = await self.session.execute(stmt)
        cached = result.scalar_one_or_none()

        if not cached:
            self._cache_misses += 1
            return None

        self._cache_hits += 1
        return cached.to_model()

    async def set_commit_verification(self, verification: CommitVerification) -> None:
        """Store commit verification in cache.

        Args:
            verification: CommitVerification to cache
        """
        cached = CommitVerificationCache.from_model(verification)
        await self.session.merge(cached)
        await self.session.commit()
        logger.debug(f"Cached commit verification for {verification.repository}@{verification.sha}")

    async def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        count = 0
        now = datetime.utcnow()

        # Clean up actor profiles
        expired_actors = select(ActorProfileCache).where(
            ActorProfileCache.cached_at < now - self.ACTOR_PROFILE_TTL
        )
        result = await self.session.execute(expired_actors)
        for actor in result.scalars():
            await self.session.delete(actor)
            count += 1

        # Clean up repository contexts
        expired_repos = select(RepositoryContextCache).where(
            RepositoryContextCache.cached_at < now - self.REPOSITORY_CONTEXT_TTL
        )
        result = await self.session.execute(expired_repos)
        for repo in result.scalars():
            await self.session.delete(repo)
            count += 1

        # Clean up workflow statuses
        expired_workflows = select(WorkflowStatusCache).where(
            WorkflowStatusCache.cached_at < now - self.WORKFLOW_STATUS_TTL
        )
        result = await self.session.execute(expired_workflows)
        for workflow in result.scalars():
            await self.session.delete(workflow)
            count += 1

        # Don't clean up commit verifications (immutable)

        if count > 0:
            await self.session.commit()
            logger.info(f"Cleaned up {count} expired cache entries")

        return count

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get cache hit/miss statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total_requests": total,
            "hit_rate": hit_rate,
        }
