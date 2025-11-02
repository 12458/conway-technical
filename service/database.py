"""Database models and session management."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from service.config import service_settings


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class GitHubEvent(Base):
    """Stores raw GitHub event data."""

    __tablename__ = "github_events"

    # Primary fields
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(50), index=True)

    # Actor information
    actor_login: Mapped[str] = mapped_column(String(100), index=True)
    actor_id: Mapped[int] = mapped_column(Integer)

    # Repository information
    repo_name: Mapped[str] = mapped_column(String(200), index=True)
    repo_id: Mapped[int] = mapped_column(Integer)

    # Organization (optional)
    org_login: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    # Raw payload
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)

    # Processing flags
    processed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "actor_login": self.actor_login,
            "actor_id": self.actor_id,
            "repo_name": self.repo_name,
            "repo_id": self.repo_id,
            "org_login": self.org_login,
            "created_at": self.created_at.isoformat(),
            "ingested_at": self.ingested_at.isoformat(),
            "payload": self.payload,
            "processed": self.processed,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
        }

    def to_event_dict(self) -> dict[str, Any]:
        """Convert to Event model dictionary format.

        Reconstructs the original Event structure from flattened database fields.
        """
        # Build actor object
        actor_data = {
            "id": self.actor_id,
            "login": self.actor_login,
            "display_login": self.actor_login,
            "gravatar_id": "",
            "url": f"https://api.github.com/users/{self.actor_login}",
            "avatar_url": f"https://avatars.githubusercontent.com/u/{self.actor_id}",
        }

        # Build repo object
        repo_data = {
            "id": self.repo_id,
            "name": self.repo_name,
            "url": f"https://api.github.com/repos/{self.repo_name}",
        }

        # Build org object if present
        org_data = None
        if self.org_login:
            org_data = {
                "id": 0,  # We don't store org_id, use placeholder
                "login": self.org_login,
                "gravatar_id": "",
                "url": f"https://api.github.com/orgs/{self.org_login}",
                "avatar_url": "https://avatars.githubusercontent.com/u/0",
            }

        # Build Event dictionary
        event_dict = {
            "id": self.id,
            "type": self.event_type,
            "actor": actor_data,
            "repo": repo_data,
            "payload": self.payload,
            "public": True,  # Assume public since we only track public events
            "created_at": self.created_at.isoformat()
            if hasattr(self.created_at, "isoformat")
            else self.created_at,
        }

        if org_data:
            event_dict["org"] = org_data

        return event_dict


class AnomalySummary(Base):
    """Stores AI-generated summaries for detected anomalies."""

    __tablename__ = "anomaly_summaries"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Related event
    event_id: Mapped[str] = mapped_column(String(50), index=True)

    # Summary fields
    title: Mapped[str] = mapped_column(String(200))
    severity: Mapped[str] = mapped_column(
        String(20), index=True
    )  # low, medium, high, critical
    severity_reasoning: Mapped[str | None] = mapped_column(
        String(500), nullable=True
    )  # LLM explanation for severity

    # Structured summary
    root_cause: Mapped[list[str]] = mapped_column(JSON)  # 3-5 bullets
    impact: Mapped[list[str]] = mapped_column(JSON)  # 3-5 bullets
    next_steps: Mapped[list[str]] = mapped_column(JSON)  # 3-5 bullets

    # Context
    suspicious_patterns: Mapped[list[str]] = mapped_column(JSON)
    anomaly_score: Mapped[float] = mapped_column(Float)

    # Event context
    event_type: Mapped[str] = mapped_column(String(50))
    actor_login: Mapped[str] = mapped_column(String(100), index=True)
    repo_name: Mapped[str] = mapped_column(String(200), index=True)

    # Raw event data for frontend display
    raw_event: Mapped[dict[str, Any]] = mapped_column(JSON)

    # Timestamps
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )

    # Metadata
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response.

        Note: This method is kept for backward compatibility.
        For typed responses, use to_response() instead.
        """
        return {
            "id": self.id,
            "event_id": self.event_id,
            "title": self.title,
            "severity": self.severity,
            "severity_reasoning": self.severity_reasoning,
            "root_cause": self.root_cause,
            "impact": self.impact,
            "next_steps": self.next_steps,
            "suspicious_patterns": self.suspicious_patterns,
            "anomaly_score": self.anomaly_score,
            "event_type": self.event_type,
            "actor_login": self.actor_login,
            "repo_name": self.repo_name,
            "raw_event": self.raw_event,
            "event_timestamp": self.event_timestamp.isoformat(),
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }

    def to_response(self) -> "AnomalySummaryResponse":
        """Convert to strongly-typed Pydantic response model.

        Returns:
            AnomalySummaryResponse with full validation and type safety
        """
        from service.sse_models import AnomalySummaryResponse, Severity

        return AnomalySummaryResponse(
            id=self.id,
            event_id=self.event_id,
            title=self.title,
            severity=Severity(self.severity),
            severity_reasoning=self.severity_reasoning,
            root_cause=self.root_cause,
            impact=self.impact,
            next_steps=self.next_steps,
            suspicious_patterns=self.suspicious_patterns,
            anomaly_score=self.anomaly_score,
            event_type=self.event_type,
            actor_login=self.actor_login,
            repo_name=self.repo_name,
            raw_event=self.raw_event,
            event_timestamp=self.event_timestamp,
            created_at=self.created_at,
            tags=self.tags,
        )


# Database engine and session
# Convert database URL to use async driver if needed
database_url = service_settings.database_url
if database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    # asyncpg doesn't support sslmode or channel_binding query params, use ssl=require instead
    if "sslmode=" in database_url:
        import re

        # Remove sslmode and channel_binding params, add ssl=require
        database_url = re.sub(r"[&?]sslmode=[^&]*", "", database_url)
        database_url = re.sub(r"[&?]channel_binding=[^&]*", "", database_url)
        # Add ssl=require
        separator = "&" if "?" in database_url else "?"
        database_url = database_url + separator + "ssl=require"
elif database_url.startswith("sqlite+aiosqlite"):
    pass  # Already correct
elif database_url.startswith("sqlite://"):
    database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://", 1)

engine = create_async_engine(
    database_url,
    echo=False,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Initialize database tables."""
    # Import cache models to register them with Base.metadata
    try:
        from github_client.enrichment_cache import (
            ActorProfileCache,
            CommitVerificationCache,
            RepositoryContextCache,
            WorkflowStatusCache,
        )
    except ImportError:
        pass  # Enrichment cache tables not available

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Get database session dependency."""
    async with AsyncSessionLocal() as session:
        yield session
