"""Database models and session management."""

import json
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from service.config import service_settings


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
    org_login: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

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


class AnomalySummary(Base):
    """Stores AI-generated summaries for detected anomalies."""

    __tablename__ = "anomaly_summaries"

    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Related event
    event_id: Mapped[str] = mapped_column(String(50), index=True)

    # Summary fields
    title: Mapped[str] = mapped_column(String(200))
    severity: Mapped[str] = mapped_column(String(20), index=True)  # low, medium, high, critical

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
    event_timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Metadata
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "event_id": self.event_id,
            "title": self.title,
            "severity": self.severity,
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


# Database engine and session
engine = create_async_engine(
    service_settings.database_url,
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
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Get database session dependency."""
    async with AsyncSessionLocal() as session:
        yield session
