"""Database models and session management for PostgreSQL."""

import logging
import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from service.config import service_settings

logger = logging.getLogger(__name__)


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


def _normalize_postgresql_url(url: str) -> str:
    """Normalize PostgreSQL URL for asyncpg driver.

    Converts standard PostgreSQL URLs to asyncpg format and handles
    SSL parameter conversion for compatibility with asyncpg.

    Args:
        url: PostgreSQL connection URL

    Returns:
        Normalized URL for asyncpg

    Notes:
        - Converts postgresql:// to postgresql+asyncpg://
        - Removes sslmode and channel_binding params (not supported by asyncpg)
        - Adds ssl=require for secure connections
    """
    if not url.startswith("postgresql://"):
        logger.warning(
            f"Database URL does not start with 'postgresql://': {url[:50]}... "
            "This may cause connection issues."
        )
        return url

    # Convert to asyncpg driver
    url = url.replace("postgresql://", "postgresql+asyncpg://", 1)

    # Handle SSL parameters (asyncpg doesn't support sslmode or channel_binding)
    if "sslmode=" in url or "channel_binding=" in url:
        logger.info("Converting PostgreSQL SSL parameters for asyncpg compatibility")

        # Remove unsupported parameters
        url = re.sub(r"[&?]sslmode=[^&]*", "", url)
        url = re.sub(r"[&?]channel_binding=[^&]*", "", url)

        # Clean up trailing separators
        url = re.sub(r"[?&]$", "", url)

        # Add ssl=require if not already present
        if "ssl=" not in url:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}ssl=require"
            logger.info("Added ssl=require parameter for secure connection")

    return url


# Database URL normalization
database_url = _normalize_postgresql_url(service_settings.database_url)
logger.info(f"Database configured: {database_url.split('@')[0]}@***")

# Create async engine with production-ready settings
engine = create_async_engine(
    database_url,
    echo=False,  # Set to True for SQL query logging during development
    future=True,
    # Connection pool settings for production
    pool_size=5,  # Number of connections to maintain
    max_overflow=10,  # Additional connections allowed beyond pool_size
    pool_timeout=30,  # Timeout waiting for connection from pool (seconds)
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_pre_ping=True,  # Verify connections before using them
    # Connection arguments
    connect_args={
        "timeout": 10,  # Connection timeout (seconds)
        "command_timeout": 60,  # Query timeout (seconds)
    },
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,  # Manual flush for better transaction control
)

logger.info("Database engine initialized with connection pooling")


async def get_session() -> AsyncSession:
    """Get database session dependency for FastAPI.

    Yields:
        AsyncSession for database operations

    Example:
        @app.get("/events")
        async def get_events(session: AsyncSession = Depends(get_session)):
            result = await session.execute(select(GitHubEvent))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        yield session


async def check_database_connection() -> bool:
    """Check if database connection is healthy.

    Returns:
        True if connection is working, False otherwise

    Example:
        if await check_database_connection():
            logger.info("Database is healthy")
        else:
            logger.error("Database connection failed")
    """
    try:
        from sqlalchemy import text

        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database health check: OK")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def close_database() -> None:
    """Close database connections gracefully.

    Should be called during application shutdown.

    Example:
        @app.on_event("shutdown")
        async def shutdown():
            await close_database()
    """
    logger.info("Closing database connections...")
    await engine.dispose()
    logger.info("Database connections closed")
