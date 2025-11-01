"""Configuration for the monitoring service."""

from pydantic import Field
from pydantic_settings import BaseSettings


class ServiceSettings(BaseSettings):
    """Service configuration settings."""

    # GitHub API settings
    github_token: str | None = Field(
        default=None,
        description="GitHub personal access token for authentication",
    )
    github_graphql_token: str | None = Field(
        default=None,
        description="GitHub token for GraphQL API enrichment (requires repo, read:org, read:user scopes)",
    )

    # Redis settings
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL for RQ",
    )

    # AI API settings
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for GPT summarization",
    )
    ai_provider: str = Field(
        default="openai",
        description="AI provider to use (openai only)",
    )
    ai_model: str = Field(
        default="gpt-4o-2024-08-06",
        description="OpenAI model to use for summarization (structured output support required)",
    )

    # Database settings
    database_url: str = Field(
        default="sqlite+aiosqlite:///./github_events.db",
        description="Database connection URL (supports PostgreSQL, SQLite)",
    )

    # Polling settings
    polling_interval: int = Field(
        default=60,
        description="Default polling interval in seconds",
    )
    max_backoff_seconds: int = Field(
        default=300,
        description="Maximum backoff time for rate limiting (5 minutes)",
    )
    backoff_multiplier: float = Field(
        default=2.0,
        description="Exponential backoff multiplier",
    )

    # Anomaly detection settings
    anomaly_threshold: float = Field(
        default=60.0,
        description="CoDisp score threshold for anomaly detection",
    )
    tree_size: int = Field(
        default=256,
        description="RRCF tree size for anomaly detection",
    )
    num_trees: int = Field(
        default=50,
        description="Number of RRCF trees in forest",
    )
    shingle_size: int = Field(
        default=1,
        description="Shingle size for streaming (1 = no shingling)",
    )

    # Queue settings
    queue_name: str = Field(
        default="github-anomalies",
        description="RQ queue name",
    )
    worker_count: int = Field(
        default=2,
        description="Number of RQ worker processes",
    )

    # Service settings
    max_events_per_fetch: int = Field(
        default=100,
        description="Maximum events to fetch per API call",
    )
    enable_bot_filtering: bool = Field(
        default=True,
        description="Filter out known bot accounts",
    )

    # GraphQL Enrichment settings
    enrichment_enabled: bool = Field(
        default=True,
        description="Enable GraphQL enrichment for detected anomalies",
    )
    enrichment_batch_size: int = Field(
        default=10,
        description="Maximum number of enrichments to batch together",
    )
    enrichment_timeout_ms: int = Field(
        default=5000,
        description="Timeout for enrichment operations in milliseconds",
    )
    graphql_api_url: str = Field(
        default="https://api.github.com/graphql",
        description="GitHub GraphQL API endpoint URL",
    )

    class Config:
        env_prefix = "SERVICE_"
        env_file = ".env"


# Global settings instance
service_settings = ServiceSettings()
