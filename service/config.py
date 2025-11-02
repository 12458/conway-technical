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

    # Velocity-based anomaly detection settings
    velocity_time_window: int = Field(
        default=300,
        description="Time window in seconds for velocity detection (default: 5 minutes)",
    )
    velocity_threshold_per_min: float = Field(
        default=20.0,
        description="Events per minute threshold for 'inhuman speed' detection",
    )
    max_timestamps_per_actor: int = Field(
        default=200,
        description="Maximum number of timestamps to store per actor (memory limit)",
    )

    # Feature extraction settings
    max_actors_tracked: int = Field(
        default=50_000,
        description="Maximum number of actors to track in feature extractor (LRU eviction)",
    )
    max_repos_tracked: int = Field(
        default=25_000,
        description="Maximum number of repositories to track in feature extractor (LRU eviction)",
    )
    max_text_bytes: int = Field(
        default=2048,
        description="Maximum text bytes to process for feature extraction (2KB limit)",
    )
    known_bots_config_path: str = Field(
        default="config/known_bots.json",
        description="Path to JSON file containing list of known bot accounts to filter",
    )

    # Event-type-specific RRCF forest settings
    enable_multi_forest: bool = Field(
        default=True,
        description="Enable separate RRCF forests per event type group",
    )
    event_type_forest_groups: dict[str, list[str]] = Field(
        default={
            "push": ["PushEvent"],
            "pull_request": [
                "PullRequestEvent",
                "PullRequestReviewEvent",
                "PullRequestReviewCommentEvent",
            ],
            "issues": ["IssuesEvent", "IssueCommentEvent"],
            "security": ["MemberEvent", "DeleteEvent", "PublicEvent"],
            "other": [
                "WatchEvent",
                "ForkEvent",
                "CreateEvent",
                "ReleaseEvent",
                "GollumEvent",
                "CommitCommentEvent",
                "DiscussionEvent",
            ],
        },
        description="Event type grouping for separate RRCF forests. Each group gets its own forest.",
    )

    # Temporal burst detection settings
    burst_window_seconds: int = Field(
        default=30,
        description="Time window in seconds for burst detection (default: 30 seconds)",
    )
    burst_threshold_events: int = Field(
        default=5,
        description="Minimum events in burst window to count as burst",
    )
    burst_tracking_window: int = Field(
        default=3600,
        description="How far back to track bursts in seconds (default: 1 hour)",
    )

    # Repo hopping time window
    repo_hopping_time_window: int = Field(
        default=300,
        description="Time window in seconds for repo hopping detection (default: 5 minutes)",
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
