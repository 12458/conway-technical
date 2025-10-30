"""Configuration settings for the GitHub Events API client."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration settings for GitHub API client."""

    github_api_base_url: str = Field(
        default="https://api.github.com",
        description="Base URL for GitHub API",
    )

    github_api_version: str = Field(
        default="2022-11-28",
        description="GitHub API version",
    )

    user_agent: str = Field(
        default="GitHubEventsClient/1.0",
        description="User agent string for API requests",
    )

    default_per_page: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Default number of results per page (1-100)",
    )

    timeout: float = Field(
        default=30.0,
        gt=0,
        description="HTTP request timeout in seconds",
    )


# Global settings instance
settings = Settings()
