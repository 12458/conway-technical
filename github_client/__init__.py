"""GitHub Events API Client.

A self-contained module for interacting with GitHub's public events REST API.
"""

from .client import GitHubEventsClient
from .config import Settings, settings
from .exceptions import (
    ForbiddenError,
    GitHubAPIError,
    RateLimitError,
    ServiceUnavailableError,
    ValidationError,
)
from .models import (
    Actor,
    CreateEventPayload,
    DeleteEventPayload,
    Event,
    EventsResponse,
    ForkEventPayload,
    GenericEventPayload,
    GollumEventPayload,
    IssueCommentEventPayload,
    IssuesEventPayload,
    MemberEventPayload,
    PublicEventPayload,
    PullRequestEventPayload,
    PushEventPayload,
    ReleaseEventPayload,
    Repo,
    WatchEventPayload,
)

__all__ = [
    # Client
    "GitHubEventsClient",
    # Config
    "Settings",
    "settings",
    # Exceptions
    "GitHubAPIError",
    "RateLimitError",
    "ForbiddenError",
    "ServiceUnavailableError",
    "ValidationError",
    # Models
    "Actor",
    "Repo",
    "Event",
    "EventsResponse",
    # Payload models
    "WatchEventPayload",
    "PushEventPayload",
    "CreateEventPayload",
    "DeleteEventPayload",
    "ForkEventPayload",
    "IssuesEventPayload",
    "PullRequestEventPayload",
    "IssueCommentEventPayload",
    "ReleaseEventPayload",
    "MemberEventPayload",
    "PublicEventPayload",
    "GollumEventPayload",
    "GenericEventPayload",
]

__version__ = "1.0.0"
