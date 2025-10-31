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
    Commit,
    CommitComment,
    CommitCommentEventPayload,
    Comment,
    CreateEventPayload,
    DeleteEventPayload,
    Discussion,
    DiscussionEventPayload,
    Event,
    EventsResponse,
    ForkEventPayload,
    GenericEventPayload,
    GollumEventPayload,
    Issue,
    IssueCommentEventPayload,
    IssuesEventPayload,
    Label,
    MemberEventPayload,
    Organization,
    PublicEventPayload,
    PullRequest,
    PullRequestEventPayload,
    PullRequestReviewCommentEventPayload,
    PullRequestReviewEventPayload,
    PushEventPayload,
    Release,
    ReleaseEventPayload,
    Repo,
    Repository,
    Review,
    User,
    WatchEventPayload,
    WikiPage,
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
    # Base Models
    "Actor",
    "Repo",
    "Event",
    "EventsResponse",
    # Nested/Shared Models
    "Organization",
    "User",
    "Label",
    "Issue",
    "Comment",
    "CommitComment",
    "PullRequest",
    "Review",
    "Repository",
    "Release",
    "Discussion",
    "WikiPage",
    "Commit",
    # Event Payload Models
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
    "CommitCommentEventPayload",
    "DiscussionEventPayload",
    "PullRequestReviewEventPayload",
    "PullRequestReviewCommentEventPayload",
    "GenericEventPayload",
]

__version__ = "1.0.0"
