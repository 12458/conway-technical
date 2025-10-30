"""Pydantic models for GitHub Events API."""

from datetime import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, HttpUrl


class Actor(BaseModel):
    """GitHub user actor information."""

    id: int = Field(description="User ID")
    login: str = Field(description="Username")
    display_login: str = Field(description="Display username")
    gravatar_id: str = Field(description="Gravatar ID")
    url: HttpUrl = Field(description="API URL for the user")
    avatar_url: HttpUrl = Field(description="Avatar image URL")


class Repo(BaseModel):
    """GitHub repository information."""

    id: int = Field(description="Repository ID")
    name: str = Field(description="Repository name in 'owner/repo' format")
    url: HttpUrl = Field(description="API URL for the repository")


# Event Payload Models for different event types


class WatchEventPayload(BaseModel):
    """Payload for WatchEvent (repository starred)."""

    action: Literal["started"] = Field(description="Action type for watch event")


class PushEventPayload(BaseModel):
    """Payload for PushEvent (commits pushed)."""

    repository_id: int = Field(description="Repository ID")
    push_id: int = Field(description="Unique push ID")
    ref: str = Field(description="Git reference (e.g., refs/heads/master)")
    head: str = Field(description="SHA of the HEAD commit after push")
    before: str = Field(description="SHA of the commit before push")
    size: int | None = Field(None, description="Number of commits in the push")
    distinct_size: int | None = Field(None, description="Number of distinct commits")
    commits: list[dict[str, Any]] | None = Field(
        None, description="List of commits in the push"
    )


class CreateEventPayload(BaseModel):
    """Payload for CreateEvent (branch or tag created)."""

    ref: str | None = Field(None, description="Git ref name")
    ref_type: Literal["branch", "tag", "repository"] = Field(
        description="Type of ref created"
    )
    master_branch: str | None = Field(None, description="Default branch name")
    description: str | None = Field(None, description="Repository description")
    pusher_type: str | None = Field(None, description="Type of pusher (user/deploy key)")


class DeleteEventPayload(BaseModel):
    """Payload for DeleteEvent (branch or tag deleted)."""

    ref: str = Field(description="Git ref name that was deleted")
    ref_type: Literal["branch", "tag"] = Field(description="Type of ref deleted")
    pusher_type: str | None = Field(None, description="Type of pusher")


class ForkEventPayload(BaseModel):
    """Payload for ForkEvent (repository forked)."""

    forkee: dict[str, Any] = Field(description="The created fork repository")


class IssuesEventPayload(BaseModel):
    """Payload for IssuesEvent (issue opened, edited, closed, etc)."""

    action: Literal[
        "opened",
        "edited",
        "deleted",
        "closed",
        "reopened",
        "assigned",
        "unassigned",
        "labeled",
        "unlabeled",
    ] = Field(description="Action performed on the issue")
    issue: dict[str, Any] = Field(description="The issue")


class PullRequestEventPayload(BaseModel):
    """Payload for PullRequestEvent."""

    action: Literal[
        "opened",
        "edited",
        "closed",
        "reopened",
        "assigned",
        "unassigned",
        "review_requested",
        "review_request_removed",
        "labeled",
        "unlabeled",
        "synchronize",
    ] = Field(description="Action performed on the pull request")
    number: int = Field(description="Pull request number")
    pull_request: dict[str, Any] = Field(description="The pull request")


class IssueCommentEventPayload(BaseModel):
    """Payload for IssueCommentEvent."""

    action: Literal["created", "edited", "deleted"] = Field(
        description="Action on the comment"
    )
    issue: dict[str, Any] = Field(description="The issue")
    comment: dict[str, Any] = Field(description="The comment")


class ReleaseEventPayload(BaseModel):
    """Payload for ReleaseEvent."""

    action: Literal["published", "unpublished", "created", "edited", "deleted"] = Field(
        description="Action performed on the release"
    )
    release: dict[str, Any] = Field(description="The release")


class MemberEventPayload(BaseModel):
    """Payload for MemberEvent (collaborator added)."""

    action: Literal["added", "deleted", "edited"] = Field(
        description="Action performed"
    )
    member: dict[str, Any] = Field(description="The user that was added/removed")


class PublicEventPayload(BaseModel):
    """Payload for PublicEvent (repository made public). Typically empty."""

    pass


class GollumEventPayload(BaseModel):
    """Payload for GollumEvent (wiki page created/updated)."""

    pages: list[dict[str, Any]] = Field(description="Wiki pages that were updated")


# Generic payload for unknown event types
class GenericEventPayload(BaseModel):
    """Generic payload for event types not explicitly defined."""

    pass

    model_config = {"extra": "allow"}


# Discriminated union for all payload types
EventPayload = Annotated[
    Union[
        WatchEventPayload,
        PushEventPayload,
        CreateEventPayload,
        DeleteEventPayload,
        ForkEventPayload,
        IssuesEventPayload,
        PullRequestEventPayload,
        IssueCommentEventPayload,
        ReleaseEventPayload,
        MemberEventPayload,
        PublicEventPayload,
        GollumEventPayload,
        GenericEventPayload,
    ],
    Field(discriminator=None),
]


class Event(BaseModel):
    """GitHub Event model."""

    id: str = Field(description="Event ID")
    type: str = Field(description="Event type (e.g., WatchEvent, PushEvent)")
    actor: Actor = Field(description="User who triggered the event")
    repo: Repo = Field(description="Repository where event occurred")
    payload: dict[str, Any] = Field(
        description="Event payload (varies by event type)"
    )
    public: bool = Field(description="Whether the event is public")
    created_at: datetime = Field(description="Event creation timestamp")
    org: dict[str, Any] | None = Field(
        None, description="Organization (if applicable)"
    )

    model_config = {"extra": "forbid"}


class EventsResponse(BaseModel):
    """Response wrapper for list of events."""

    events: list[Event] = Field(description="List of GitHub events")
    etag: str | None = Field(None, description="ETag for caching")
    poll_interval: int | None = Field(
        None, description="Recommended polling interval in seconds"
    )
