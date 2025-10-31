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


class Organization(BaseModel):
    """GitHub organization information."""

    id: int = Field(description="Organization ID")
    login: str = Field(description="Organization name")
    gravatar_id: str = Field(description="Gravatar ID")
    url: HttpUrl = Field(description="API URL for the organization")
    avatar_url: HttpUrl = Field(description="Avatar image URL")


class User(BaseModel):
    """GitHub user information (for members, assignees, etc.)."""

    id: int = Field(description="User ID")
    login: str = Field(description="Username")
    gravatar_id: str | None = Field(None, description="Gravatar ID")
    url: HttpUrl = Field(description="API URL for the user")
    avatar_url: HttpUrl = Field(description="Avatar image URL")
    type: str | None = Field(None, description="User type (User, Bot, etc.)")


class Label(BaseModel):
    """GitHub issue/PR label."""

    id: int = Field(description="Label ID")
    name: str = Field(description="Label name")
    color: str = Field(description="Label color (hex)")
    url: HttpUrl | None = Field(None, description="API URL for the label")
    default: bool | None = Field(None, description="Whether this is a default label")


class Issue(BaseModel):
    """GitHub issue information."""

    id: int = Field(description="Issue ID")
    number: int = Field(description="Issue number")
    title: str = Field(description="Issue title")
    state: str = Field(description="Issue state (open, closed)")
    body: str | None = Field(None, description="Issue body text")
    user: User = Field(description="User who created the issue")
    labels: list[Label] | None = Field(None, description="Issue labels")
    assignee: User | None = Field(None, description="Assigned user")
    assignees: list[User] | None = Field(None, description="All assigned users")
    url: HttpUrl = Field(description="API URL for the issue")
    html_url: HttpUrl = Field(description="Web URL for the issue")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    closed_at: datetime | None = Field(None, description="Closure timestamp")


class Comment(BaseModel):
    """GitHub comment (issue or commit comment)."""

    id: int = Field(description="Comment ID")
    body: str = Field(description="Comment text")
    user: User = Field(description="User who created the comment")
    url: HttpUrl = Field(description="API URL for the comment")
    html_url: HttpUrl = Field(description="Web URL for the comment")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


class CommitComment(BaseModel):
    """GitHub commit comment."""

    id: int = Field(description="Comment ID")
    body: str = Field(description="Comment text")
    user: User = Field(description="User who created the comment")
    commit_id: str = Field(description="SHA of the commit")
    path: str | None = Field(None, description="File path if comment is on a file")
    position: int | None = Field(None, description="Position in the diff")
    line: int | None = Field(None, description="Line number in the file")
    url: HttpUrl = Field(description="API URL for the comment")
    html_url: HttpUrl = Field(description="Web URL for the comment")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


class PullRequest(BaseModel):
    """GitHub pull request information."""

    id: int = Field(description="Pull request ID")
    number: int = Field(description="Pull request number")
    title: str = Field(description="Pull request title")
    state: str = Field(description="Pull request state (open, closed)")
    body: str | None = Field(None, description="Pull request body text")
    user: User = Field(description="User who created the pull request")
    labels: list[Label] | None = Field(None, description="Pull request labels")
    assignee: User | None = Field(None, description="Assigned user")
    assignees: list[User] | None = Field(None, description="All assigned users")
    merged: bool | None = Field(None, description="Whether PR is merged")
    url: HttpUrl = Field(description="API URL for the pull request")
    html_url: HttpUrl = Field(description="Web URL for the pull request")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    closed_at: datetime | None = Field(None, description="Closure timestamp")
    merged_at: datetime | None = Field(None, description="Merge timestamp")


class Review(BaseModel):
    """GitHub pull request review."""

    id: int = Field(description="Review ID")
    body: str | None = Field(None, description="Review comment text")
    user: User = Field(description="User who created the review")
    state: str = Field(description="Review state (APPROVED, CHANGES_REQUESTED, COMMENTED)")
    commit_id: str = Field(description="SHA of the commit being reviewed")
    url: HttpUrl = Field(description="API URL for the review")
    html_url: HttpUrl = Field(description="Web URL for the review")
    submitted_at: datetime | None = Field(None, description="Submission timestamp")


class Repository(BaseModel):
    """Detailed GitHub repository information (for forkee, etc.)."""

    id: int = Field(description="Repository ID")
    name: str = Field(description="Repository name")
    full_name: str = Field(description="Full repository name in 'owner/repo' format")
    owner: User = Field(description="Repository owner")
    private: bool = Field(description="Whether repository is private")
    description: str | None = Field(None, description="Repository description")
    fork: bool = Field(description="Whether this is a fork")
    url: HttpUrl = Field(description="API URL for the repository")
    html_url: HttpUrl = Field(description="Web URL for the repository")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    pushed_at: datetime | None = Field(None, description="Last push timestamp")
    default_branch: str | None = Field(None, description="Default branch name")


class Release(BaseModel):
    """GitHub release information."""

    id: int = Field(description="Release ID")
    tag_name: str = Field(description="Git tag name")
    name: str | None = Field(None, description="Release name")
    body: str | None = Field(None, description="Release description")
    draft: bool = Field(description="Whether this is a draft release")
    prerelease: bool = Field(description="Whether this is a prerelease")
    author: User = Field(description="User who created the release")
    url: HttpUrl = Field(description="API URL for the release")
    html_url: HttpUrl = Field(description="Web URL for the release")
    created_at: datetime = Field(description="Creation timestamp")
    published_at: datetime | None = Field(None, description="Publication timestamp")


class Discussion(BaseModel):
    """GitHub discussion information."""

    id: int = Field(description="Discussion ID")
    number: int = Field(description="Discussion number")
    title: str = Field(description="Discussion title")
    body: str | None = Field(None, description="Discussion body text")
    user: User = Field(description="User who created the discussion")
    state: str = Field(description="Discussion state")
    url: HttpUrl = Field(description="API URL for the discussion")
    html_url: HttpUrl = Field(description="Web URL for the discussion")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


class WikiPage(BaseModel):
    """GitHub wiki page information."""

    page_name: str = Field(description="Page name")
    title: str = Field(description="Page title")
    summary: str | None = Field(None, description="Optional page summary")
    action: Literal["created", "edited"] = Field(description="Action performed on the page")
    sha: str = Field(description="Latest commit SHA of the page")
    html_url: HttpUrl = Field(description="Web URL for the wiki page")


class Commit(BaseModel):
    """Git commit information."""

    sha: str = Field(description="Commit SHA")
    message: str = Field(description="Commit message")
    author: dict[str, Any] = Field(description="Commit author information")
    url: HttpUrl = Field(description="API URL for the commit")
    distinct: bool | None = Field(None, description="Whether commit is distinct")


# Event Payload Models for different event types


class WatchEventPayload(BaseModel):
    """Payload for WatchEvent (repository starred)."""

    action: Literal["started"] = Field(description="Action type for watch event")


class PushEventPayload(BaseModel):
    """Payload for PushEvent (commits pushed)."""

    repository_id: int = Field(description="Repository ID")
    push_id: int = Field(description="Unique push ID")
    ref: str = Field(description="Git reference (e.g., refs/heads/main)")
    head: str = Field(description="SHA of the HEAD commit after push")
    before: str = Field(description="SHA of the commit before push")
    size: int | None = Field(None, description="Number of commits in the push")
    distinct_size: int | None = Field(None, description="Number of distinct commits")
    commits: list[Commit] | None = Field(None, description="List of commits in the push")


class CreateEventPayload(BaseModel):
    """Payload for CreateEvent (branch or tag created)."""

    ref: str | None = Field(None, description="Git ref name (branch or null if repository)")
    ref_type: Literal["branch", "tag", "repository"] = Field(
        description="Type of ref created"
    )
    full_ref: str | None = Field(
        None, description="Fully-formed ref (e.g., refs/heads/<branch_name>)"
    )
    master_branch: str | None = Field(None, description="Default branch name")
    description: str | None = Field(None, description="Repository description")
    pusher_type: str | None = Field(None, description="Type of pusher (user/deploy key)")


class DeleteEventPayload(BaseModel):
    """Payload for DeleteEvent (branch or tag deleted)."""

    ref: str = Field(description="Git ref name that was deleted")
    ref_type: Literal["branch", "tag"] = Field(description="Type of ref deleted")
    full_ref: str | None = Field(
        None, description="Fully-formed ref (e.g., refs/heads/<branch_name>)"
    )
    pusher_type: str | None = Field(None, description="Type of pusher")


class ForkEventPayload(BaseModel):
    """Payload for ForkEvent (repository forked)."""

    action: Literal["forked"] | None = Field(None, description="Action performed")
    forkee: Repository = Field(description="The created fork repository")


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
    issue: Issue = Field(description="The issue")
    assignee: User | None = Field(None, description="User assigned/unassigned (if applicable)")
    assignees: list[User] | None = Field(
        None, description="Array of assignee objects (if applicable)"
    )
    label: Label | None = Field(None, description="Label added/removed (if applicable)")
    labels: list[Label] | None = Field(None, description="Array of label objects")


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
    pull_request: PullRequest = Field(description="The pull request")
    assignee: User | None = Field(None, description="User assigned/unassigned (if applicable)")
    assignees: list[User] | None = Field(
        None, description="Array of assignee objects (if applicable)"
    )
    label: Label | None = Field(None, description="Label added/removed (if applicable)")
    labels: list[Label] | None = Field(None, description="Array of label objects")


class IssueCommentEventPayload(BaseModel):
    """Payload for IssueCommentEvent."""

    action: Literal["created", "edited", "deleted"] = Field(
        description="Action on the comment"
    )
    issue: Issue = Field(description="The issue the comment belongs to")
    comment: Comment = Field(description="The comment itself")


class ReleaseEventPayload(BaseModel):
    """Payload for ReleaseEvent."""

    action: Literal["published", "unpublished", "created", "edited", "deleted"] = Field(
        description="Action performed on the release"
    )
    release: Release = Field(description="The release object")


class MemberEventPayload(BaseModel):
    """Payload for MemberEvent (collaborator added)."""

    action: Literal["added", "deleted", "edited"] = Field(
        description="Action performed"
    )
    member: User = Field(description="The user that was added/removed")


class PublicEventPayload(BaseModel):
    """Payload for PublicEvent (repository made public). Typically empty."""

    pass


class GollumEventPayload(BaseModel):
    """Payload for GollumEvent (wiki page created/updated)."""

    pages: list[WikiPage] = Field(description="Wiki pages that were updated")


class CommitCommentEventPayload(BaseModel):
    """Payload for CommitCommentEvent (commit comment created)."""

    action: Literal["created"] = Field(description="Action performed on the comment")
    comment: CommitComment = Field(description="The commit comment resource")


class DiscussionEventPayload(BaseModel):
    """Payload for DiscussionEvent (discussion created)."""

    action: Literal["created"] = Field(description="Action performed")
    discussion: Discussion = Field(description="The discussion that was created")


class PullRequestReviewEventPayload(BaseModel):
    """Payload for PullRequestReviewEvent."""

    action: Literal["created", "edited", "dismissed"] = Field(
        description="Action performed on the review"
    )
    pull_request: PullRequest = Field(description="The pull request the review pertains to")
    review: Review = Field(description="The review that was affected")


class PullRequestReviewCommentEventPayload(BaseModel):
    """Payload for PullRequestReviewCommentEvent."""

    action: Literal["created"] = Field(description="Action performed on the comment")
    pull_request: PullRequest = Field(description="The pull request the comment belongs to")
    comment: Comment = Field(description="The comment itself")


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
        CommitCommentEventPayload,
        DiscussionEventPayload,
        PullRequestReviewEventPayload,
        PullRequestReviewCommentEventPayload,
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
    org: Organization | None = Field(
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
