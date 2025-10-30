"""Unit tests for Pydantic models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from github_client.models import (
    Actor,
    CreateEventPayload,
    DeleteEventPayload,
    Event,
    EventsResponse,
    ForkEventPayload,
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


@pytest.mark.unit
class TestActorModel:
    """Tests for Actor model."""

    def test_actor_valid_data(self, sample_actor_data):
        """Test Actor model with valid data."""
        actor = Actor.model_validate(sample_actor_data)

        assert actor.id == 583231
        assert actor.login == "octocat"
        assert actor.display_login == "octocat"
        assert actor.gravatar_id == ""
        assert str(actor.url) == "https://api.github.com/users/octocat"
        assert "avatars.githubusercontent.com" in str(actor.avatar_url)

    def test_actor_missing_required_field(self, sample_actor_data):
        """Test Actor model fails with missing required field."""
        del sample_actor_data["login"]

        with pytest.raises(ValidationError) as exc_info:
            Actor.model_validate(sample_actor_data)

        assert "login" in str(exc_info.value)

    def test_actor_invalid_url(self, sample_actor_data):
        """Test Actor model fails with invalid URL."""
        sample_actor_data["url"] = "not-a-valid-url"

        with pytest.raises(ValidationError) as exc_info:
            Actor.model_validate(sample_actor_data)

        assert "url" in str(exc_info.value).lower()


@pytest.mark.unit
class TestRepoModel:
    """Tests for Repo model."""

    def test_repo_valid_data(self, sample_repo_data):
        """Test Repo model with valid data."""
        repo = Repo.model_validate(sample_repo_data)

        assert repo.id == 1296269
        assert repo.name == "octocat/Hello-World"
        assert str(repo.url) == "https://api.github.com/repos/octocat/Hello-World"

    def test_repo_missing_required_field(self, sample_repo_data):
        """Test Repo model fails with missing required field."""
        del sample_repo_data["name"]

        with pytest.raises(ValidationError) as exc_info:
            Repo.model_validate(sample_repo_data)

        assert "name" in str(exc_info.value)


@pytest.mark.unit
class TestEventPayloads:
    """Tests for different event payload models."""

    def test_watch_event_payload(self):
        """Test WatchEventPayload model."""
        payload = WatchEventPayload.model_validate({"action": "started"})
        assert payload.action == "started"

    def test_watch_event_payload_invalid_action(self):
        """Test WatchEventPayload fails with invalid action."""
        with pytest.raises(ValidationError):
            WatchEventPayload.model_validate({"action": "invalid"})

    def test_push_event_payload(self):
        """Test PushEventPayload model."""
        data = {
            "repository_id": 123,
            "push_id": 456,
            "ref": "refs/heads/main",
            "head": "abc123",
            "before": "def456",
        }
        payload = PushEventPayload.model_validate(data)

        assert payload.repository_id == 123
        assert payload.push_id == 456
        assert payload.ref == "refs/heads/main"
        assert payload.head == "abc123"
        assert payload.before == "def456"

    def test_push_event_payload_with_optional_fields(self):
        """Test PushEventPayload with optional fields."""
        data = {
            "repository_id": 123,
            "push_id": 456,
            "ref": "refs/heads/main",
            "head": "abc123",
            "before": "def456",
            "size": 3,
            "distinct_size": 2,
            "commits": [{"sha": "abc123", "message": "Test commit"}],
        }
        payload = PushEventPayload.model_validate(data)

        assert payload.size == 3
        assert payload.distinct_size == 2
        assert len(payload.commits) == 1

    def test_create_event_payload(self):
        """Test CreateEventPayload model."""
        data = {
            "ref": "feature-branch",
            "ref_type": "branch",
            "master_branch": "main",
        }
        payload = CreateEventPayload.model_validate(data)

        assert payload.ref == "feature-branch"
        assert payload.ref_type == "branch"
        assert payload.master_branch == "main"

    def test_create_event_payload_tag(self):
        """Test CreateEventPayload with tag ref_type."""
        data = {"ref": "v1.0.0", "ref_type": "tag"}
        payload = CreateEventPayload.model_validate(data)
        assert payload.ref_type == "tag"

    def test_delete_event_payload(self):
        """Test DeleteEventPayload model."""
        data = {"ref": "old-branch", "ref_type": "branch"}
        payload = DeleteEventPayload.model_validate(data)

        assert payload.ref == "old-branch"
        assert payload.ref_type == "branch"

    def test_fork_event_payload(self):
        """Test ForkEventPayload model."""
        data = {"forkee": {"id": 789, "name": "forked-repo"}}
        payload = ForkEventPayload.model_validate(data)
        assert payload.forkee["id"] == 789

    def test_issues_event_payload(self):
        """Test IssuesEventPayload model."""
        data = {"action": "opened", "issue": {"number": 123}}
        payload = IssuesEventPayload.model_validate(data)

        assert payload.action == "opened"
        assert payload.issue["number"] == 123

    def test_pull_request_event_payload(self):
        """Test PullRequestEventPayload model."""
        data = {"action": "opened", "number": 42, "pull_request": {"id": 100}}
        payload = PullRequestEventPayload.model_validate(data)

        assert payload.action == "opened"
        assert payload.number == 42

    def test_issue_comment_event_payload(self):
        """Test IssueCommentEventPayload model."""
        data = {
            "action": "created",
            "issue": {"number": 10},
            "comment": {"id": 20},
        }
        payload = IssueCommentEventPayload.model_validate(data)
        assert payload.action == "created"

    def test_release_event_payload(self):
        """Test ReleaseEventPayload model."""
        data = {"action": "published", "release": {"tag_name": "v1.0.0"}}
        payload = ReleaseEventPayload.model_validate(data)
        assert payload.action == "published"

    def test_member_event_payload(self):
        """Test MemberEventPayload model."""
        data = {"action": "added", "member": {"login": "newuser"}}
        payload = MemberEventPayload.model_validate(data)
        assert payload.action == "added"

    def test_public_event_payload(self):
        """Test PublicEventPayload model."""
        payload = PublicEventPayload.model_validate({})
        assert payload is not None

    def test_gollum_event_payload(self):
        """Test GollumEventPayload model."""
        data = {"pages": [{"page_name": "Home", "action": "edited"}]}
        payload = GollumEventPayload.model_validate(data)
        assert len(payload.pages) == 1


@pytest.mark.unit
class TestEventModel:
    """Tests for Event model."""

    def test_event_valid_data(self, sample_watch_event_data):
        """Test Event model with valid data."""
        event = Event.model_validate(sample_watch_event_data)

        assert event.id == "22249084947"
        assert event.type == "WatchEvent"
        assert event.actor.login == "octocat"
        assert event.repo.name == "octocat/Hello-World"
        assert event.payload == {"action": "started"}
        assert event.public is True
        assert isinstance(event.created_at, datetime)

    def test_event_with_push_payload(self, sample_push_event_data):
        """Test Event model with PushEvent payload."""
        event = Event.model_validate(sample_push_event_data)

        assert event.type == "PushEvent"
        assert "push_id" in event.payload
        assert event.payload["push_id"] == 10115855396

    def test_event_datetime_parsing(self, sample_watch_event_data):
        """Test Event model correctly parses datetime."""
        event = Event.model_validate(sample_watch_event_data)

        assert isinstance(event.created_at, datetime)
        assert event.created_at.year == 2022
        assert event.created_at.month == 6
        assert event.created_at.day == 9

    def test_event_with_org(self, sample_watch_event_data):
        """Test Event model with optional org field."""
        sample_watch_event_data["org"] = {"id": 999, "login": "testorg"}
        event = Event.model_validate(sample_watch_event_data)

        assert event.org is not None
        assert event.org["login"] == "testorg"

    def test_event_extra_fields_forbidden(self, sample_watch_event_data):
        """Test Event model rejects extra fields."""
        sample_watch_event_data["extra_field"] = "should_fail"

        with pytest.raises(ValidationError) as exc_info:
            Event.model_validate(sample_watch_event_data)

        assert "extra_field" in str(exc_info.value).lower()


@pytest.mark.unit
class TestEventsResponse:
    """Tests for EventsResponse model."""

    def test_events_response_with_data(self, sample_events_list, mock_etag, mock_poll_interval):
        """Test EventsResponse model with all fields."""
        events = [Event.model_validate(e) for e in sample_events_list]
        response = EventsResponse(
            events=events,
            etag=mock_etag,
            poll_interval=mock_poll_interval,
        )

        assert len(response.events) == 2
        assert response.etag == mock_etag
        assert response.poll_interval == 60

    def test_events_response_empty_events(self):
        """Test EventsResponse with empty events list."""
        response = EventsResponse(events=[], etag=None, poll_interval=None)

        assert len(response.events) == 0
        assert response.etag is None
        assert response.poll_interval is None

    def test_events_response_with_only_etag(self, mock_etag):
        """Test EventsResponse with only etag (304 response scenario)."""
        response = EventsResponse(events=[], etag=mock_etag, poll_interval=60)

        assert len(response.events) == 0
        assert response.etag == mock_etag
        assert response.poll_interval == 60
