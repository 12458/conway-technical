"""Unit tests for Pydantic models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from github_client.models import (
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

    # test_push_event_payload_with_optional_fields removed because GitHub Events API
    # does not provide size, distinct_size, or commits fields

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

    def test_fork_event_payload(self, sample_repository_data):
        """Test ForkEventPayload model."""
        data = {"forkee": sample_repository_data}
        payload = ForkEventPayload.model_validate(data)
        assert payload.forkee.id == 1296269

    def test_issues_event_payload(self, sample_issue_data):
        """Test IssuesEventPayload model."""
        data = {"action": "opened", "issue": sample_issue_data}
        payload = IssuesEventPayload.model_validate(data)

        assert payload.action == "opened"
        assert payload.issue.number == 1347

    def test_pull_request_event_payload(self, sample_pull_request_data):
        """Test PullRequestEventPayload model."""
        data = {
            "action": "opened",
            "number": 42,
            "pull_request": sample_pull_request_data,
        }
        payload = PullRequestEventPayload.model_validate(data)

        assert payload.action == "opened"
        assert payload.number == 42

    def test_issue_comment_event_payload(self, sample_issue_data, sample_comment_data):
        """Test IssueCommentEventPayload model."""
        data = {
            "action": "created",
            "issue": sample_issue_data,
            "comment": sample_comment_data,
        }
        payload = IssueCommentEventPayload.model_validate(data)
        assert payload.action == "created"

    def test_release_event_payload(self, sample_release_data):
        """Test ReleaseEventPayload model."""
        data = {"action": "published", "release": sample_release_data}
        payload = ReleaseEventPayload.model_validate(data)
        assert payload.action == "published"

    def test_member_event_payload(self, sample_user_data):
        """Test MemberEventPayload model."""
        data = {"action": "added", "member": sample_user_data}
        payload = MemberEventPayload.model_validate(data)
        assert payload.action == "added"

    def test_public_event_payload(self):
        """Test PublicEventPayload model."""
        payload = PublicEventPayload.model_validate({})
        assert payload is not None

    def test_gollum_event_payload(self, sample_wiki_page_data):
        """Test GollumEventPayload model."""
        data = {"pages": [sample_wiki_page_data]}
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

    def test_event_with_org(self, sample_watch_event_data, sample_organization_data):
        """Test Event model with optional org field."""
        sample_watch_event_data["org"] = sample_organization_data
        event = Event.model_validate(sample_watch_event_data)

        assert event.org is not None
        assert event.org.login == "github"

    def test_event_extra_fields_forbidden(self, sample_watch_event_data):
        """Test Event model rejects extra fields."""
        sample_watch_event_data["extra_field"] = "should_fail"

        with pytest.raises(ValidationError) as exc_info:
            Event.model_validate(sample_watch_event_data)

        assert "extra_field" in str(exc_info.value).lower()


@pytest.mark.unit
class TestEventsResponse:
    """Tests for EventsResponse model."""

    def test_events_response_with_data(
        self, sample_events_list, mock_etag, mock_poll_interval
    ):
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


@pytest.mark.unit
class TestOrganizationModel:
    """Tests for Organization model."""

    def test_organization_valid_data(self, sample_organization_data):
        """Test Organization model with valid data."""
        org = Organization.model_validate(sample_organization_data)

        assert org.id == 1
        assert org.login == "github"
        assert org.gravatar_id == ""
        assert str(org.url) == "https://api.github.com/orgs/github"
        assert "avatars.githubusercontent.com" in str(org.avatar_url)

    def test_organization_missing_required_field(self, sample_organization_data):
        """Test Organization model fails with missing required field."""
        del sample_organization_data["login"]

        with pytest.raises(ValidationError) as exc_info:
            Organization.model_validate(sample_organization_data)

        assert "login" in str(exc_info.value)


@pytest.mark.unit
class TestUserModel:
    """Tests for User model."""

    def test_user_valid_data(self, sample_user_data):
        """Test User model with valid data."""
        user = User.model_validate(sample_user_data)

        assert user.id == 1
        assert user.login == "octocat"
        assert user.type == "User"
        assert str(user.url) == "https://api.github.com/users/octocat"

    def test_user_optional_fields(self, sample_user_data):
        """Test User model with optional fields as None."""
        sample_user_data["gravatar_id"] = None
        sample_user_data["type"] = None
        user = User.model_validate(sample_user_data)

        assert user.gravatar_id is None
        assert user.type is None


@pytest.mark.unit
class TestLabelModel:
    """Tests for Label model."""

    def test_label_valid_data(self, sample_label_data):
        """Test Label model with valid data."""
        label = Label.model_validate(sample_label_data)

        assert label.id == 208045946
        assert label.name == "bug"
        assert label.color == "d73a4a"
        assert label.default is True


@pytest.mark.unit
class TestIssueModel:
    """Tests for Issue model."""

    def test_issue_valid_data(self, sample_issue_data):
        """Test Issue model with valid data."""
        issue = Issue.model_validate(sample_issue_data)

        assert issue.id == 1
        assert issue.number == 1347
        assert issue.title == "Found a bug"
        assert issue.state == "open"
        assert issue.user.login == "octocat"
        assert len(issue.labels) == 1
        assert issue.labels[0].name == "bug"

    def test_issue_optional_fields(self, sample_issue_data):
        """Test Issue model with optional fields as None."""
        sample_issue_data["closed_at"] = None
        sample_issue_data["body"] = None
        issue = Issue.model_validate(sample_issue_data)

        assert issue.closed_at is None
        assert issue.body is None


@pytest.mark.unit
class TestCommentModel:
    """Tests for Comment model."""

    def test_comment_valid_data(self, sample_comment_data):
        """Test Comment model with valid data."""
        comment = Comment.model_validate(sample_comment_data)

        assert comment.id == 1
        assert comment.body == "This is a comment"
        assert comment.user.login == "octocat"


@pytest.mark.unit
class TestCommitCommentModel:
    """Tests for CommitComment model."""

    def test_commit_comment_valid_data(self, sample_commit_comment_data):
        """Test CommitComment model with valid data."""
        comment = CommitComment.model_validate(sample_commit_comment_data)

        assert comment.id == 1
        assert comment.body == "Great commit!"
        assert comment.commit_id == "7a8f3ac80e2ad2f6842cb86f576d4bfe2c03e300"
        assert comment.path == "file.py"
        assert comment.line == 10


@pytest.mark.unit
class TestPullRequestModel:
    """Tests for PullRequest model."""

    def test_pull_request_valid_data(self, sample_pull_request_data):
        """Test PullRequest model with valid data."""
        pr = PullRequest.model_validate(sample_pull_request_data)

        assert pr.id == 1
        assert pr.number == 42
        assert pr.title == "Amazing new feature"
        assert pr.state == "open"
        assert pr.merged is False
        assert pr.user.login == "octocat"


@pytest.mark.unit
class TestReviewModel:
    """Tests for Review model."""

    def test_review_valid_data(self, sample_review_data):
        """Test Review model with valid data."""
        review = Review.model_validate(sample_review_data)

        assert review.id == 1
        assert review.body == "Looks good to me"
        assert review.state == "APPROVED"
        assert review.user.login == "octocat"


@pytest.mark.unit
class TestRepositoryModel:
    """Tests for Repository model."""

    def test_repository_valid_data(self, sample_repository_data):
        """Test Repository model with valid data."""
        repo = Repository.model_validate(sample_repository_data)

        assert repo.id == 1296269
        assert repo.name == "Hello-World"
        assert repo.full_name == "octocat/Hello-World"
        assert repo.owner.login == "octocat"
        assert repo.private is False
        assert repo.fork is False


@pytest.mark.unit
class TestReleaseModel:
    """Tests for Release model."""

    def test_release_valid_data(self, sample_release_data):
        """Test Release model with valid data."""
        release = Release.model_validate(sample_release_data)

        assert release.id == 1
        assert release.tag_name == "v1.0.0"
        assert release.name == "Production Release"
        assert release.draft is False
        assert release.prerelease is False


@pytest.mark.unit
class TestDiscussionModel:
    """Tests for Discussion model."""

    def test_discussion_valid_data(self, sample_discussion_data):
        """Test Discussion model with valid data."""
        discussion = Discussion.model_validate(sample_discussion_data)

        assert discussion.id == 1
        assert discussion.number == 1
        assert discussion.title == "How do I do X?"
        assert discussion.state == "open"


@pytest.mark.unit
class TestWikiPageModel:
    """Tests for WikiPage model."""

    def test_wiki_page_valid_data(self, sample_wiki_page_data):
        """Test WikiPage model with valid data."""
        page = WikiPage.model_validate(sample_wiki_page_data)

        assert page.page_name == "Home"
        assert page.title == "Home Page"
        assert page.action == "edited"
        assert page.sha == "7a8f3ac80e2ad2f6842cb86f576d4bfe2c03e300"

    def test_wiki_page_action_validation(self, sample_wiki_page_data):
        """Test WikiPage validates action field."""
        sample_wiki_page_data["action"] = "invalid"

        with pytest.raises(ValidationError):
            WikiPage.model_validate(sample_wiki_page_data)


@pytest.mark.unit
class TestCommitModel:
    """Tests for Commit model."""

    def test_commit_valid_data(self, sample_commit_data):
        """Test Commit model with valid data."""
        commit = Commit.model_validate(sample_commit_data)

        assert commit.sha == "7a8f3ac80e2ad2f6842cb86f576d4bfe2c03e300"
        assert commit.message == "Fix bug"
        assert commit.distinct is True
        assert commit.author["name"] == "Octocat"


@pytest.mark.unit
class TestNewEventPayloads:
    """Tests for new event payload models."""

    def test_commit_comment_event_payload(self, sample_commit_comment_data):
        """Test CommitCommentEventPayload model."""
        data = {
            "action": "created",
            "comment": sample_commit_comment_data,
        }
        payload = CommitCommentEventPayload.model_validate(data)

        assert payload.action == "created"
        assert payload.comment.body == "Great commit!"

    def test_discussion_event_payload(self, sample_discussion_data):
        """Test DiscussionEventPayload model."""
        data = {
            "action": "created",
            "discussion": sample_discussion_data,
        }
        payload = DiscussionEventPayload.model_validate(data)

        assert payload.action == "created"
        assert payload.discussion.title == "How do I do X?"

    def test_pull_request_review_event_payload(
        self, sample_pull_request_data, sample_review_data
    ):
        """Test PullRequestReviewEventPayload model."""
        data = {
            "action": "created",
            "pull_request": sample_pull_request_data,
            "review": sample_review_data,
        }
        payload = PullRequestReviewEventPayload.model_validate(data)

        assert payload.action == "created"
        assert payload.pull_request.number == 42
        assert payload.review.state == "APPROVED"

    def test_pull_request_review_event_action_validation(
        self, sample_pull_request_data, sample_review_data
    ):
        """Test PullRequestReviewEventPayload validates action field."""
        data = {
            "action": "invalid",
            "pull_request": sample_pull_request_data,
            "review": sample_review_data,
        }

        with pytest.raises(ValidationError):
            PullRequestReviewEventPayload.model_validate(data)

    def test_pull_request_review_comment_event_payload(
        self, sample_pull_request_data, sample_comment_data
    ):
        """Test PullRequestReviewCommentEventPayload model."""
        data = {
            "action": "created",
            "pull_request": sample_pull_request_data,
            "comment": sample_comment_data,
        }
        payload = PullRequestReviewCommentEventPayload.model_validate(data)

        assert payload.action == "created"
        assert payload.pull_request.number == 42
        assert payload.comment.body == "This is a comment"


@pytest.mark.unit
class TestUpdatedEventPayloads:
    """Tests for updated event payload models with new fields."""

    def test_create_event_with_full_ref(self):
        """Test CreateEventPayload with full_ref field."""
        data = {
            "ref": "feature-branch",
            "ref_type": "branch",
            "full_ref": "refs/heads/feature-branch",
            "master_branch": "main",
        }
        payload = CreateEventPayload.model_validate(data)

        assert payload.full_ref == "refs/heads/feature-branch"

    def test_delete_event_with_full_ref(self):
        """Test DeleteEventPayload with full_ref field."""
        data = {
            "ref": "old-branch",
            "ref_type": "branch",
            "full_ref": "refs/heads/old-branch",
        }
        payload = DeleteEventPayload.model_validate(data)

        assert payload.full_ref == "refs/heads/old-branch"

    def test_fork_event_with_repository(self, sample_repository_data):
        """Test ForkEventPayload with Repository model."""
        data = {"forkee": sample_repository_data}
        payload = ForkEventPayload.model_validate(data)

        assert payload.forkee.full_name == "octocat/Hello-World"

    def test_issues_event_with_additional_fields(
        self, sample_issue_data, sample_user_data, sample_label_data
    ):
        """Test IssuesEventPayload with assignee and label fields."""
        data = {
            "action": "opened",
            "issue": sample_issue_data,
            "assignee": sample_user_data,
            "assignees": [sample_user_data],
            "label": sample_label_data,
            "labels": [sample_label_data],
        }
        payload = IssuesEventPayload.model_validate(data)

        assert payload.assignee.login == "octocat"
        assert len(payload.assignees) == 1
        assert payload.label.name == "bug"
        assert len(payload.labels) == 1

    def test_pull_request_event_with_additional_fields(
        self, sample_pull_request_data, sample_user_data, sample_label_data
    ):
        """Test PullRequestEventPayload with assignee and label fields."""
        data = {
            "action": "opened",
            "number": 42,
            "pull_request": sample_pull_request_data,
            "assignee": sample_user_data,
            "assignees": [sample_user_data],
            "label": sample_label_data,
            "labels": [sample_label_data],
        }
        payload = PullRequestEventPayload.model_validate(data)

        assert payload.pull_request.title == "Amazing new feature"
        assert payload.assignee.login == "octocat"
        assert payload.label.name == "bug"

    def test_issue_comment_event_with_models(
        self, sample_issue_data, sample_comment_data
    ):
        """Test IssueCommentEventPayload with typed models."""
        data = {
            "action": "created",
            "issue": sample_issue_data,
            "comment": sample_comment_data,
        }
        payload = IssueCommentEventPayload.model_validate(data)

        assert payload.issue.number == 1347
        assert payload.comment.body == "This is a comment"

    def test_release_event_with_model(self, sample_release_data):
        """Test ReleaseEventPayload with Release model."""
        data = {"action": "published", "release": sample_release_data}
        payload = ReleaseEventPayload.model_validate(data)

        assert payload.release.tag_name == "v1.0.0"

    def test_member_event_with_user_model(self, sample_user_data):
        """Test MemberEventPayload with User model."""
        data = {"action": "added", "member": sample_user_data}
        payload = MemberEventPayload.model_validate(data)

        assert payload.member.login == "octocat"

    def test_gollum_event_with_wiki_pages(self, sample_wiki_page_data):
        """Test GollumEventPayload with WikiPage models."""
        data = {"pages": [sample_wiki_page_data]}
        payload = GollumEventPayload.model_validate(data)

        assert len(payload.pages) == 1
        assert payload.pages[0].page_name == "Home"

    # test_push_event_with_commits removed because GitHub Events API
    # does not provide commits field


@pytest.mark.unit
class TestEventWithOrganization:
    """Tests for Event model with Organization."""

    def test_event_with_organization_model(
        self, sample_watch_event_data, sample_organization_data
    ):
        """Test Event model with Organization type for org field."""
        sample_watch_event_data["org"] = sample_organization_data
        event = Event.model_validate(sample_watch_event_data)

        assert event.org is not None
        assert event.org.login == "github"
        assert event.org.id == 1
