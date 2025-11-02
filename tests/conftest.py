"""Pytest fixtures for github_client tests."""

import gzip
import json

import pytest


@pytest.fixture
def sample_actor_data():
    """Sample actor data for testing."""
    return {
        "id": 583231,
        "login": "octocat",
        "display_login": "octocat",
        "gravatar_id": "",
        "url": "https://api.github.com/users/octocat",
        "avatar_url": "https://avatars.githubusercontent.com/u/583231?v=4",
    }


@pytest.fixture
def sample_repo_data():
    """Sample repository data for testing."""
    return {
        "id": 1296269,
        "name": "octocat/Hello-World",
        "url": "https://api.github.com/repos/octocat/Hello-World",
    }


@pytest.fixture
def sample_watch_event_data(sample_actor_data, sample_repo_data):
    """Sample WatchEvent data for testing."""
    return {
        "id": "22249084947",
        "type": "WatchEvent",
        "actor": sample_actor_data,
        "repo": sample_repo_data,
        "payload": {"action": "started"},
        "public": True,
        "created_at": "2022-06-09T12:47:28Z",
    }


@pytest.fixture
def sample_push_event_data(sample_actor_data, sample_repo_data):
    """Sample PushEvent data for testing."""
    return {
        "id": "22249084964",
        "type": "PushEvent",
        "actor": sample_actor_data,
        "repo": sample_repo_data,
        "payload": {
            "repository_id": 1296269,
            "push_id": 10115855396,
            "ref": "refs/heads/master",
            "head": "7a8f3ac80e2ad2f6842cb86f576d4bfe2c03e300",
            "before": "883efe034920928c47fe18598c01249d1a9fdabd",
        },
        "public": True,
        "created_at": "2022-06-07T07:50:26Z",
    }


@pytest.fixture
def sample_create_event_data(sample_actor_data, sample_repo_data):
    """Sample CreateEvent data for testing."""
    return {
        "id": "22249084965",
        "type": "CreateEvent",
        "actor": sample_actor_data,
        "repo": sample_repo_data,
        "payload": {
            "ref": "feature-branch",
            "ref_type": "branch",
            "master_branch": "main",
            "description": "Test repository",
        },
        "public": True,
        "created_at": "2022-06-07T08:00:00Z",
    }


@pytest.fixture
def sample_events_list(sample_watch_event_data, sample_push_event_data):
    """Sample list of events for testing."""
    return [sample_watch_event_data, sample_push_event_data]


@pytest.fixture
def mock_etag():
    """Sample ETag value for testing."""
    return "a18c3bded88eb5dbb5c849a489412bf3"


@pytest.fixture
def mock_poll_interval():
    """Sample poll interval for testing."""
    return 60


# New fixtures for nested models


@pytest.fixture
def sample_organization_data():
    """Sample organization data for testing."""
    return {
        "id": 1,
        "login": "github",
        "gravatar_id": "",
        "url": "https://api.github.com/orgs/github",
        "avatar_url": "https://avatars.githubusercontent.com/u/1?v=4",
    }


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "id": 1,
        "login": "octocat",
        "gravatar_id": "",
        "url": "https://api.github.com/users/octocat",
        "avatar_url": "https://avatars.githubusercontent.com/u/1?v=4",
        "type": "User",
    }


@pytest.fixture
def sample_label_data():
    """Sample label data for testing."""
    return {
        "id": 208045946,
        "name": "bug",
        "color": "d73a4a",
        "url": "https://api.github.com/repos/octocat/Hello-World/labels/bug",
        "default": True,
    }


@pytest.fixture
def sample_issue_data(sample_user_data, sample_label_data):
    """Sample issue data for testing."""
    return {
        "id": 1,
        "number": 1347,
        "title": "Found a bug",
        "state": "open",
        "body": "I'm having a problem with this.",
        "user": sample_user_data,
        "labels": [sample_label_data],
        "assignee": sample_user_data,
        "assignees": [sample_user_data],
        "url": "https://api.github.com/repos/octocat/Hello-World/issues/1347",
        "html_url": "https://github.com/octocat/Hello-World/issues/1347",
        "created_at": "2022-06-09T12:00:00Z",
        "updated_at": "2022-06-09T13:00:00Z",
        "closed_at": None,
    }


@pytest.fixture
def sample_comment_data(sample_user_data):
    """Sample comment data for testing."""
    return {
        "id": 1,
        "body": "This is a comment",
        "user": sample_user_data,
        "url": "https://api.github.com/repos/octocat/Hello-World/issues/comments/1",
        "html_url": "https://github.com/octocat/Hello-World/issues/1347#issuecomment-1",
        "created_at": "2022-06-09T12:30:00Z",
        "updated_at": "2022-06-09T12:30:00Z",
    }


@pytest.fixture
def sample_commit_comment_data(sample_user_data):
    """Sample commit comment data for testing."""
    return {
        "id": 1,
        "body": "Great commit!",
        "user": sample_user_data,
        "commit_id": "7a8f3ac80e2ad2f6842cb86f576d4bfe2c03e300",
        "path": "file.py",
        "position": 1,
        "line": 10,
        "url": "https://api.github.com/repos/octocat/Hello-World/comments/1",
        "html_url": "https://github.com/octocat/Hello-World/commit/7a8f3ac#commitcomment-1",
        "created_at": "2022-06-09T12:30:00Z",
        "updated_at": "2022-06-09T12:30:00Z",
    }


@pytest.fixture
def sample_pull_request_data(sample_user_data, sample_label_data):
    """Sample pull request data for testing."""
    return {
        "id": 1,
        "number": 42,
        "title": "Amazing new feature",
        "state": "open",
        "body": "Please merge this",
        "user": sample_user_data,
        "labels": [sample_label_data],
        "assignee": sample_user_data,
        "assignees": [sample_user_data],
        "merged": False,
        "url": "https://api.github.com/repos/octocat/Hello-World/pulls/42",
        "html_url": "https://github.com/octocat/Hello-World/pull/42",
        "created_at": "2022-06-09T12:00:00Z",
        "updated_at": "2022-06-09T13:00:00Z",
        "closed_at": None,
        "merged_at": None,
    }


@pytest.fixture
def sample_review_data(sample_user_data):
    """Sample PR review data for testing."""
    return {
        "id": 1,
        "body": "Looks good to me",
        "user": sample_user_data,
        "state": "APPROVED",
        "commit_id": "7a8f3ac80e2ad2f6842cb86f576d4bfe2c03e300",
        "url": "https://api.github.com/repos/octocat/Hello-World/pulls/42/reviews/1",
        "html_url": "https://github.com/octocat/Hello-World/pull/42#pullrequestreview-1",
        "submitted_at": "2022-06-09T14:00:00Z",
    }


@pytest.fixture
def sample_repository_data(sample_user_data):
    """Sample detailed repository data for testing."""
    return {
        "id": 1296269,
        "name": "Hello-World",
        "full_name": "octocat/Hello-World",
        "owner": sample_user_data,
        "private": False,
        "description": "My first repository",
        "fork": False,
        "url": "https://api.github.com/repos/octocat/Hello-World",
        "html_url": "https://github.com/octocat/Hello-World",
        "created_at": "2022-01-01T00:00:00Z",
        "updated_at": "2022-06-09T12:00:00Z",
        "pushed_at": "2022-06-09T11:00:00Z",
        "default_branch": "main",
    }


@pytest.fixture
def sample_release_data(sample_user_data):
    """Sample release data for testing."""
    return {
        "id": 1,
        "tag_name": "v1.0.0",
        "name": "Production Release",
        "body": "Description of the release",
        "draft": False,
        "prerelease": False,
        "author": sample_user_data,
        "url": "https://api.github.com/repos/octocat/Hello-World/releases/1",
        "html_url": "https://github.com/octocat/Hello-World/releases/tag/v1.0.0",
        "created_at": "2022-06-09T10:00:00Z",
        "published_at": "2022-06-09T12:00:00Z",
    }


@pytest.fixture
def sample_discussion_data(sample_user_data):
    """Sample discussion data for testing."""
    return {
        "id": 1,
        "number": 1,
        "title": "How do I do X?",
        "body": "I need help with X",
        "user": sample_user_data,
        "state": "open",
        "url": "https://api.github.com/repos/octocat/Hello-World/discussions/1",
        "html_url": "https://github.com/octocat/Hello-World/discussions/1",
        "created_at": "2022-06-09T12:00:00Z",
        "updated_at": "2022-06-09T12:30:00Z",
    }


@pytest.fixture
def sample_wiki_page_data():
    """Sample wiki page data for testing."""
    return {
        "page_name": "Home",
        "title": "Home Page",
        "summary": "Updated home page",
        "action": "edited",
        "sha": "7a8f3ac80e2ad2f6842cb86f576d4bfe2c03e300",
        "html_url": "https://github.com/octocat/Hello-World/wiki/Home",
    }


@pytest.fixture
def sample_commit_data():
    """Sample commit data for testing."""
    return {
        "sha": "7a8f3ac80e2ad2f6842cb86f576d4bfe2c03e300",
        "message": "Fix bug",
        "author": {
            "name": "Octocat",
            "email": "octocat@github.com",
        },
        "url": "https://api.github.com/repos/octocat/Hello-World/commits/7a8f3ac",
        "distinct": True,
    }


# New event fixtures


@pytest.fixture
def sample_commit_comment_event_data(
    sample_actor_data, sample_repo_data, sample_commit_comment_data
):
    """Sample CommitCommentEvent data for testing."""
    return {
        "id": "22249084970",
        "type": "CommitCommentEvent",
        "actor": sample_actor_data,
        "repo": sample_repo_data,
        "payload": {
            "action": "created",
            "comment": sample_commit_comment_data,
        },
        "public": True,
        "created_at": "2022-06-09T14:00:00Z",
    }


@pytest.fixture
def sample_discussion_event_data(
    sample_actor_data, sample_repo_data, sample_discussion_data
):
    """Sample DiscussionEvent data for testing."""
    return {
        "id": "22249084971",
        "type": "DiscussionEvent",
        "actor": sample_actor_data,
        "repo": sample_repo_data,
        "payload": {
            "action": "created",
            "discussion": sample_discussion_data,
        },
        "public": True,
        "created_at": "2022-06-09T15:00:00Z",
    }


@pytest.fixture
def sample_pull_request_review_event_data(
    sample_actor_data, sample_repo_data, sample_pull_request_data, sample_review_data
):
    """Sample PullRequestReviewEvent data for testing."""
    return {
        "id": "22249084972",
        "type": "PullRequestReviewEvent",
        "actor": sample_actor_data,
        "repo": sample_repo_data,
        "payload": {
            "action": "created",
            "pull_request": sample_pull_request_data,
            "review": sample_review_data,
        },
        "public": True,
        "created_at": "2022-06-09T16:00:00Z",
    }


@pytest.fixture
def sample_pull_request_review_comment_event_data(
    sample_actor_data, sample_repo_data, sample_pull_request_data, sample_comment_data
):
    """Sample PullRequestReviewCommentEvent data for testing."""
    return {
        "id": "22249084973",
        "type": "PullRequestReviewCommentEvent",
        "actor": sample_actor_data,
        "repo": sample_repo_data,
        "payload": {
            "action": "created",
            "pull_request": sample_pull_request_data,
            "comment": sample_comment_data,
        },
        "public": True,
        "created_at": "2022-06-09T17:00:00Z",
    }


# GH Archive fixtures


@pytest.fixture
def sample_gharchive_file(
    tmp_path,
    sample_watch_event_data,
    sample_push_event_data,
    sample_create_event_data,
):
    """Create a sample GH Archive .json.gz file for testing.

    This fixture creates a temporary GH Archive file containing multiple
    valid event records in NDJSON format.
    """
    archive_file = tmp_path / "2015-01-01-15.json.gz"

    events = [
        sample_watch_event_data,
        sample_push_event_data,
        sample_create_event_data,
    ]

    with gzip.open(archive_file, "wt", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    return archive_file


@pytest.fixture
def sample_gharchive_file_with_invalid_events(tmp_path, sample_watch_event_data):
    """Create a GH Archive file with both valid and invalid events.

    This fixture creates a file containing:
    - Valid events
    - Events with missing required fields
    - Events with invalid data types
    """
    archive_file = tmp_path / "mixed_validity.json.gz"

    # Valid event
    valid_event = sample_watch_event_data

    # Invalid event - missing required 'actor' field
    invalid_event_1 = {
        "id": "22249084948",
        "type": "WatchEvent",
        "repo": {
            "id": 1296269,
            "name": "octocat/Hello-World",
            "url": "https://api.github.com/repos/octocat/Hello-World",
        },
        "payload": {"action": "started"},
        "public": True,
        "created_at": "2022-06-09T12:47:28Z",
    }

    # Invalid event - wrong data type for 'id'
    invalid_event_2 = {
        "id": 12345,  # Should be string
        "type": "WatchEvent",
        "actor": sample_watch_event_data["actor"],
        "repo": sample_watch_event_data["repo"],
        "payload": {"action": "started"},
        "public": True,
        "created_at": "2022-06-09T12:47:28Z",
    }

    events = [valid_event, invalid_event_1, invalid_event_2]

    with gzip.open(archive_file, "wt", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    return archive_file


@pytest.fixture
def sample_large_gharchive_file(tmp_path, sample_watch_event_data):
    """Create a larger GH Archive file for testing iteration and limits.

    This fixture creates a file with 100 events for testing pagination,
    limits, and performance.
    """
    archive_file = tmp_path / "large_archive.json.gz"

    with gzip.open(archive_file, "wt", encoding="utf-8") as f:
        for i in range(100):
            event = sample_watch_event_data.copy()
            event["id"] = f"event_{i}"
            f.write(json.dumps(event) + "\n")

    return archive_file
