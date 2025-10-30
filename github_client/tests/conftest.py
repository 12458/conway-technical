"""Pytest fixtures for github_client tests."""

from datetime import datetime, timezone

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
