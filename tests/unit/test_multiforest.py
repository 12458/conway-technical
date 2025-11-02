"""Test script for multi-forest anomaly detection and new features."""

import datetime

import pytest

from github_client.models import Event, Actor, Repo, Organization
from service.anomaly_detector import MultiForestAnomalyDetector
from github_client.feature_extractor import GitHubFeatureExtractor


def create_test_event(event_type: str, actor_login: str, repo_name: str, created_at: datetime.datetime) -> Event:
    """Create a test event for testing purposes."""
    return Event(
        id=f"test-{event_type}-{actor_login}",
        type=event_type,
        actor=Actor(
            id=12345,
            login=actor_login,
            display_login=actor_login,
            gravatar_id="",
            url="https://api.github.com/users/test",
            avatar_url="https://avatars.githubusercontent.com/u/12345",
        ),
        repo=Repo(
            id=67890,
            name=repo_name,
            url="https://api.github.com/repos/test/repo",
        ),
        payload={},
        public=True,
        created_at=created_at,
        org=None,
    )


@pytest.mark.unit
def test_feature_dimensions():
    """Test that feature extraction produces correct dimensions."""
    extractor = GitHubFeatureExtractor(filter_bots=False)

    # Create test events
    now = datetime.datetime.now(datetime.timezone.utc)
    event = create_test_event("IssuesEvent", "testuser", "owner/repo", now)

    features = extractor.extract_features(event)

    assert features is not None, "Feature extraction should not return None"
    # 64 actor + 4 behavior + 1 entropy + 64 repo + 2 repo_activity + 1 repo_entropy + 32 org + 100 type_specific + 5 velocity + 3 burst
    expected_dimensions = 276
    assert len(features) == expected_dimensions, f"Expected {expected_dimensions} dimensions, got {len(features)}"


@pytest.mark.unit
def test_burst_detection():
    """Test temporal burst detection."""
    extractor = GitHubFeatureExtractor(filter_bots=False)

    # Create burst of events (10 events in 15 seconds)
    base_time = datetime.datetime.now(datetime.timezone.utc)
    actor = "burst-tester"

    for i in range(10):
        event_time = base_time + datetime.timedelta(seconds=i * 1.5)
        event = create_test_event("IssuesEvent", actor, f"owner/repo{i % 3}", event_time)
        features = extractor.extract_features(event)

    # Get burst features
    last_timestamp = (base_time + datetime.timedelta(seconds=13.5)).timestamp()
    burst_features = extractor.get_temporal_burst_features(actor, last_timestamp)

    # burst_features[0] is burst count, should be > 0
    # burst_features[1] is silence ratio (lower = more bot-like)
    # burst_features[2] is inter-event CV (lower = more uniform/robotic)
    assert burst_features[0] >= 0, "Burst count should be non-negative"
    assert 0 <= burst_features[1] <= 1, "Silence ratio should be between 0 and 1"
    assert burst_features[2] >= 0, "Inter-event CV should be non-negative"


@pytest.mark.unit
def test_repo_hopping():
    """Test time-windowed repo hopping detection."""
    extractor = GitHubFeatureExtractor(filter_bots=False)

    # Create events across multiple repos
    base_time = datetime.datetime.now(datetime.timezone.utc)
    actor = "repo-hopper"
    num_repos = 15

    for i in range(num_repos):
        event_time = base_time + datetime.timedelta(seconds=i * 4)
        event = create_test_event("PushEvent", actor, f"owner/repo{i}", event_time)
        features = extractor.extract_features(event)

    # Get repo hopping stats
    last_timestamp = (base_time + datetime.timedelta(seconds=(num_repos - 1) * 4)).timestamp()
    windowed_repos = extractor.get_repo_hopping_features(actor, last_timestamp)

    # Get all-time repos from actor_repos
    alltime_repos = len(extractor.actor_repos.get(actor, set()))

    assert alltime_repos == num_repos, f"Expected {num_repos} all-time repos, got {alltime_repos}"
    assert windowed_repos > 0, "Windowed repos should be positive"
    assert windowed_repos <= alltime_repos, "Windowed repos should not exceed all-time repos"


@pytest.mark.unit
def test_multiforest_routing():
    """Test that events are routed to correct forests."""
    detector = MultiForestAnomalyDetector()

    # Test different event types
    test_cases = [
        ("PushEvent", "push"),
        ("IssuesEvent", "issues"),
        ("IssueCommentEvent", "issues"),
        ("PullRequestEvent", "pull_request"),
        ("MemberEvent", "security"),
        ("DeleteEvent", "security"),
        ("WatchEvent", "other"),
        ("UnknownEvent", "other"),
    ]

    for event_type, expected_group in test_cases:
        actual_group = detector._get_forest_group(event_type)
        assert actual_group == expected_group, f"{event_type} should route to {expected_group}, got {actual_group}"


@pytest.mark.unit
def test_anomaly_scores():
    """Test anomaly score differences between event types."""
    detector = MultiForestAnomalyDetector()

    base_time = datetime.datetime.now(datetime.timezone.utc)

    # Feed 20 push events
    for i in range(20):
        event_time = base_time + datetime.timedelta(seconds=i * 60)
        event = create_test_event("PushEvent", f"user{i % 5}", f"owner/repo{i % 3}", event_time)
        score, patterns, features, vel_score, is_inhuman, vel_reason = detector.process_event(event)

    # Feed 5 issue events from same users
    issue_scores = []
    for i in range(5):
        event_time = base_time + datetime.timedelta(seconds=(20 + i) * 60)
        event = create_test_event("IssuesEvent", f"user{i}", f"owner/repo{i % 3}", event_time)
        score, patterns, features, vel_score, is_inhuman, vel_reason = detector.process_event(event)
        if score is not None:
            issue_scores.append(score)

    # Verify we got some scores
    assert len(issue_scores) > 0, "Should have received some anomaly scores"

    # Verify scores are reasonable (non-negative)
    for score in issue_scores:
        assert score >= 0, "Anomaly scores should be non-negative"

    # Verify forest statistics
    stats = detector.get_stats()
    assert 'forest_stats' in stats, "Stats should contain forest_stats"
    assert len(stats['forest_stats']) > 0, "Should have at least one forest with stats"
