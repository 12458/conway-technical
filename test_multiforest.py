"""Test script for multi-forest anomaly detection and new features."""

import datetime
from github_client.models import Event, Actor, Repo, Organization
from service.anomaly_detector import MultiForestAnomalyDetector
from github_client.feature_extractor import GitHubFeatureExtractor

def create_test_event(event_type: str, actor_login: str, repo_name: str, created_at: datetime.datetime) -> Event:
    """Create a test event."""
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


def test_feature_dimensions():
    """Test that feature extraction produces correct dimensions."""
    print("=" * 80)
    print("TEST 1: Feature Dimensions")
    print("=" * 80)

    extractor = GitHubFeatureExtractor(filter_bots=False)

    # Create test events
    now = datetime.datetime.now(datetime.timezone.utc)
    event = create_test_event("IssuesEvent", "testuser", "owner/repo", now)

    features = extractor.extract_features(event)

    if features is not None:
        print(f"✓ Feature vector extracted successfully")
        print(f"  Feature dimensions: {len(features)}")
        print(f"  Expected: 276 (64 actor + 4 behavior + 1 entropy + 64 repo + 2 repo_activity + 1 repo_entropy + 32 org + 100 type_specific + 5 velocity + 3 burst)")

        if len(features) == 276:
            print(f"  ✓ Dimensions match expected!")
        else:
            print(f"  ✗ Dimension mismatch! Expected 276, got {len(features)}")
    else:
        print("✗ Feature extraction returned None")

    print()


def test_burst_detection():
    """Test temporal burst detection."""
    print("=" * 80)
    print("TEST 2: Temporal Burst Detection")
    print("=" * 80)

    extractor = GitHubFeatureExtractor(filter_bots=False)

    # Create burst of events (10 events in 15 seconds)
    base_time = datetime.datetime.now(datetime.timezone.utc)
    actor = "burst-tester"

    print(f"Creating burst: 10 events in 15 seconds from {actor}")
    for i in range(10):
        event_time = base_time + datetime.timedelta(seconds=i * 1.5)
        event = create_test_event("IssuesEvent", actor, f"owner/repo{i % 3}", event_time)
        features = extractor.extract_features(event)

    # Get burst features
    last_timestamp = (base_time + datetime.timedelta(seconds=13.5)).timestamp()
    burst_features = extractor.get_temporal_burst_features(actor, last_timestamp)

    print(f"  Burst count: {burst_features[0]:.0f}")
    print(f"  Silence ratio: {burst_features[1]:.3f} (lower = more bot-like)")
    print(f"  Inter-event CV: {burst_features[2]:.3f} (lower = more uniform/robotic)")

    if burst_features[0] > 0:
        print(f"  ✓ Burst detected!")
    else:
        print(f"  ⚠ No burst detected (may need more events or time)")

    print()


def test_repo_hopping():
    """Test time-windowed repo hopping detection."""
    print("=" * 80)
    print("TEST 3: Time-Windowed Repo Hopping")
    print("=" * 80)

    extractor = GitHubFeatureExtractor(filter_bots=False)

    # Create events across multiple repos
    base_time = datetime.datetime.now(datetime.timezone.utc)
    actor = "repo-hopper"
    num_repos = 15

    print(f"Creating events: {num_repos} different repos in 60 seconds from {actor}")
    for i in range(num_repos):
        event_time = base_time + datetime.timedelta(seconds=i * 4)
        event = create_test_event("PushEvent", actor, f"owner/repo{i}", event_time)
        features = extractor.extract_features(event)

    # Get repo hopping stats
    last_timestamp = (base_time + datetime.timedelta(seconds=(num_repos - 1) * 4)).timestamp()
    windowed_repos = extractor.get_repo_hopping_features(actor, last_timestamp)

    # Get all-time repos from actor_repos
    alltime_repos = len(extractor.actor_repos.get(actor, set()))

    print(f"  All-time unique repos: {alltime_repos}")
    print(f"  Windowed unique repos (5 min): {windowed_repos}")

    if windowed_repos > 10:
        print(f"  ✓ High repo hopping detected in time window!")

    print()


def test_multiforest_routing():
    """Test that events are routed to correct forests."""
    print("=" * 80)
    print("TEST 4: Multi-Forest Event Routing")
    print("=" * 80)

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

    print(f"Testing event type routing:")
    for event_type, expected_group in test_cases:
        actual_group = detector._get_forest_group(event_type)
        status = "✓" if actual_group == expected_group else "✗"
        print(f"  {status} {event_type:25s} -> {actual_group:15s} (expected: {expected_group})")

    print()


def test_anomaly_scores():
    """Test anomaly score differences between event types."""
    print("=" * 80)
    print("TEST 5: Anomaly Score Separation by Event Type")
    print("=" * 80)

    detector = MultiForestAnomalyDetector()

    base_time = datetime.datetime.now(datetime.timezone.utc)

    # Simulate a stream with mostly PushEvents
    print("Simulating event stream: 20 PushEvents, then 5 IssuesEvents")
    print()

    # Feed 20 push events
    for i in range(20):
        event_time = base_time + datetime.timedelta(seconds=i * 60)
        event = create_test_event("PushEvent", f"user{i % 5}", f"owner/repo{i % 3}", event_time)
        score, patterns, features, vel_score, is_inhuman, vel_reason = detector.process_event(event)
        if score is not None and i % 5 == 0:
            print(f"  PushEvent #{i+1:2d}: CoDisp = {score:.2f}")

    print()

    # Feed 5 issue events from same users
    issue_scores = []
    for i in range(5):
        event_time = base_time + datetime.timedelta(seconds=(20 + i) * 60)
        event = create_test_event("IssuesEvent", f"user{i}", f"owner/repo{i % 3}", event_time)
        score, patterns, features, vel_score, is_inhuman, vel_reason = detector.process_event(event)
        if score is not None:
            issue_scores.append(score)
            print(f"  IssuesEvent #{i+1}: CoDisp = {score:.2f}")

    print()

    if issue_scores:
        avg_issue_score = sum(issue_scores) / len(issue_scores)
        print(f"Average IssuesEvent score: {avg_issue_score:.2f}")
        print(f"Threshold: {detector.threshold:.2f}")

        if avg_issue_score < detector.threshold:
            print(f"✓ IssuesEvents have reasonable scores (below threshold)")
            print(f"  This shows that issues are NOT being flagged just for being rare!")
        else:
            print(f"⚠ Some IssuesEvents exceed threshold (may be expected for initial events)")

    print()

    # Print forest statistics
    stats = detector.get_stats()
    print("Forest Statistics:")
    for group_name, group_stats in stats['forest_stats'].items():
        print(f"  {group_name:15s}: {group_stats['points_processed']:3d} events processed, "
              f"avg tree size: {group_stats['avg_tree_size']:.1f}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MULTI-FOREST ANOMALY DETECTION TEST SUITE")
    print("=" * 80)
    print()

    test_feature_dimensions()
    test_burst_detection()
    test_repo_hopping()
    test_multiforest_routing()
    test_anomaly_scores()

    print("=" * 80)
    print("TEST SUITE COMPLETED")
    print("=" * 80)
    print()
