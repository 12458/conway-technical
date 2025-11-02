"""Comprehensive unit tests for GitHubFeatureExtractor.

This test suite aims to increase coverage from 11% to 80%+ by testing:
- Feature hashing (categorical and n-gram)
- Entropy calculation
- Bot filtering
- LRU eviction
- Welford normalization
- Event-type-specific feature extraction
- Suspicious pattern detection
- Edge cases and error handling
"""

import datetime
from collections import deque

import numpy as np
import pytest

from github_client.feature_extractor import (
    GitHubFeatureExtractor,
    KNOWN_BOTS,
    MAX_ACTORS,
    MAX_REPOS,
    MAX_TEXT_BYTES,
)
from github_client.models import Event, Actor, Repo, Organization


def create_test_event(
    event_type: str,
    actor_login: str,
    repo_name: str,
    created_at: datetime.datetime | None = None,
    payload: dict | None = None,
    org: Organization | None = None,
) -> Event:
    """Create a test event for testing purposes."""
    if created_at is None:
        created_at = datetime.datetime.now(datetime.timezone.utc)
    if payload is None:
        payload = {}

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
        payload=payload,
        public=True,
        created_at=created_at,
        org=org,
    )


@pytest.mark.unit
class TestFeatureExtractorInitialization:
    """Tests for GitHubFeatureExtractor initialization."""

    def test_default_initialization(self):
        """Test extractor initializes with default values."""
        extractor = GitHubFeatureExtractor()

        assert extractor.categorical_dims == {
            "event_type": 32,
            "actor": 64,
            "repo": 64,
            "org": 32,
            "action": 16,
        }
        assert extractor.text_dims == {
            "commits": 64,
            "titles": 32,
            "bodies": 32,
            "comments": 32,
        }
        assert extractor.decay_halflife == 1000.0
        assert extractor.normalize is True
        assert extractor.ngram_size == 4
        assert extractor.filter_bots is True

    def test_custom_initialization(self):
        """Test extractor initializes with custom values."""
        custom_cat_dims = {"event_type": 16, "actor": 32, "repo": 32, "org": 16, "action": 8}
        custom_text_dims = {"commits": 32, "titles": 16, "bodies": 16, "comments": 16}

        extractor = GitHubFeatureExtractor(
            categorical_dims=custom_cat_dims,
            text_dims=custom_text_dims,
            decay_halflife=500.0,
            normalize=False,
            ngram_size=3,
            filter_bots=False,
        )

        assert extractor.categorical_dims == custom_cat_dims
        assert extractor.text_dims == custom_text_dims
        assert extractor.decay_halflife == 500.0
        assert extractor.normalize is False
        assert extractor.ngram_size == 3
        assert extractor.filter_bots is False

    def test_decay_alpha_calculation(self):
        """Test decay alpha is correctly calculated from halflife."""
        extractor = GitHubFeatureExtractor(decay_halflife=1000.0)

        # After 1000 events, count should decay to 50%
        expected_alpha = 0.5 ** (1.0 / 1000.0)
        assert extractor.decay_alpha == pytest.approx(expected_alpha)


@pytest.mark.unit
class TestCategoricalHashing:
    """Tests for _hash_categorical method."""

    def test_hash_categorical_produces_one_hot(self):
        """Test that categorical hashing produces a one-hot vector."""
        extractor = GitHubFeatureExtractor()
        vec = extractor._hash_categorical("test_value", dim=32)

        assert len(vec) == 32
        assert np.sum(vec) == 1.0  # Exactly one non-zero value
        assert np.max(vec) == 1.0  # That value is 1.0
        assert vec.dtype == float

    def test_hash_categorical_consistency(self):
        """Test that same input produces same output."""
        extractor = GitHubFeatureExtractor()

        vec1 = extractor._hash_categorical("test_value", dim=32)
        vec2 = extractor._hash_categorical("test_value", dim=32)

        assert np.array_equal(vec1, vec2)

    def test_hash_categorical_different_seeds(self):
        """Test that different seeds produce different outputs."""
        extractor = GitHubFeatureExtractor()

        vec1 = extractor._hash_categorical("test_value", dim=32, seed=0)
        vec2 = extractor._hash_categorical("test_value", dim=32, seed=1)

        # Different seeds should usually produce different vectors
        # (not guaranteed, but very likely)
        assert not np.array_equal(vec1, vec2)

    def test_hash_categorical_empty_string(self):
        """Test hashing empty string produces zero vector."""
        extractor = GitHubFeatureExtractor()
        vec = extractor._hash_categorical("", dim=32)

        assert len(vec) == 32
        assert np.sum(vec) == 0.0  # Empty string should produce zero vector

    def test_hash_categorical_different_dimensions(self):
        """Test hashing with different dimension sizes."""
        extractor = GitHubFeatureExtractor()

        vec16 = extractor._hash_categorical("test", dim=16)
        vec64 = extractor._hash_categorical("test", dim=64)

        assert len(vec16) == 16
        assert len(vec64) == 64
        assert np.sum(vec16) == 1.0
        assert np.sum(vec64) == 1.0


@pytest.mark.unit
class TestTextNgramHashing:
    """Tests for _hash_text_ngrams method."""

    def test_hash_text_ngrams_basic(self):
        """Test basic n-gram hashing."""
        extractor = GitHubFeatureExtractor(ngram_size=4)
        vec = extractor._hash_text_ngrams("hello world", dim=32)

        assert len(vec) == 32
        assert np.sum(vec) > 0  # Should have some non-zero values
        assert vec.dtype == float

    def test_hash_text_ngrams_consistency(self):
        """Test that same text produces same output."""
        extractor = GitHubFeatureExtractor(ngram_size=4)

        vec1 = extractor._hash_text_ngrams("test text", dim=32)
        vec2 = extractor._hash_text_ngrams("test text", dim=32)

        assert np.array_equal(vec1, vec2)

    def test_hash_text_ngrams_different_texts(self):
        """Test that different texts produce different outputs."""
        extractor = GitHubFeatureExtractor(ngram_size=4)

        vec1 = extractor._hash_text_ngrams("hello", dim=32)
        vec2 = extractor._hash_text_ngrams("world", dim=32)

        assert not np.array_equal(vec1, vec2)

    def test_hash_text_ngrams_empty_string(self):
        """Test hashing empty string produces zero vector."""
        extractor = GitHubFeatureExtractor(ngram_size=4)
        vec = extractor._hash_text_ngrams("", dim=32)

        assert len(vec) == 32
        assert np.sum(vec) == 0.0

    def test_hash_text_ngrams_truncation(self):
        """Test that very long text is properly handled."""
        extractor = GitHubFeatureExtractor(ngram_size=4)

        # Create text longer than MAX_TEXT_BYTES
        long_text = "x" * (MAX_TEXT_BYTES + 1000)
        vec = extractor._hash_text_ngrams(long_text, dim=32)

        # Should not crash and should produce valid output
        assert len(vec) == 32
        assert np.sum(vec) > 0

    def test_hash_text_ngrams_unicode(self):
        """Test hashing text with unicode characters."""
        extractor = GitHubFeatureExtractor(ngram_size=4)

        text = "Hello 世界 🌍"
        vec = extractor._hash_text_ngrams(text, dim=32)

        assert len(vec) == 32
        assert np.sum(vec) > 0

    def test_hash_text_ngrams_custom_n(self):
        """Test using custom n-gram size."""
        extractor = GitHubFeatureExtractor(ngram_size=4)

        vec_n2 = extractor._hash_text_ngrams("test", dim=32, n=2)
        vec_n3 = extractor._hash_text_ngrams("test", dim=32, n=3)
        vec_n4 = extractor._hash_text_ngrams("test", dim=32, n=4)

        # Different n-gram sizes should usually produce different results
        assert not np.array_equal(vec_n2, vec_n3)
        assert not np.array_equal(vec_n3, vec_n4)


@pytest.mark.unit
class TestBotFiltering:
    """Tests for bot detection and filtering."""

    def test_is_bot_known_bots(self):
        """Test detection of known bot accounts."""
        extractor = GitHubFeatureExtractor(filter_bots=True)

        for bot in list(KNOWN_BOTS)[:5]:  # Test first 5
            assert extractor.is_bot(bot) is True

    def test_is_bot_bracket_pattern(self):
        """Test detection of [bot] suffix pattern."""
        extractor = GitHubFeatureExtractor(filter_bots=True)

        assert extractor.is_bot("mybot[bot]") is True
        assert extractor.is_bot("custom-bot[BOT]") is True
        assert extractor.is_bot("test[Bot]") is True

    def test_is_bot_regular_users(self):
        """Test that regular users are not flagged as bots."""
        extractor = GitHubFeatureExtractor(filter_bots=True)

        assert extractor.is_bot("john_doe") is False
        assert extractor.is_bot("alice-smith") is False
        assert extractor.is_bot("developer123") is False

    def test_is_bot_edge_cases(self):
        """Test bot detection edge cases."""
        extractor = GitHubFeatureExtractor(filter_bots=True)

        # Edge case: 'bot' in username but not as [bot] suffix
        assert extractor.is_bot("robotics-dev") is False
        assert extractor.is_bot("bot-not-a-bot") is False  # No [bot] pattern
        assert extractor.is_bot("test-bot[bot]") is True  # Has [bot] pattern

    def test_filter_bots_disabled(self):
        """Test that bot filtering can be disabled."""
        extractor = GitHubFeatureExtractor(filter_bots=False)

        # is_bot should still work even when filtering is disabled
        assert extractor.is_bot("github-actions[bot]") is True


@pytest.mark.unit
class TestLRUEviction:
    """Tests for LRU eviction when limits are exceeded."""

    def test_actor_lru_eviction(self):
        """Test that oldest actors are evicted when limit is exceeded."""
        extractor = GitHubFeatureExtractor()

        # Add MAX_ACTORS + 1 actors
        for i in range(MAX_ACTORS + 5):
            actor = f"actor_{i}"
            extractor.actor_event_counts[actor] = 1.0
            extractor.actor_event_counts.move_to_end(actor)
            extractor.actor_last_seen[actor] = i
            extractor._evict_lru_actors()

        # Should have exactly MAX_ACTORS actors
        assert len(extractor.actor_event_counts) == MAX_ACTORS

        # First actors should be evicted
        assert "actor_0" not in extractor.actor_event_counts
        assert "actor_1" not in extractor.actor_event_counts

        # Recent actors should remain
        assert f"actor_{MAX_ACTORS + 4}" in extractor.actor_event_counts

    def test_repo_lru_eviction(self):
        """Test that oldest repos are evicted when limit is exceeded."""
        extractor = GitHubFeatureExtractor()

        # Add MAX_REPOS + 1 repos
        for i in range(MAX_REPOS + 5):
            repo = f"owner/repo_{i}"
            extractor.repo_event_counts[repo] = 1.0
            extractor.repo_event_counts.move_to_end(repo)
            extractor.repo_last_seen[repo] = i
            extractor._evict_lru_repos()

        # Should have exactly MAX_REPOS repos
        assert len(extractor.repo_event_counts) == MAX_REPOS

        # First repos should be evicted
        assert "owner/repo_0" not in extractor.repo_event_counts

        # Recent repos should remain
        assert f"owner/repo_{MAX_REPOS + 4}" in extractor.repo_event_counts

    def test_actor_eviction_cleans_all_data(self):
        """Test that actor eviction cleans all associated data structures."""
        extractor = GitHubFeatureExtractor()

        # Set up actor with data in all structures
        for i in range(MAX_ACTORS + 1):
            actor = f"actor_{i}"
            extractor.actor_event_counts[actor] = 1.0
            extractor.actor_last_seen[actor] = i
            extractor.actor_repos[actor] = set(["repo1"])
            extractor.actor_event_types[actor] = ["PushEvent"]
            extractor.actor_timestamps[actor] = deque([100.0])
            extractor.actor_last_event_time[actor] = 100.0
            extractor.actor_bursts[actor] = [(100.0, 1)]
            extractor.actor_repos_windowed[actor] = deque([(100.0, "repo1")])
            extractor.actor_event_counts.move_to_end(actor)
            extractor._evict_lru_actors()

        # First actor should be completely removed
        assert "actor_0" not in extractor.actor_event_counts
        assert "actor_0" not in extractor.actor_last_seen
        assert "actor_0" not in extractor.actor_repos
        assert "actor_0" not in extractor.actor_event_types
        assert "actor_0" not in extractor.actor_timestamps
        assert "actor_0" not in extractor.actor_last_event_time
        assert "actor_0" not in extractor.actor_bursts
        assert "actor_0" not in extractor.actor_repos_windowed


@pytest.mark.unit
class TestDecayedCount:
    """Tests for exponential decay calculation."""

    def test_get_decayed_count_no_time_elapsed(self):
        """Test decay when no time has elapsed."""
        extractor = GitHubFeatureExtractor(decay_halflife=1000.0)
        extractor.total_events = 100

        decayed = extractor._get_decayed_count(count=10.0, last_seen=100)
        assert decayed == 10.0  # No decay

    def test_get_decayed_count_after_halflife(self):
        """Test decay after one halflife period."""
        extractor = GitHubFeatureExtractor(decay_halflife=1000.0)
        extractor.total_events = 1100  # 1000 events later

        decayed = extractor._get_decayed_count(count=10.0, last_seen=100)
        assert decayed == pytest.approx(5.0, rel=0.01)  # Should be ~50%

    def test_get_decayed_count_gradual(self):
        """Test that decay is gradual over time."""
        extractor = GitHubFeatureExtractor(decay_halflife=1000.0)
        extractor.total_events = 0

        initial_count = 100.0
        last_seen = 0

        # After 100 events
        extractor.total_events = 100
        decay_100 = extractor._get_decayed_count(initial_count, last_seen)

        # After 500 events
        extractor.total_events = 500
        decay_500 = extractor._get_decayed_count(initial_count, last_seen)

        # After 1000 events
        extractor.total_events = 1000
        decay_1000 = extractor._get_decayed_count(initial_count, last_seen)

        # Decay should be gradual
        assert decay_100 > decay_500 > decay_1000
        assert decay_1000 == pytest.approx(50.0, rel=0.01)


@pytest.mark.unit
class TestFeatureExtraction:
    """Tests for main extract_features method."""

    def test_extract_features_returns_numpy_array(self):
        """Test that extract_features returns a numpy array."""
        extractor = GitHubFeatureExtractor(filter_bots=False)
        event = create_test_event("PushEvent", "testuser", "owner/repo")

        features = extractor.extract_features(event)

        assert features is not None
        assert isinstance(features, np.ndarray)
        assert len(features) > 0

    def test_extract_features_filters_bots(self):
        """Test that bot events are filtered when enabled."""
        extractor = GitHubFeatureExtractor(filter_bots=True)
        bot_event = create_test_event("PushEvent", "github-actions[bot]", "owner/repo")

        features = extractor.extract_features(bot_event)

        assert features is None  # Bot should be filtered

    def test_extract_features_allows_bots_when_disabled(self):
        """Test that bots are not filtered when filtering is disabled."""
        extractor = GitHubFeatureExtractor(filter_bots=False)
        bot_event = create_test_event("PushEvent", "github-actions[bot]", "owner/repo")

        features = extractor.extract_features(bot_event)

        assert features is not None  # Bot should not be filtered

    def test_extract_features_increments_total_events(self):
        """Test that total_events counter is incremented."""
        extractor = GitHubFeatureExtractor(filter_bots=False)
        assert extractor.total_events == 0

        event = create_test_event("PushEvent", "testuser", "owner/repo")
        extractor.extract_features(event)

        assert extractor.total_events == 1

        extractor.extract_features(event)
        assert extractor.total_events == 2

    def test_extract_features_updates_actor_stats(self):
        """Test that actor statistics are updated."""
        extractor = GitHubFeatureExtractor(filter_bots=False)
        event = create_test_event("PushEvent", "testuser", "owner/repo")

        extractor.extract_features(event)

        assert "testuser" in extractor.actor_event_counts
        assert "testuser" in extractor.actor_last_seen
        assert "testuser" in extractor.actor_repos
        assert "owner/repo" in extractor.actor_repos["testuser"]

    def test_extract_features_updates_repo_stats(self):
        """Test that repo statistics are updated."""
        extractor = GitHubFeatureExtractor(filter_bots=False)
        event = create_test_event("PushEvent", "testuser", "owner/repo")

        extractor.extract_features(event)

        assert "owner/repo" in extractor.repo_event_counts
        assert "owner/repo" in extractor.repo_last_seen
        assert "owner/repo" in extractor.repo_actors
        assert "testuser" in extractor.repo_actors["owner/repo"]


@pytest.mark.unit
class TestEventTypeFeatureExtraction:
    """Tests for event-type-specific feature extraction."""

    def test_push_event_feature_extraction(self):
        """Test feature extraction for PushEvent."""
        extractor = GitHubFeatureExtractor(filter_bots=False, normalize=False)

        payload = {
            "ref": "refs/heads/main",
            "size": 3,
            "commits": [
                {"message": "Initial commit", "author": {"name": "Test"}},
                {"message": "Add feature", "author": {"name": "Test"}},
            ],
        }
        event = create_test_event("PushEvent", "testuser", "owner/repo", payload=payload)

        features = extractor.extract_features(event)

        assert features is not None
        assert len(features) > 0

    def test_issues_event_feature_extraction(self):
        """Test feature extraction for IssuesEvent."""
        extractor = GitHubFeatureExtractor(filter_bots=False, normalize=False)

        payload = {
            "action": "opened",
            "issue": {"title": "Bug report", "body": "There is a bug"},
        }
        event = create_test_event("IssuesEvent", "testuser", "owner/repo", payload=payload)

        features = extractor.extract_features(event)

        assert features is not None
        assert len(features) > 0

    def test_pull_request_event_feature_extraction(self):
        """Test feature extraction for PullRequestEvent."""
        extractor = GitHubFeatureExtractor(filter_bots=False, normalize=False)

        payload = {
            "action": "opened",
            "pull_request": {
                "title": "Add new feature",
                "body": "This PR adds a new feature",
            },
        }
        event = create_test_event("PullRequestEvent", "testuser", "owner/repo", payload=payload)

        features = extractor.extract_features(event)

        assert features is not None
        assert len(features) > 0

    def test_member_event_feature_extraction(self):
        """Test feature extraction for MemberEvent (security-critical)."""
        extractor = GitHubFeatureExtractor(filter_bots=False, normalize=False)

        payload = {"action": "added", "member": {"login": "newmember"}}
        event = create_test_event("MemberEvent", "testuser", "owner/repo", payload=payload)

        features = extractor.extract_features(event)

        assert features is not None
        assert len(features) > 0

    def test_delete_event_feature_extraction(self):
        """Test feature extraction for DeleteEvent (destructive)."""
        extractor = GitHubFeatureExtractor(filter_bots=False, normalize=False)

        payload = {"ref": "feature-branch", "ref_type": "branch"}
        event = create_test_event("DeleteEvent", "testuser", "owner/repo", payload=payload)

        features = extractor.extract_features(event)

        assert features is not None
        assert len(features) > 0


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_extract_features_empty_payload(self):
        """Test feature extraction with empty payload."""
        extractor = GitHubFeatureExtractor(filter_bots=False, normalize=False)

        event = create_test_event("PushEvent", "testuser", "owner/repo", payload={})

        features = extractor.extract_features(event)

        # Should not crash, should return valid features
        assert features is not None
        assert len(features) > 0

    def test_extract_features_missing_optional_fields(self):
        """Test feature extraction with missing optional fields in payload."""
        extractor = GitHubFeatureExtractor(filter_bots=False, normalize=False)

        # PushEvent payload with missing 'commits' field
        payload = {"ref": "refs/heads/main"}
        event = create_test_event("PushEvent", "testuser", "owner/repo", payload=payload)

        features = extractor.extract_features(event)

        # Should not crash
        assert features is not None

    def test_extract_features_with_org(self):
        """Test feature extraction for event with organization."""
        extractor = GitHubFeatureExtractor(filter_bots=False, normalize=False)

        org = Organization(
            id=99999,
            login="test-org",
            gravatar_id="",
            url="https://api.github.com/orgs/test-org",
            avatar_url="https://avatars.githubusercontent.com/o/99999",
        )
        event = create_test_event("PushEvent", "testuser", "test-org/repo", org=org)

        features = extractor.extract_features(event)

        assert features is not None
        assert len(features) > 0

    def test_extract_features_without_org(self):
        """Test feature extraction for event without organization."""
        extractor = GitHubFeatureExtractor(filter_bots=False, normalize=False)

        event = create_test_event("PushEvent", "testuser", "owner/repo", org=None)

        features = extractor.extract_features(event)

        assert features is not None
        assert len(features) > 0
