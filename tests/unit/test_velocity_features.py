"""Unit tests for time-based velocity features in anomaly detection."""

import time
from datetime import datetime, timezone

import numpy as np
import pytest

from github_client.feature_extractor import GitHubFeatureExtractor
from github_client.models import Actor, Event, Repo
from service.anomaly_detector import StreamingAnomalyDetector
from service.config import service_settings


class TestVelocityFeatures:
    """Test time-based velocity feature extraction."""

    def test_timestamp_tracking(self):
        """Test that timestamps are properly tracked for actors."""
        extractor = GitHubFeatureExtractor()

        # Create test event
        current_time = time.time()
        extractor._update_actor_timestamps("test_user", current_time)

        # Check timestamp was stored
        assert "test_user" in extractor.actor_timestamps
        assert len(extractor.actor_timestamps["test_user"]) == 1
        assert extractor.actor_timestamps["test_user"][0] == current_time
        assert extractor.actor_last_event_time["test_user"] == current_time

    def test_multiple_timestamps(self):
        """Test tracking multiple timestamps for same actor."""
        extractor = GitHubFeatureExtractor()

        current_time = time.time()
        # Add 5 events over time
        for i in range(5):
            extractor._update_actor_timestamps("test_user", current_time + i)

        # Check all timestamps stored
        assert len(extractor.actor_timestamps["test_user"]) == 5
        assert extractor.actor_last_event_time["test_user"] == current_time + 4

    def test_timestamp_cleanup(self):
        """Test that old timestamps are cleaned up."""
        extractor = GitHubFeatureExtractor()

        current_time = time.time()
        time_window = service_settings.velocity_time_window

        # Add events outside time window
        for i in range(5):
            extractor._update_actor_timestamps(
                "test_user", current_time - time_window - 100 + i
            )

        # Add current event (triggers cleanup)
        extractor._update_actor_timestamps("test_user", current_time)

        # Old events should be cleaned up
        timestamps = list(extractor.actor_timestamps["test_user"])
        assert all(ts >= current_time - time_window for ts in timestamps)

    def test_time_based_features_no_history(self):
        """Test time-based features with no history."""
        extractor = GitHubFeatureExtractor()

        current_time = time.time()
        features = extractor._get_time_based_features("new_user", current_time)

        # Should return zeros
        assert features.shape == (5,)
        assert np.all(features == 0)

    def test_time_based_features_single_event(self):
        """Test time-based features with single event."""
        extractor = GitHubFeatureExtractor()

        current_time = time.time()
        extractor._update_actor_timestamps("test_user", current_time)

        features = extractor._get_time_based_features("test_user", current_time)

        # Feature 0: Events in window should be 1
        assert features[0] == 1
        # Other features should be 0 (need at least 2 events)
        assert features[1] == 0  # avg inter-event time
        assert features[2] == 0  # std dev
        assert features[3] == 0  # time since last

    def test_time_based_features_multiple_events(self):
        """Test time-based features with multiple events."""
        extractor = GitHubFeatureExtractor()

        current_time = time.time()
        # Add 10 events, 10 seconds apart
        for i in range(10):
            extractor._update_actor_timestamps("test_user", current_time + i * 10)

        features = extractor._get_time_based_features("test_user", current_time + 90)

        # Feature 0: Events in window (depends on window size)
        assert features[0] > 0
        # Feature 1: Average inter-event time should be ~10 seconds
        assert 9 <= features[1] <= 11
        # Feature 2: Std dev should be low (consistent timing)
        assert features[2] < 5
        # Feature 3: Time since last event
        assert features[3] == 0  # Last event is current time

    def test_velocity_score_calculation(self):
        """Test velocity score (events per minute) calculation."""
        extractor = GitHubFeatureExtractor()

        current_time = time.time()
        # Add 20 events in 60 seconds (1 per 3 seconds)
        for i in range(20):
            extractor._update_actor_timestamps("test_user", current_time + i * 3)

        features = extractor._get_time_based_features("test_user", current_time + 60)

        # Feature 4: Velocity score should be ~20 events/min
        assert 18 <= features[4] <= 22


class TestVelocityAnomalyDetection:
    """Test velocity-based anomaly detection."""

    def test_no_history(self):
        """Test velocity anomaly with no history."""
        extractor = GitHubFeatureExtractor()

        current_time = time.time()
        velocity_score, is_inhuman, reason = extractor.get_velocity_anomaly_score(
            "new_user", current_time
        )

        assert velocity_score == 0.0
        assert is_inhuman is False
        assert "No history" in reason

    def test_normal_velocity(self):
        """Test normal human velocity (below threshold)."""
        extractor = GitHubFeatureExtractor()

        current_time = time.time()
        # Add 10 events over 5 minutes (2 events/min - normal)
        for i in range(10):
            extractor._update_actor_timestamps("test_user", current_time + i * 30)

        velocity_score, is_inhuman, reason = extractor.get_velocity_anomaly_score(
            "test_user", current_time + 300
        )

        # Should be flagged as normal (< 20 events/min)
        assert velocity_score < service_settings.velocity_threshold_per_min
        assert is_inhuman is False
        assert "events" in reason

    def test_inhuman_velocity(self):
        """Test inhuman velocity (above threshold)."""
        extractor = GitHubFeatureExtractor()

        current_time = time.time()
        # Add 100 events in 5 minutes (20 events/min - exactly at threshold)
        # Add 110 to exceed threshold
        for i in range(110):
            extractor._update_actor_timestamps("test_user", current_time + i * 2.7)

        velocity_score, is_inhuman, reason = extractor.get_velocity_anomaly_score(
            "test_user", current_time + 300
        )

        # Should be flagged as inhuman (> 20 events/min)
        assert velocity_score > service_settings.velocity_threshold_per_min
        assert is_inhuman is True
        assert "EXCEEDS" in reason
        assert str(service_settings.velocity_threshold_per_min) in reason

    def test_velocity_reason_format(self):
        """Test that velocity reason is properly formatted."""
        extractor = GitHubFeatureExtractor()

        current_time = time.time()
        # Add 50 events over 2 minutes
        for i in range(50):
            extractor._update_actor_timestamps("test_user", current_time + i * 2.4)

        velocity_score, is_inhuman, reason = extractor.get_velocity_anomaly_score(
            "test_user", current_time + 120
        )

        # Reason should contain key information
        assert "events" in reason
        assert "events/min" in reason
        assert "avg" in reason


class TestStreamingAnomalyDetectorVelocity:
    """Test velocity integration in StreamingAnomalyDetector."""

    def create_test_event(
        self, actor_login: str, event_id: str, timestamp: str
    ) -> Event:
        """Create a test event with given parameters."""
        return Event(
            id=event_id,
            type="PushEvent",
            actor=Actor(id=12345, login=actor_login, display_login=actor_login),
            repo=Repo(id=67890, name="test/repo"),
            created_at=timestamp,
            payload={},
        )

    def test_process_event_returns_velocity_info(self):
        """Test that process_event returns velocity information."""
        detector = StreamingAnomalyDetector()

        # Create test event
        event = self.create_test_event(
            "test_user",
            "event1",
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )

        # Process event
        result = detector.process_event(event)

        # Should return 6 values
        assert len(result) == 6
        score, patterns, features, velocity_score, is_inhuman, velocity_reason = result

        # Check types
        assert isinstance(velocity_score, float)
        assert isinstance(is_inhuman, bool)
        assert isinstance(velocity_reason, str)

    def test_is_anomaly_with_velocity(self):
        """Test that is_anomaly considers velocity flag."""
        detector = StreamingAnomalyDetector()

        # Normal CoDisp score but inhuman velocity should trigger anomaly
        is_anomaly = detector.is_anomaly(
            score=50.0,  # Below threshold
            patterns=[],
            is_inhuman_speed=True,  # But velocity is high
        )

        assert is_anomaly is True

    def test_is_anomaly_without_velocity(self):
        """Test that is_anomaly works without velocity flag."""
        detector = StreamingAnomalyDetector()

        # High CoDisp score should trigger anomaly
        is_anomaly = detector.is_anomaly(
            score=70.0,  # Above threshold
            patterns=[],
            is_inhuman_speed=False,
        )

        assert is_anomaly is True

    def test_rapid_events_detection(self):
        """Test detection of rapid event generation."""
        detector = StreamingAnomalyDetector()

        # Create 25 events in rapid succession (1 per second)
        current_time = datetime.now(timezone.utc)

        for i in range(25):
            # Create event with incrementing timestamp
            event_time = current_time.timestamp() + i
            event = self.create_test_event(
                "bot_user",
                f"event{i}",
                datetime.fromtimestamp(event_time, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            )

            # Process event
            score, patterns, features, velocity_score, is_inhuman, velocity_reason = (
                detector.process_event(event)
            )

            # After enough events, should detect high velocity
            if i >= 20:
                # 21+ events in ~20 seconds = ~60 events/min >> threshold
                assert velocity_score > 0

        # Final event should show inhuman velocity
        assert is_inhuman is True


class TestFeatureVectorIntegration:
    """Test that time-based features are properly integrated into feature vector."""

    def test_feature_vector_size(self):
        """Test that feature vector has correct size with time features."""
        extractor = GitHubFeatureExtractor()

        # Create test event
        event = Event(
            id="test123",
            type="PushEvent",
            actor=Actor(id=12345, login="test_user", display_login="test_user"),
            repo=Repo(id=67890, name="test/repo"),
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            payload={},
        )

        # Extract features
        features = extractor.extract_features(event)

        # Should be 304 dimensions (299 + 5 time-based)
        assert features is not None
        assert features.shape == (304,)

    def test_time_features_in_vector(self):
        """Test that time features appear in feature vector."""
        extractor = GitHubFeatureExtractor()

        # Create and process multiple events
        current_time = datetime.now(timezone.utc)
        for i in range(5):
            event_time = current_time.timestamp() + i * 10
            event = Event(
                id=f"event{i}",
                type="PushEvent",
                actor=Actor(id=12345, login="test_user", display_login="test_user"),
                repo=Repo(id=67890, name="test/repo"),
                created_at=datetime.fromtimestamp(event_time, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                payload={},
            )
            features = extractor.extract_features(event)

        # Last 5 features should be time-based (non-zero for 5+ events)
        assert features is not None
        time_features = features[-5:]

        # At least some time features should be non-zero
        assert np.any(time_features != 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
