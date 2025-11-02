"""Test velocity calculation fix for zero time span."""

import pytest

from github_client.feature_extractor import GitHubFeatureExtractor


@pytest.mark.unit
def test_velocity_with_same_timestamp():
    """Test that velocity calculation doesn't produce infinity for events at same time."""
    extractor = GitHubFeatureExtractor()

    # Simulate events at the same timestamp
    actor = "test_user"
    timestamp = 1000.0

    # Add first event
    extractor._update_actor_timestamps(actor, timestamp)

    # Add second event at exact same timestamp (simulating startup load)
    extractor._update_actor_timestamps(actor, timestamp)

    # Get velocity score
    velocity_score, is_inhuman, reason = extractor.get_velocity_anomaly_score(
        actor, timestamp
    )

    # Verify we don't get infinity
    assert velocity_score != float("inf"), "Velocity should not be infinity"
    assert velocity_score > 0, "Velocity should be positive"

    # With 2 events in 0s (clamped to 1s), we should get 120 events/min
    expected_velocity = 120.0  # 2 events / 1 second * 60
    assert abs(velocity_score - expected_velocity) < 0.1, (
        f"Expected ~{expected_velocity}, got {velocity_score}"
    )
