"""Test script for SSE models to verify type safety and backward compatibility."""

import json
from datetime import datetime

from service.sse_models import (
    AnomalyMessage,
    AnomalySummaryResponse,
    ConnectedMessage,
    SSEMessage,
)


def test_connected_message():
    """Test ConnectedMessage model."""
    print("Testing ConnectedMessage...")

    # Create with defaults
    msg = ConnectedMessage()
    assert msg.type == "connected"
    assert msg.message == "Stream connected"

    # Serialize to JSON
    json_str = msg.model_dump_json()
    json_obj = json.loads(json_str)
    assert json_obj == {"type": "connected", "message": "Stream connected"}

    print("✓ ConnectedMessage works correctly")


def test_anomaly_summary_response():
    """Test AnomalySummaryResponse model."""
    print("\nTesting AnomalySummaryResponse...")

    # Create with sample data
    response = AnomalySummaryResponse(
        id=42,
        event_id="12345678901",
        title="Test Anomaly",
        severity="high",
        root_cause=["Cause 1", "Cause 2", "Cause 3"],
        impact=["Impact 1", "Impact 2", "Impact 3"],
        next_steps=["Step 1", "Step 2", "Step 3"],
        suspicious_patterns=["Pattern 1", "Pattern 2"],
        anomaly_score=12.3,
        event_type="PushEvent",
        actor_login="testuser",
        repo_name="test/repo",
        raw_event={"id": "123", "type": "PushEvent"},
        event_timestamp=datetime(2025, 1, 1, 12, 0, 0),
        created_at=datetime(2025, 1, 1, 12, 1, 30),
        tags=["security", "test"],
    )

    # Validate fields
    assert response.id == 42
    assert response.severity == "high"
    assert len(response.root_cause) == 3
    assert response.anomaly_score == 12.3

    # Serialize to JSON
    json_obj = response.model_dump(mode="json")
    assert json_obj["id"] == 42
    assert json_obj["severity"] == "high"
    assert json_obj["event_timestamp"] == "2025-01-01T12:00:00"
    assert json_obj["created_at"] == "2025-01-01T12:01:30"

    print("✓ AnomalySummaryResponse works correctly")


def test_anomaly_message():
    """Test AnomalyMessage model."""
    print("\nTesting AnomalyMessage...")

    # Create nested structure
    summary = AnomalySummaryResponse(
        id=42,
        event_id="12345678901",
        title="Test Anomaly",
        severity="high",
        root_cause=["Cause 1", "Cause 2", "Cause 3"],
        impact=["Impact 1", "Impact 2", "Impact 3"],
        next_steps=["Step 1", "Step 2", "Step 3"],
        suspicious_patterns=["Pattern 1"],
        anomaly_score=12.3,
        event_type="PushEvent",
        actor_login="testuser",
        repo_name="test/repo",
        raw_event={"id": "123", "type": "PushEvent"},
        event_timestamp=datetime(2025, 1, 1, 12, 0, 0),
        created_at=datetime(2025, 1, 1, 12, 1, 30),
        tags=["security"],
    )

    msg = AnomalyMessage(data=summary)
    assert msg.type == "anomaly"
    assert msg.data.id == 42

    # Serialize to JSON (this is what gets sent via SSE)
    json_obj = msg.model_dump(mode="json")
    assert json_obj["type"] == "anomaly"
    assert json_obj["data"]["id"] == 42
    assert json_obj["data"]["severity"] == "high"
    assert json_obj["data"]["event_timestamp"] == "2025-01-01T12:00:00"

    print("✓ AnomalyMessage works correctly")


def test_severity_validation():
    """Test that invalid severity values are rejected."""
    print("\nTesting severity validation...")

    try:
        AnomalySummaryResponse(
            id=1,
            event_id="123",
            title="Test",
            severity="invalid",  # Should fail
            root_cause=["a", "b", "c"],
            impact=["a", "b", "c"],
            next_steps=["a", "b", "c"],
            suspicious_patterns=[],
            anomaly_score=1.0,
            event_type="PushEvent",
            actor_login="user",
            repo_name="repo",
            raw_event={},
            event_timestamp=datetime.now(),
            created_at=datetime.now(),
            tags=[],
        )
        assert False, "Should have raised validation error"
    except Exception as e:
        assert "severity" in str(e).lower()
        print("✓ Invalid severity correctly rejected")


def test_backward_compatibility():
    """Test that the new models produce the same JSON structure as the old dict-based approach."""
    print("\nTesting backward compatibility...")

    # Old approach (dict)
    old_dict = {
        "id": 42,
        "event_id": "12345678901",
        "title": "Test",
        "severity": "high",
        "root_cause": ["a", "b", "c"],
        "impact": ["a", "b", "c"],
        "next_steps": ["a", "b", "c"],
        "suspicious_patterns": ["p1"],
        "anomaly_score": 12.3,
        "event_type": "PushEvent",
        "actor_login": "user",
        "repo_name": "repo/name",
        "raw_event": {"id": "123"},
        "event_timestamp": "2025-01-01T12:00:00",
        "created_at": "2025-01-01T12:01:30",
        "tags": ["tag1"],
    }

    # New approach (Pydantic model)
    new_model = AnomalySummaryResponse(
        id=42,
        event_id="12345678901",
        title="Test",
        severity="high",
        root_cause=["a", "b", "c"],
        impact=["a", "b", "c"],
        next_steps=["a", "b", "c"],
        suspicious_patterns=["p1"],
        anomaly_score=12.3,
        event_type="PushEvent",
        actor_login="user",
        repo_name="repo/name",
        raw_event={"id": "123"},
        event_timestamp=datetime(2025, 1, 1, 12, 0, 0),
        created_at=datetime(2025, 1, 1, 12, 1, 30),
        tags=["tag1"],
    )
    new_dict = new_model.model_dump(mode="json")

    # Compare all keys
    assert set(old_dict.keys()) == set(new_dict.keys()), "Keys don't match"

    # Compare all values
    for key in old_dict.keys():
        assert old_dict[key] == new_dict[key], f"Value mismatch for key: {key}"

    print("✓ Backward compatibility maintained - output format unchanged")


def test_sse_message_format():
    """Test that SSE messages match the expected format."""
    print("\nTesting SSE message format...")

    # Connected message
    connected = ConnectedMessage()
    connected_json = json.loads(connected.model_dump_json())
    assert connected_json == {"type": "connected", "message": "Stream connected"}

    # Anomaly message
    summary = AnomalySummaryResponse(
        id=1,
        event_id="123",
        title="Test",
        severity="low",
        root_cause=["a", "b", "c"],
        impact=["a", "b", "c"],
        next_steps=["a", "b", "c"],
        suspicious_patterns=[],
        anomaly_score=1.0,
        event_type="PushEvent",
        actor_login="user",
        repo_name="repo",
        raw_event={},
        event_timestamp=datetime(2025, 1, 1, 12, 0, 0),
        created_at=datetime(2025, 1, 1, 12, 0, 0),
        tags=[],
    )
    anomaly = AnomalyMessage(data=summary)
    anomaly_json = anomaly.model_dump(mode="json")

    assert anomaly_json["type"] == "anomaly"
    assert "data" in anomaly_json
    assert anomaly_json["data"]["id"] == 1

    print("✓ SSE message format is correct")


if __name__ == "__main__":
    print("=" * 60)
    print("SSE Models Test Suite")
    print("=" * 60)

    test_connected_message()
    test_anomaly_summary_response()
    test_anomaly_message()
    test_severity_validation()
    test_backward_compatibility()
    test_sse_message_format()

    print("\n" + "=" * 60)
    print("✓ All tests passed! SSE models are working correctly.")
    print("=" * 60)
