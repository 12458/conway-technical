"""Integration test to verify SSE models work with database models."""

from datetime import datetime

from service.database import AnomalySummary
from service.sse_models import AnomalyMessage, ConnectedMessage


def test_database_to_response():
    """Test that AnomalySummary.to_response() works correctly."""
    print("Testing AnomalySummary.to_response() integration...")

    # Create a mock AnomalySummary instance (not saved to DB)
    summary = AnomalySummary(
        id=42,
        event_id="12345678901",
        title="Suspicious Force Push",
        severity="high",
        root_cause=["Force push detected", "New account", "No history"],
        impact=["Code rewrite", "Lost signatures", "Malicious code risk"],
        next_steps=["Review commits", "Verify identity", "Enable protection"],
        suspicious_patterns=["High score", "Force push", "New account"],
        anomaly_score=12.3,
        event_type="PushEvent",
        actor_login="suspicious-user",
        repo_name="acme/repo",
        raw_event={
            "id": "12345678901",
            "type": "PushEvent",
            "actor": {"id": 123, "login": "suspicious-user"},
            "repo": {"id": 456, "name": "acme/repo"},
            "payload": {"ref": "refs/heads/main"},
            "public": True,
            "created_at": "2025-01-01T12:00:00Z",
        },
        event_timestamp=datetime(2025, 1, 1, 12, 0, 0),
        created_at=datetime(2025, 1, 1, 12, 1, 30),
        tags=["security", "risk:high"],
    )

    # Convert to Pydantic response model
    response = summary.to_response()

    # Verify all fields are correct
    assert response.id == 42
    assert response.event_id == "12345678901"
    assert response.title == "Suspicious Force Push"
    assert response.severity == "high"
    assert len(response.root_cause) == 3
    assert len(response.impact) == 3
    assert len(response.next_steps) == 3
    assert response.anomaly_score == 12.3
    assert response.event_type == "PushEvent"
    assert response.actor_login == "suspicious-user"
    assert response.repo_name == "acme/repo"
    assert response.event_timestamp == datetime(2025, 1, 1, 12, 0, 0)
    assert response.created_at == datetime(2025, 1, 1, 12, 1, 30)
    assert len(response.tags) == 2

    print("✓ AnomalySummary.to_response() works correctly")


def test_sse_message_creation():
    """Test creating SSE messages from database models."""
    print("\nTesting SSE message creation from database models...")

    # Create mock database model
    summary = AnomalySummary(
        id=1,
        event_id="123",
        title="Test",
        severity="low",
        root_cause=["a", "b", "c"],
        impact=["a", "b", "c"],
        next_steps=["a", "b", "c"],
        suspicious_patterns=["p1"],
        anomaly_score=1.0,
        event_type="PushEvent",
        actor_login="user",
        repo_name="repo",
        raw_event={},
        event_timestamp=datetime(2025, 1, 1, 12, 0, 0),
        created_at=datetime(2025, 1, 1, 12, 0, 0),
        tags=[],
    )

    # Create AnomalyMessage using to_response()
    anomaly_msg = AnomalyMessage(data=summary.to_response())

    # Serialize to JSON (as would be done in SSE broadcast)
    message_dict = anomaly_msg.model_dump(mode="json")

    # Verify structure
    assert message_dict["type"] == "anomaly"
    assert "data" in message_dict
    assert message_dict["data"]["id"] == 1
    assert message_dict["data"]["title"] == "Test"
    assert message_dict["data"]["severity"] == "low"

    print("✓ SSE message creation from database models works correctly")


def test_connected_message_creation():
    """Test ConnectedMessage creation."""
    print("\nTesting ConnectedMessage creation...")

    # Create connected message
    connected_msg = ConnectedMessage()

    # Serialize as would be done in SSE stream
    json_str = connected_msg.model_dump_json()

    assert "connected" in json_str
    assert "Stream connected" in json_str

    print("✓ ConnectedMessage creation works correctly")


def test_backward_compat_to_dict():
    """Test that to_dict() still works for backward compatibility."""
    print("\nTesting backward compatibility of to_dict()...")

    summary = AnomalySummary(
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

    # Old method should still work
    dict_result = summary.to_dict()

    assert dict_result["id"] == 1
    assert dict_result["title"] == "Test"
    assert dict_result["event_timestamp"] == "2025-01-01T12:00:00"
    assert isinstance(dict_result, dict)

    print("✓ to_dict() backward compatibility maintained")


if __name__ == "__main__":
    print("=" * 60)
    print("SSE Models Integration Test Suite")
    print("=" * 60)

    test_database_to_response()
    test_sse_message_creation()
    test_connected_message_creation()
    test_backward_compat_to_dict()

    print("\n" + "=" * 60)
    print("✓ All integration tests passed!")
    print("=" * 60)
