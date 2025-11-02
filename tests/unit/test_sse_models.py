"""Test script for SSE models to verify type safety and backward compatibility."""

import json
from datetime import datetime

import pytest

from service.sse_models import (
    AnomalyMessage,
    AnomalySummaryResponse,
    ConnectedMessage,
    SSEMessage,
)


@pytest.mark.unit
class TestConnectedMessage:
    """Tests for ConnectedMessage model."""

    def test_connected_message_defaults(self):
        """Test ConnectedMessage model creation with defaults."""
        msg = ConnectedMessage()
        assert msg.type == "connected"
        assert msg.message == "Stream connected"

    def test_connected_message_serialization(self):
        """Test ConnectedMessage JSON serialization."""
        msg = ConnectedMessage()
        json_str = msg.model_dump_json()
        json_obj = json.loads(json_str)
        assert json_obj == {"type": "connected", "message": "Stream connected"}


@pytest.mark.unit
class TestAnomalySummaryResponse:
    """Tests for AnomalySummaryResponse model."""

    def test_anomaly_summary_response_creation(self):
        """Test AnomalySummaryResponse model with sample data."""
        response = AnomalySummaryResponse(
            id=42,
            event_id="12345678901",
            title="Test Anomaly",
            severity="high",
            severity_reasoning="Classified as high severity due to suspicious pattern detection",
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

    def test_anomaly_summary_response_serialization(self):
        """Test AnomalySummaryResponse JSON serialization."""
        response = AnomalySummaryResponse(
            id=42,
            event_id="12345678901",
            title="Test Anomaly",
            severity="high",
            severity_reasoning="Classified as high severity due to suspicious pattern detection",
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

        # Serialize to JSON
        json_obj = response.model_dump(mode="json")
        assert json_obj["id"] == 42
        assert json_obj["severity"] == "high"
        assert json_obj["event_timestamp"] == "2025-01-01T12:00:00"
        assert json_obj["created_at"] == "2025-01-01T12:01:30"

    def test_severity_validation(self):
        """Test that invalid severity values are rejected."""
        with pytest.raises(Exception) as exc_info:
            AnomalySummaryResponse(
                id=1,
                event_id="123",
                title="Test",
                severity="invalid",  # Should fail
                severity_reasoning="Test reasoning",
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
        assert "severity" in str(exc_info.value).lower()


@pytest.mark.unit
class TestAnomalyMessage:
    """Tests for AnomalyMessage model."""

    def test_anomaly_message_creation(self):
        """Test AnomalyMessage model with nested structure."""
        summary = AnomalySummaryResponse(
            id=42,
            event_id="12345678901",
            title="Test Anomaly",
            severity="high",
            severity_reasoning="Classified as high severity due to suspicious pattern detection",
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

    def test_anomaly_message_serialization(self):
        """Test AnomalyMessage JSON serialization for SSE."""
        summary = AnomalySummaryResponse(
            id=42,
            event_id="12345678901",
            title="Test Anomaly",
            severity="high",
            severity_reasoning="Classified as high severity due to suspicious pattern detection",
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
        # Serialize to JSON (this is what gets sent via SSE)
        json_obj = msg.model_dump(mode="json")
        assert json_obj["type"] == "anomaly"
        assert json_obj["data"]["id"] == 42
        assert json_obj["data"]["severity"] == "high"
        assert json_obj["data"]["event_timestamp"] == "2025-01-01T12:00:00"


@pytest.mark.unit
class TestBackwardCompatibility:
    """Tests for backward compatibility with old dict-based approach."""

    def test_backward_compatibility(self):
        """Test that the new models produce the same JSON structure as the old dict-based approach."""
        # Old approach (dict)
        old_dict = {
            "id": 42,
            "event_id": "12345678901",
            "title": "Test",
            "severity": "high",
            "severity_reasoning": "Test reasoning for severity",
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
            severity_reasoning="Test reasoning for severity",
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


@pytest.mark.unit
class TestSSEMessageFormat:
    """Tests for SSE message format compliance."""

    def test_connected_message_format(self):
        """Test that connected messages match the expected format."""
        connected = ConnectedMessage()
        connected_json = json.loads(connected.model_dump_json())
        assert connected_json == {"type": "connected", "message": "Stream connected"}

    def test_anomaly_message_format(self):
        """Test that anomaly messages match the expected format."""
        summary = AnomalySummaryResponse(
            id=1,
            event_id="123",
            title="Test",
            severity="low",
            severity_reasoning="Classified as low severity",
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
