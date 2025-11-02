"""Unit tests for database models and conversions.

Tests include:
- GitHubEvent.to_dict() serialization
- GitHubEvent.to_event_dict() reconstruction
- AnomalySummary.to_dict() serialization
- AnomalySummary.to_response() Pydantic conversion
"""

from datetime import datetime, timezone

import pytest

from service.database import GitHubEvent, AnomalySummary


@pytest.mark.unit
class TestGitHubEventModel:
    """Tests for GitHubEvent database model."""

    def test_to_dict_with_all_fields(self):
        """Test to_dict() serialization with all fields."""
        now = datetime.now(timezone.utc)

        event = GitHubEvent(
            id="12345",
            event_type="PushEvent",
            actor_login="testuser",
            actor_id=123,
            repo_name="owner/repo",
            repo_id=456,
            org_login="test-org",
            created_at=now,
            ingested_at=now,
            payload={"ref": "refs/heads/main"},
            processed=True,
            is_anomaly=False,
            anomaly_score=0.5,
        )

        result = event.to_dict()

        assert result["id"] == "12345"
        assert result["event_type"] == "PushEvent"
        assert result["actor_login"] == "testuser"
        assert result["actor_id"] == 123
        assert result["repo_name"] == "owner/repo"
        assert result["repo_id"] == 456
        assert result["org_login"] == "test-org"
        assert result["processed"] is True
        assert result["is_anomaly"] is False
        assert result["anomaly_score"] == 0.5
        assert "created_at" in result
        assert "ingested_at" in result

    def test_to_dict_without_org(self):
        """Test to_dict() serialization without organization."""
        now = datetime.now(timezone.utc)

        event = GitHubEvent(
            id="12345",
            event_type="PushEvent",
            actor_login="testuser",
            actor_id=123,
            repo_name="owner/repo",
            repo_id=456,
            org_login=None,  # No org
            created_at=now,
            ingested_at=now,
            payload={},
            processed=False,
            is_anomaly=False,
            anomaly_score=None,
        )

        result = event.to_dict()

        assert result["org_login"] is None
        assert result["anomaly_score"] is None

    def test_to_event_dict_with_org(self):
        """Test to_event_dict() reconstruction with organization."""
        now = datetime.now(timezone.utc)

        event = GitHubEvent(
            id="12345",
            event_type="PushEvent",
            actor_login="testuser",
            actor_id=123,
            repo_name="owner/repo",
            repo_id=456,
            org_login="test-org",
            created_at=now,
            ingested_at=now,
            payload={"ref": "refs/heads/main", "size": 1},
            processed=True,
            is_anomaly=False,
            anomaly_score=0.5,
        )

        result = event.to_event_dict()

        # Check Event structure
        assert result["id"] == "12345"
        assert result["type"] == "PushEvent"
        assert result["public"] is True

        # Check actor reconstruction
        assert result["actor"]["id"] == 123
        assert result["actor"]["login"] == "testuser"
        assert result["actor"]["display_login"] == "testuser"
        assert "api.github.com/users" in result["actor"]["url"]

        # Check repo reconstruction
        assert result["repo"]["id"] == 456
        assert result["repo"]["name"] == "owner/repo"
        assert "api.github.com/repos" in result["repo"]["url"]

        # Check org reconstruction
        assert "org" in result
        assert result["org"]["login"] == "test-org"
        assert "api.github.com/orgs" in result["org"]["url"]

        # Check payload
        assert result["payload"]["ref"] == "refs/heads/main"

    def test_to_event_dict_without_org(self):
        """Test to_event_dict() reconstruction without organization."""
        now = datetime.now(timezone.utc)

        event = GitHubEvent(
            id="67890",
            event_type="IssuesEvent",
            actor_login="developer",
            actor_id=789,
            repo_name="myorg/myrepo",
            repo_id=999,
            org_login=None,  # No org
            created_at=now,
            ingested_at=now,
            payload={"action": "opened"},
            processed=False,
            is_anomaly=True,
            anomaly_score=10.5,
        )

        result = event.to_event_dict()

        # Should not have org field
        assert "org" not in result

        # Other fields should be correct
        assert result["type"] == "IssuesEvent"
        assert result["actor"]["login"] == "developer"
        assert result["repo"]["name"] == "myorg/myrepo"


@pytest.mark.unit
class TestAnomalySummaryModel:
    """Tests for AnomalySummary database model."""

    def test_to_dict_all_fields(self):
        """Test to_dict() serialization with all fields."""
        event_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        created_time = datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc)

        summary = AnomalySummary(
            id=1,
            event_id="12345",
            title="Suspicious Activity",
            severity="high",
            severity_reasoning="New account with force push",
            root_cause=["Force push detected", "New account", "No commit history"],
            impact=["Code overwrite", "Lost signatures", "Potential backdoor"],
            next_steps=["Review commits", "Verify identity", "Rollback if needed"],
            suspicious_patterns=["High anomaly score", "Force push", "New account"],
            anomaly_score=15.5,
            event_type="PushEvent",
            actor_login="suspicious-user",
            repo_name="org/critical-repo",
            raw_event={"id": "12345", "type": "PushEvent"},
            event_timestamp=event_time,
            created_at=created_time,
            tags=["security", "force-push"],
        )

        result = summary.to_dict()

        assert result["id"] == 1
        assert result["event_id"] == "12345"
        assert result["title"] == "Suspicious Activity"
        assert result["severity"] == "high"
        assert result["severity_reasoning"] == "New account with force push"
        assert len(result["root_cause"]) == 3
        assert len(result["impact"]) == 3
        assert len(result["next_steps"]) == 3
        assert result["anomaly_score"] == 15.5
        assert result["event_type"] == "PushEvent"
        assert result["actor_login"] == "suspicious-user"
        assert result["repo_name"] == "org/critical-repo"
        assert result["event_timestamp"] == "2025-01-01T12:00:00+00:00"
        assert result["created_at"] == "2025-01-01T12:01:00+00:00"
        assert len(result["tags"]) == 2

    def test_to_response_pydantic_conversion(self):
        """Test to_response() converts to Pydantic model."""
        event_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        created_time = datetime(2025, 1, 1, 12, 1, 0, tzinfo=timezone.utc)

        summary = AnomalySummary(
            id=2,
            event_id="67890",
            title="Test Alert",
            severity="medium",
            severity_reasoning="Unusual pattern",
            root_cause=["Pattern 1", "Pattern 2", "Pattern 3"],
            impact=["Impact 1", "Impact 2", "Impact 3"],
            next_steps=["Step 1", "Step 2", "Step 3"],
            suspicious_patterns=["Pattern A"],
            anomaly_score=8.0,
            event_type="DeleteEvent",
            actor_login="testuser",
            repo_name="test/repo",
            raw_event={"id": "67890"},
            event_timestamp=event_time,
            created_at=created_time,
            tags=["test"],
        )

        response = summary.to_response()

        # Check Pydantic model fields
        assert response.id == 2
        assert response.event_id == "67890"
        assert response.title == "Test Alert"
        assert response.severity.value == "medium"  # Enum
        assert response.severity_reasoning == "Unusual pattern"
        assert response.anomaly_score == 8.0
        assert response.event_type == "DeleteEvent"
        assert response.actor_login == "testuser"
        assert response.repo_name == "test/repo"
        assert response.event_timestamp == event_time
        assert response.created_at == created_time

    def test_to_response_all_severity_levels(self):
        """Test to_response() with all severity levels."""
        event_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        for severity_level in ["low", "medium", "high", "critical"]:
            summary = AnomalySummary(
                id=1,
                event_id="123",
                title="Test",
                severity=severity_level,
                severity_reasoning=f"Test reasoning for {severity_level}",
                root_cause=["a", "b", "c"],
                impact=["a", "b", "c"],
                next_steps=["a", "b", "c"],
                suspicious_patterns=[],
                anomaly_score=1.0,
                event_type="PushEvent",
                actor_login="user",
                repo_name="repo",
                raw_event={},
                event_timestamp=event_time,
                created_at=event_time,
                tags=[],
            )

            response = summary.to_response()
            assert response.severity.value == severity_level

    def test_to_dict_backward_compatibility(self):
        """Test that to_dict() maintains backward compatibility."""
        event_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        summary = AnomalySummary(
            id=10,
            event_id="test123",
            title="Backward Compat Test",
            severity="low",
            severity_reasoning=None,
            root_cause=["a", "b", "c"],
            impact=["x", "y", "z"],
            next_steps=["1", "2", "3"],
            suspicious_patterns=["pattern1"],
            anomaly_score=2.0,
            event_type="WatchEvent",
            actor_login="watcher",
            repo_name="watched/repo",
            raw_event={"id": "test123"},
            event_timestamp=event_time,
            created_at=event_time,
            tags=[],
        )

        result = summary.to_dict()

        # Verify all expected fields are present (backward compatibility)
        expected_fields = [
            "id",
            "event_id",
            "title",
            "severity",
            "severity_reasoning",
            "root_cause",
            "impact",
            "next_steps",
            "suspicious_patterns",
            "anomaly_score",
            "event_type",
            "actor_login",
            "repo_name",
            "raw_event",
            "event_timestamp",
            "created_at",
            "tags",
        ]

        for field in expected_fields:
            assert field in result, f"Field {field} missing from to_dict() output"

    def test_model_with_empty_optional_fields(self):
        """Test models with empty lists and minimal data."""
        event_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        summary = AnomalySummary(
            id=5,
            event_id="empty123",
            title="Minimal Data",
            severity="low",
            severity_reasoning="Minimal activity",  # Required field
            root_cause=["a", "b", "c"],
            impact=["a", "b", "c"],
            next_steps=["a", "b", "c"],
            suspicious_patterns=[],  # Empty list
            anomaly_score=0.1,
            event_type="WatchEvent",
            actor_login="user",
            repo_name="repo",
            raw_event={},  # Empty dict
            event_timestamp=event_time,
            created_at=event_time,
            tags=[],  # Empty list
        )

        # Should not raise errors
        dict_result = summary.to_dict()
        assert dict_result["suspicious_patterns"] == []
        assert dict_result["tags"] == []

        response_result = summary.to_response()
        assert response_result.suspicious_patterns == []
        assert response_result.tags == []
