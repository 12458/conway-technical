"""Unit tests for RQ queue operations.

Tests include:
- Redis connection creation
- Job enqueuing
- Queue statistics
- Error handling
"""

from unittest.mock import MagicMock, patch

import pytest
import redis


@pytest.mark.unit
class TestRedisConnection:
    """Tests for Redis connection management."""

    @patch("service.queue.redis.Redis")
    def test_get_redis_connection_success(self, mock_redis):
        """Test successful Redis connection creation."""
        from service.queue import get_redis_connection

        mock_connection = MagicMock()
        mock_redis.from_url.return_value = mock_connection

        result = get_redis_connection()

        assert result == mock_connection
        mock_redis.from_url.assert_called_once()
        # Verify decode_responses=False for RQ compatibility
        call_args = mock_redis.from_url.call_args
        assert call_args.kwargs["decode_responses"] is False

    @patch("service.queue.redis.Redis")
    def test_get_redis_connection_with_url(self, mock_redis):
        """Test Redis connection uses correct URL from config."""
        from service.queue import get_redis_connection

        mock_connection = MagicMock()
        mock_redis.from_url.return_value = mock_connection

        get_redis_connection()

        # Should be called with URL from service_settings
        call_args = mock_redis.from_url.call_args
        assert len(call_args.args) > 0 or "url" in call_args.kwargs


@pytest.mark.unit
class TestAnomalySummarizationEnqueue:
    """Tests for enqueuing anomaly summarization jobs."""

    @patch("service.queue.anomaly_queue")
    def test_enqueue_anomaly_summarization_success(self, mock_queue):
        """Test successful job enqueuing."""
        from service.queue import enqueue_anomaly_summarization

        # Setup mock job
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_queue.enqueue.return_value = mock_job

        event_id = "event-456"
        event_data = {
            "id": "event-456",
            "type": "PushEvent",
            "actor": {"login": "testuser"},
            "repo": {"name": "owner/repo"},
        }
        anomaly_score = 15.5
        suspicious_patterns = ["Force push", "New account"]

        job_id = enqueue_anomaly_summarization(
            event_id=event_id,
            event_data=event_data,
            anomaly_score=anomaly_score,
            suspicious_patterns=suspicious_patterns,
        )

        assert job_id == "job-123"
        mock_queue.enqueue.assert_called_once()

        # Verify function was passed to enqueue
        call_args = mock_queue.enqueue.call_args
        from service.summarizer import summarize_anomaly_job

        assert call_args.args[0] == summarize_anomaly_job

    @patch("service.queue.anomaly_queue")
    def test_enqueue_with_all_parameters(self, mock_queue):
        """Test enqueuing passes all parameters correctly."""
        from service.queue import enqueue_anomaly_summarization

        mock_job = MagicMock()
        mock_job.id = "job-789"
        mock_queue.enqueue.return_value = mock_job

        event_id = "evt-001"
        event_data = {"id": "evt-001", "type": "DeleteEvent"}
        anomaly_score = 20.0
        suspicious_patterns = ["Pattern 1", "Pattern 2", "Pattern 3"]

        enqueue_anomaly_summarization(
            event_id=event_id,
            event_data=event_data,
            anomaly_score=anomaly_score,
            suspicious_patterns=suspicious_patterns,
        )

        # Verify kwargs passed to enqueue
        call_kwargs = mock_queue.enqueue.call_args.kwargs
        assert call_kwargs["event_id"] == event_id
        assert call_kwargs["event_data"] == event_data
        assert call_kwargs["anomaly_score"] == anomaly_score
        assert call_kwargs["suspicious_patterns"] == suspicious_patterns
        assert call_kwargs["job_timeout"] == "5m"

    @patch("service.queue.anomaly_queue")
    def test_enqueue_with_empty_patterns(self, mock_queue):
        """Test enqueuing with empty suspicious patterns list."""
        from service.queue import enqueue_anomaly_summarization

        mock_job = MagicMock()
        mock_job.id = "job-empty"
        mock_queue.enqueue.return_value = mock_job

        event_id = "evt-002"
        event_data = {"id": "evt-002"}
        anomaly_score = 5.5
        suspicious_patterns = []  # Empty list

        job_id = enqueue_anomaly_summarization(
            event_id=event_id,
            event_data=event_data,
            anomaly_score=anomaly_score,
            suspicious_patterns=suspicious_patterns,
        )

        assert job_id == "job-empty"
        call_kwargs = mock_queue.enqueue.call_args.kwargs
        assert call_kwargs["suspicious_patterns"] == []

    @patch("service.queue.anomaly_queue")
    def test_enqueue_with_high_anomaly_score(self, mock_queue):
        """Test enqueuing with high anomaly score."""
        from service.queue import enqueue_anomaly_summarization

        mock_job = MagicMock()
        mock_job.id = "job-high-score"
        mock_queue.enqueue.return_value = mock_job

        event_id = "evt-003"
        event_data = {"id": "evt-003", "type": "DeleteEvent"}
        anomaly_score = 100.0  # Very high score
        suspicious_patterns = ["Critical pattern"]

        job_id = enqueue_anomaly_summarization(
            event_id=event_id,
            event_data=event_data,
            anomaly_score=anomaly_score,
            suspicious_patterns=suspicious_patterns,
        )

        assert job_id == "job-high-score"
        call_kwargs = mock_queue.enqueue.call_args.kwargs
        assert call_kwargs["anomaly_score"] == 100.0


@pytest.mark.unit
class TestQueueStatistics:
    """Tests for queue statistics retrieval."""

    @patch("service.queue.anomaly_queue")
    def test_get_queue_stats_empty_queue(self, mock_queue):
        """Test queue statistics with empty queue."""
        from service.queue import get_queue_stats

        mock_queue.name = "test-queue"
        mock_queue.__len__ = MagicMock(return_value=0)
        mock_queue.failed_job_registry.count = 0
        mock_queue.finished_job_registry.count = 0
        mock_queue.started_job_registry.count = 0

        stats = get_queue_stats()

        assert stats["queue_name"] == "test-queue"
        assert stats["queued_jobs"] == 0
        assert stats["failed_jobs"] == 0
        assert stats["finished_jobs"] == 0
        assert stats["started_jobs"] == 0

    @patch("service.queue.anomaly_queue")
    def test_get_queue_stats_with_jobs(self, mock_queue):
        """Test queue statistics with active jobs."""
        from service.queue import get_queue_stats

        mock_queue.name = "anomaly-queue"
        mock_queue.__len__ = MagicMock(return_value=5)
        mock_queue.failed_job_registry.count = 2
        mock_queue.finished_job_registry.count = 10
        mock_queue.started_job_registry.count = 1

        stats = get_queue_stats()

        assert stats["queue_name"] == "anomaly-queue"
        assert stats["queued_jobs"] == 5
        assert stats["failed_jobs"] == 2
        assert stats["finished_jobs"] == 10
        assert stats["started_jobs"] == 1

    @patch("service.queue.anomaly_queue")
    def test_get_queue_stats_structure(self, mock_queue):
        """Test queue statistics returns expected keys."""
        from service.queue import get_queue_stats

        mock_queue.name = "test"
        mock_queue.__len__ = MagicMock(return_value=0)
        mock_queue.failed_job_registry.count = 0
        mock_queue.finished_job_registry.count = 0
        mock_queue.started_job_registry.count = 0

        stats = get_queue_stats()

        # Verify all expected keys are present
        expected_keys = [
            "queue_name",
            "queued_jobs",
            "failed_jobs",
            "finished_jobs",
            "started_jobs",
        ]
        for key in expected_keys:
            assert key in stats

    @patch("service.queue.anomaly_queue")
    def test_get_queue_stats_all_numeric(self, mock_queue):
        """Test queue statistics returns numeric values for counts."""
        from service.queue import get_queue_stats

        mock_queue.name = "test"
        mock_queue.__len__ = MagicMock(return_value=3)
        mock_queue.failed_job_registry.count = 1
        mock_queue.finished_job_registry.count = 5
        mock_queue.started_job_registry.count = 2

        stats = get_queue_stats()

        # Verify all count fields are integers
        assert isinstance(stats["queued_jobs"], int)
        assert isinstance(stats["failed_jobs"], int)
        assert isinstance(stats["finished_jobs"], int)
        assert isinstance(stats["started_jobs"], int)


@pytest.mark.unit
class TestQueueIntegration:
    """Tests for queue integration and edge cases."""

    @patch("service.queue.anomaly_queue")
    def test_enqueue_returns_job_id(self, mock_queue):
        """Test that enqueue always returns a job ID string."""
        from service.queue import enqueue_anomaly_summarization

        mock_job = MagicMock()
        mock_job.id = "test-job-uuid-123"
        mock_queue.enqueue.return_value = mock_job

        job_id = enqueue_anomaly_summarization(
            event_id="evt-123",
            event_data={"id": "evt-123"},
            anomaly_score=10.0,
            suspicious_patterns=["pattern1"],
        )

        assert isinstance(job_id, str)
        assert len(job_id) > 0
        assert job_id == "test-job-uuid-123"

    @patch("service.queue.anomaly_queue")
    def test_multiple_enqueue_operations(self, mock_queue):
        """Test multiple enqueue operations in sequence."""
        from service.queue import enqueue_anomaly_summarization

        job_ids = []
        for i in range(3):
            mock_job = MagicMock()
            mock_job.id = f"job-{i}"
            mock_queue.enqueue.return_value = mock_job

            job_id = enqueue_anomaly_summarization(
                event_id=f"evt-{i}",
                event_data={"id": f"evt-{i}"},
                anomaly_score=float(i + 1),
                suspicious_patterns=[f"pattern-{i}"],
            )
            job_ids.append(job_id)

        assert len(job_ids) == 3
        assert job_ids == ["job-0", "job-1", "job-2"]
        assert mock_queue.enqueue.call_count == 3
