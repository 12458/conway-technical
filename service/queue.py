"""RQ-based job queue for anomaly summarization."""

import logging
from typing import Any

import redis
from rq import Queue, Retry

from service.config import service_settings

logger = logging.getLogger(__name__)


# Redis connection
def get_redis_connection() -> redis.Redis:
    """Get Redis connection from URL.

    Returns:
        Redis connection instance
    """
    return redis.Redis.from_url(
        service_settings.redis_url,
        decode_responses=False,  # RQ needs bytes mode
    )


# Initialize RQ queue
redis_conn = get_redis_connection()
anomaly_queue = Queue(
    name=service_settings.queue_name,
    connection=redis_conn,
)

logger.info(
    f"Initialized RQ queue '{service_settings.queue_name}' "
    f"with Redis at {service_settings.redis_url}"
)


def enqueue_anomaly_summarization(
    event_id: str,
    event_data: dict[str, Any],
    anomaly_score: float,
    suspicious_patterns: list[str],
) -> str:
    """Enqueue an anomaly for AI summarization.

    Args:
        event_id: Database event ID
        event_data: Full event data dictionary
        anomaly_score: CoDisp score
        suspicious_patterns: List of detected patterns

    Returns:
        RQ job ID
    """
    from service.summarizer import summarize_anomaly_job

    job = anomaly_queue.enqueue(
        summarize_anomaly_job,
        event_id=event_id,
        event_data=event_data,
        anomaly_score=anomaly_score,
        suspicious_patterns=suspicious_patterns,
        job_timeout="5m",  # 5 minute timeout for AI API calls
        retry=Retry(max=3, interval=[10, 30, 60]),  # Retry up to 3 times with exponential backoff
        failure_ttl=86400,  # Keep failed jobs for 24 hours for debugging
    )

    logger.info(
        f"Enqueued summarization job {job.id} for event {event_id} "
        f"(score: {anomaly_score:.2f}, patterns: {len(suspicious_patterns)})"
    )

    return job.id


def get_queue_stats() -> dict[str, Any]:
    """Get queue statistics.

    Returns:
        Dictionary with queue stats
    """
    return {
        "queue_name": anomaly_queue.name,
        "queued_jobs": len(anomaly_queue),
        "failed_jobs": anomaly_queue.failed_job_registry.count,
        "finished_jobs": anomaly_queue.finished_job_registry.count,
        "started_jobs": anomaly_queue.started_job_registry.count,
    }
