"""RQ worker entry point for processing jobs."""

import logging
import sys

from rq import Worker

from service.queue import anomaly_queue, redis_conn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_worker():
    """Run RQ worker process."""
    logger.info(f"Starting RQ worker for queue: {anomaly_queue.name}")

    # Create worker
    worker = Worker(
        [anomaly_queue],
        connection=redis_conn,
        name=f"worker-{anomaly_queue.name}",
    )

    # Run worker (blocking)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    run_worker()
