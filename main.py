"""Main entry point for running the GitHub anomaly monitoring service.

This script starts:
1. The FastAPI web server (with background poller)
2. RQ worker processes for AI summarization

Usage:
    python run_service.py [--workers N] [--host HOST] [--port PORT]
"""

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time
import uuid

import uvicorn

from service.config import service_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cleanup_stale_workers():
    """Clean up stale worker registrations from Redis.

    This prevents crashes when restarting the app with workers that
    didn't properly unregister.
    """
    from rq import Worker
    from rq.worker import WorkerStatus
    from service.queue import redis_conn

    try:
        workers = Worker.all(connection=redis_conn)
        cleaned = 0

        for worker in workers:
            # Check if worker is actually alive by checking its state
            try:
                state = worker.get_state()
                # If worker is idle, busy, or starting, it's alive
                # If it's suspended or we can't get state, clean it up
                if state not in (WorkerStatus.STARTED, WorkerStatus.IDLE, WorkerStatus.BUSY):
                    logger.info(f"Cleaning up stale worker: {worker.name} (state: {state})")
                    worker.register_death()
                    cleaned += 1
            except Exception as e:
                # If we can't get state, worker is likely stale
                logger.info(f"Cleaning up unreachable worker {worker.name}: {e}")
                try:
                    worker.register_death()
                    cleaned += 1
                except Exception:
                    pass

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} stale worker(s)")
        else:
            logger.info("No stale workers found")

    except Exception as e:
        logger.warning(f"Failed to cleanup stale workers: {e}")


def run_worker(worker_num: int):
    """Run a single RQ worker process.

    Args:
        worker_num: Worker number for identification
    """
    # Import here to avoid issues with multiprocessing
    from rq import Worker

    from service.queue import anomaly_queue, redis_conn

    # Generate unique worker name to avoid conflicts on restart
    unique_id = str(uuid.uuid4())[:8]
    worker_name = f"worker-{worker_num}-{unique_id}"

    logger.info(f"Starting RQ worker #{worker_num} with name: {worker_name}")

    worker = None
    try:
        worker = Worker(
            [anomaly_queue],
            connection=redis_conn,
            name=worker_name,
        )

        # Setup signal handlers for graceful shutdown
        def cleanup_handler(signum, frame):
            logger.info(
                f"Worker {worker_name} received signal {signum}, cleaning up..."
            )
            if worker:
                worker.register_death()
            sys.exit(0)

        signal.signal(signal.SIGTERM, cleanup_handler)
        signal.signal(signal.SIGINT, cleanup_handler)

        worker.work(with_scheduler=True)
    except Exception as e:
        logger.exception(f"Worker #{worker_num} ({worker_name}) crashed: {e}")
        if worker:
            try:
                worker.register_death()
            except Exception:
                pass
        sys.exit(1)


def run_api_server(host: str, port: int):
    """Run FastAPI server.

    Args:
        host: Server host
        port: Server port
    """
    logger.info(f"Starting FastAPI server on {host}:{port}")

    try:
        uvicorn.run(
            "service.app:app",
            host=host,
            port=port,
            log_level="info",
            access_log=True,
        )
    except Exception as e:
        logger.exception(f"Failed to start FastAPI server: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GitHub Anomaly Monitoring Service")
    parser.add_argument(
        "--workers",
        type=int,
        default=service_settings.worker_count,
        help=f"Number of RQ worker processes (default: {service_settings.worker_count})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="FastAPI server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="FastAPI server port (default: 8000)",
    )
    parser.add_argument(
        "--workers-only",
        action="store_true",
        help="Run only workers (no API server)",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Run only API server (no workers)",
    )

    args = parser.parse_args()

    # Validate configuration
    logger.info("=" * 60)
    logger.info("GitHub Anomaly Monitoring Service")
    logger.info("=" * 60)
    logger.info(f"Redis URL: {service_settings.redis_url}")
    logger.info(f"Database: {service_settings.database_url}")
    logger.info(f"AI Provider: {service_settings.ai_provider}")
    logger.info(f"AI Model: {service_settings.ai_model}")
    logger.info(f"Anomaly Threshold: {service_settings.anomaly_threshold}")
    logger.info(f"Bot Filtering: {service_settings.enable_bot_filtering}")
    logger.info("=" * 60)

    # Check for required API keys
    if (
        service_settings.ai_provider == "anthropic"
        and not service_settings.anthropic_api_key
    ):
        logger.warning("⚠️  Anthropic API key not set - summarization will fail")
    elif (
        service_settings.ai_provider == "openai" and not service_settings.openai_api_key
    ):
        logger.warning("⚠️  OpenAI API key not set - summarization will fail")

    if not service_settings.github_token:
        logger.warning("⚠️  GitHub token not set - rate limits will be stricter")

    # Validate required GraphQL token for enrichment
    if not service_settings.github_graphql_token:
        logger.error(
            "❌ GitHub GraphQL token is required for anomaly summarization.\n"
            "   Set SERVICE_GITHUB_GRAPHQL_TOKEN environment variable.\n"
            "   Required scopes: repo, read:org, read:user"
        )
        sys.exit(1)

    logger.info("✓ GitHub GraphQL token configured")

    # Clean up any stale workers from previous runs
    cleanup_stale_workers()

    # Start processes
    processes = []

    try:
        # Start workers
        if not args.api_only:
            logger.info(f"Starting {args.workers} RQ worker process(es)")
            for i in range(args.workers):
                p = multiprocessing.Process(
                    target=run_worker,
                    args=(i + 1,),
                    name=f"worker-{i + 1}",
                )
                p.start()
                processes.append(p)
                logger.info(f"Worker #{i + 1} started (PID: {p.pid})")

        # Start API server
        if not args.workers_only:
            # Run in main process
            run_api_server(args.host, args.port)
        else:
            # Just wait for workers
            logger.info("Workers-only mode - API server not started")
            logger.info("Press Ctrl+C to stop")

            # Wait for workers
            for p in processes:
                p.join()

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")

        # Terminate workers
        for p in processes:
            logger.info(f"Terminating {p.name} (PID: {p.pid})")
            p.terminate()

        # Wait for graceful shutdown
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                logger.warning(f"Force killing {p.name}")
                p.kill()

        logger.info("All processes terminated")

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set start method for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
