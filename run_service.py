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

import uvicorn

from service.config import service_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_worker(worker_num: int):
    """Run a single RQ worker process.

    Args:
        worker_num: Worker number for identification
    """
    # Import here to avoid issues with multiprocessing
    from rq import Worker

    from service.queue import anomaly_queue, redis_conn

    logger.info(f"Starting RQ worker #{worker_num}")

    try:
        worker = Worker(
            [anomaly_queue],
            connection=redis_conn,
            name=f"worker-{worker_num}",
        )
        worker.work(with_scheduler=True)
    except Exception as e:
        logger.exception(f"Worker #{worker_num} crashed: {e}")
        sys.exit(1)


def run_api_server(host: str, port: int):
    """Run FastAPI server.

    Args:
        host: Server host
        port: Server port
    """
    logger.info(f"Starting FastAPI server on {host}:{port}")

    uvicorn.run(
        "service.app:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GitHub Anomaly Monitoring Service"
    )
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
    if service_settings.ai_provider == "anthropic" and not service_settings.anthropic_api_key:
        logger.warning("⚠️  Anthropic API key not set - will use fallback summaries")
    elif service_settings.ai_provider == "openai" and not service_settings.openai_api_key:
        logger.warning("⚠️  OpenAI API key not set - will use fallback summaries")

    if not service_settings.github_token:
        logger.warning("⚠️  GitHub token not set - rate limits will be stricter")

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
