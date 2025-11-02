"""FastAPI application for GitHub anomaly monitoring."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select

from service.anomaly_detector import detector
from service.database import AnomalySummary, AsyncSessionLocal, init_db
from service.poller import poller
from service.queue import get_queue_stats
from service.sse_models import AnomalyMessage, ConnectedMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# SSE broadcaster for new summaries
class SSEBroadcaster:
    """Server-Sent Events broadcaster."""

    def __init__(self):
        self.clients: set[asyncio.Queue] = set()

    async def connect(self) -> asyncio.Queue:
        """Connect a new SSE client.

        Returns:
            Queue for receiving messages
        """
        queue: asyncio.Queue = asyncio.Queue()
        self.clients.add(queue)
        logger.info(f"SSE client connected (total: {len(self.clients)})")
        return queue

    async def disconnect(self, queue: asyncio.Queue) -> None:
        """Disconnect an SSE client.

        Args:
            queue: Client's message queue
        """
        self.clients.discard(queue)
        logger.info(f"SSE client disconnected (total: {len(self.clients)})")

    async def broadcast(self, message: dict) -> None:
        """Broadcast message to all connected clients.

        Args:
            message: Message dictionary to broadcast
        """
        dead_queues = set()
        for queue in self.clients:
            try:
                await queue.put(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                dead_queues.add(queue)

        # Clean up dead queues
        for queue in dead_queues:
            self.clients.discard(queue)


broadcaster = SSEBroadcaster()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown.

    Args:
        app: FastAPI application
    """
    # Startup
    logger.info("Starting application")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Start background poller
    poller_task = asyncio.create_task(poller.start())
    logger.info("Background poller started")

    # Start summary broadcaster (polls DB for new summaries)
    broadcaster_task = asyncio.create_task(_summary_broadcaster())
    logger.info("Summary broadcaster started")

    yield

    # Shutdown
    logger.info("Shutting down application")
    await poller.stop()
    poller_task.cancel()
    broadcaster_task.cancel()

    try:
        await poller_task
    except asyncio.CancelledError:
        pass

    try:
        await broadcaster_task
    except asyncio.CancelledError:
        pass

    logger.info("Application stopped")


# Create FastAPI app
app = FastAPI(
    title="GitHub Anomaly Monitor",
    description="Real-time GitHub event anomaly detection and alerting",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "GitHub Anomaly Monitor",
        "version": "1.0.0",
        "endpoints": {
            "summary": "/summary",
            "stream": "/stream",
            "health": "/health",
            "stats": "/stats",
        },
    }


@app.get("/summary")
async def get_summaries(
    since: str | None = Query(
        None, description="ISO timestamp to fetch summaries since"
    ),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of summaries"),
    severity: str | None = Query(None, description="Filter by severity"),
):
    """Get anomaly summaries from database.

    Args:
        since: ISO timestamp to fetch summaries created after
        limit: Maximum number of results
        severity: Filter by severity level

    Returns:
        List of anomaly summaries
    """
    async with AsyncSessionLocal() as session:
        query = select(AnomalySummary).order_by(desc(AnomalySummary.created_at))

        # Apply filters
        if since:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            query = query.where(AnomalySummary.created_at >= since_dt)

        if severity:
            query = query.where(AnomalySummary.severity == severity)

        query = query.limit(limit)

        result = await session.execute(query)
        summaries = result.scalars().all()

        return {
            "count": len(summaries),
            "summaries": [s.to_dict() for s in summaries],
        }


@app.get("/stream")
async def stream_summaries():
    """Server-Sent Events stream for real-time anomaly summaries.

    Returns:
        StreamingResponse with SSE stream
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events.

        Yields:
            SSE formatted messages
        """
        queue = await broadcaster.connect()

        try:
            # Send initial connection message (strongly typed)
            connected_msg = ConnectedMessage()
            yield f"data: {connected_msg.model_dump_json()}\n\n"

            # Stream messages
            while True:
                message = await queue.get()

                # Format as SSE
                event_data = json.dumps(message)
                yield f"data: {event_data}\n\n"

        except asyncio.CancelledError:
            logger.info("SSE stream cancelled")
        finally:
            await broadcaster.disconnect(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "poller": {
            "running": poller.running,
            "polls": poller.poll_count,
            "events_processed": poller.event_count,
            "anomalies_detected": poller.anomaly_count,
        },
        "detector": detector.get_stats(),
        "queue": get_queue_stats(),
        "sse_clients": len(broadcaster.clients),
    }


@app.get("/stats")
async def get_stats():
    """Get detailed statistics.

    Returns:
        Detailed service statistics
    """
    async with AsyncSessionLocal() as session:
        # Count total summaries
        result = await session.execute(select(AnomalySummary))
        total_summaries = len(result.scalars().all())

        # Count by severity
        severity_counts = {}
        for severity in ["low", "medium", "high", "critical"]:
            result = await session.execute(
                select(AnomalySummary).where(AnomalySummary.severity == severity)
            )
            severity_counts[severity] = len(result.scalars().all())

    return {
        "poller": poller.get_stats(),
        "detector": detector.get_stats(),
        "queue": get_queue_stats(),
        "summaries": {
            "total": total_summaries,
            "by_severity": severity_counts,
        },
        "sse": {
            "connected_clients": len(broadcaster.clients),
        },
    }


async def _summary_broadcaster():
    """Background task to broadcast new summaries via SSE."""
    last_id = 0

    while True:
        try:
            # Poll database for new summaries
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(AnomalySummary)
                    .where(AnomalySummary.id > last_id)
                    .order_by(AnomalySummary.id)
                )
                new_summaries = result.scalars().all()

                # Broadcast each new summary (strongly typed)
                for summary in new_summaries:
                    anomaly_msg = AnomalyMessage(data=summary.to_response())
                    message = anomaly_msg.model_dump(mode="json")
                    await broadcaster.broadcast(message)
                    last_id = summary.id
                    logger.info(f"Broadcasted summary {summary.id} to SSE clients")

            # Wait before next poll
            await asyncio.sleep(2)

        except Exception as e:
            logger.exception(f"Error in summary broadcaster: {e}")
            await asyncio.sleep(5)
