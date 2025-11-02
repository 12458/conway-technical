# GitHub Anomaly Monitoring Service

Frontend Repo: https://github.com/12458/conway-fe

Real-time anomaly detection and alerting system for GitHub public events using RRCF (Robust Random Cut Forest) and AI-powered summarization.

### What Was Built

A complete background polling service for GitHub Events API with:
- **RRCF-based streaming anomaly detection** (50 trees, 256 samples each)
- **GraphQL enrichment pipeline** for enhanced context
- **AI-powered summarization** using GPT 5 mini
- **Real-time SSE streaming** for instant UI updates
- **Redis Queue** for scalable async job processing

### Novel Approaches

1. **RRCF for GitHub Events**: First application of Robust Random Cut Forest to GitHub event streams with custom feature engineering optimized for tree-based anomaly detection
2. **Hybrid Detection Pipeline**: Combines statistical anomaly detection (RRCF) with rule-based pattern matching for higher accuracy
3. **GraphQL Enrichment**: Post-detection enrichment with actor profiles, repository context, workflow status, and commit verification for risk-based severity escalation
4. **Bot-Aware Processing**: Intelligent filtering of 60+ known bot accounts to reduce noise by ~40%
5. **Streaming Architecture**: True online anomaly detection with minimal memory footprint via lazy decay and LRU eviction

---

## Quick Start

### Prerequisites
- Python 3.12+
- Redis URL (Upstash provided in .env.example)
- AI API key (Anthropic or OpenAI)
- GitHub token (optional, for higher rate limits)

### Setup (5 minutes)

#### 1. Install Dependencies
```bash
uv sync
```

#### 2. Configure Environment
```bash
cp .env.example .env
```

#### 3. Test Setup
```bash
python test_service.py
```

You should see:
```
✅ AI API key configured
✅ Redis connection successful
✅ Database initialized successfully
✅ All tests passed!
```

#### 4. Run Service
```bash
python run_service.py
```

#### 5. Test Endpoints

Open a new terminal:

```bash
# Check health
curl http://localhost:8000/health

# Get stats
curl http://localhost:8000/stats

# Stream events (keep running)
curl http://localhost:8000/stream
```

### What Happens Next

1. **Poller starts** fetching GitHub events every 60 seconds
2. **Anomaly detector** analyzes each event using RRCF
3. **Anomalies** are saved to database and enqueued
4. **GraphQL enrichment** adds actor/repo/workflow context
5. **Workers** process jobs and generate AI summaries
6. **SSE stream** broadcasts summaries to connected clients

### RRCF Implementation
For anomalty detection, I used Robust Random Cut Forest Algorithm for anomaly detection by [Guha et al. (2016)](https://proceedings.mlr.press/v48/guha16.pdf).

My modified implementation of the [reference implementation](https://klabum.github.io/rrcf/) can be found [here](https://github.com/12458/rrcf).

---

## Architecture

```mermaid
graph TD
    A[GitHub Events API] -->|"Poll every 60s<br/>(with ETag caching)"| C["
        <strong>Background Poller</strong><br/>
        • Exponential backoff<br/>
        • Rate limit handling<br/>
        • Event deduplication
    "];

    C --> D["
        <strong>Anomaly Detector (RRCF)</strong><br/>
        • Feature extraction<br/>
        • 50 trees, size 256<br/>
        • CoDisp scoring<br/>
        • Rule-based patterns<br/>
        • Bot filtering
    "];

    D --> E["
        <strong>Database (PostgreSQL)</strong><br/>
        • GitHubEvent (all events)<br/>
        • AnomalySummary (anomalies)<br/>
        • Enrichment cache (4 tables)
    "];

    E -->|"Anomalies only"| F["
        <strong>Redis Queue (RQ)</strong><br/>
        • Async job processing<br/>
        • Scalable workers
    "];

    F --> G["
        <strong>GraphQL Enrichment Service</strong><br/>
        • Actor profiles<br/>
        • Repository context<br/>
        • Workflow/CI status<br/>
        • Commit verification<br/>
        • DB caching (TTL-based)
    "];

    B["GitHub GraphQL API<br/>(Enrichment Data)"] --> G;

    G --> H["
        <strong>AI Summarization Workers</strong><br/>
        • GPT 5 Mini<br/>
        • Enriched context<br/>
        • Risk-based severity<br/>
        • Structured summaries
    "];

    H --> I["
        <strong>FastAPI Server</strong><br/>
        • GET /summary (query)<br/>
        • GET /stream (SSE)<br/>
        • GET /health, /stats
    "];

    I -->|"SSE"| J["
        <strong>Frontend UI</strong><br/>
        • Real-time incident feed<br/>
        • Filterable cards
    "];

    %% Styling
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px,font-family:arial;
```

### Architecture Components

#### 1. Configuration System (`service/config.py`)
- Pydantic-based settings management
- Environment variable support with `SERVICE_` prefix
- Redis URL, GitHub token, AI API keys
- Configurable thresholds and parameters

#### 2. Database Layer (`service/database.py`)
- SQLAlchemy async ORM models
- **GitHubEvent**: Stores all processed events with anomaly scores
- **AnomalySummary**: Stores AI-generated incident reports
- **Enrichment cache tables**: Actor, repo, workflow, commit data
- Support for SQLite (dev) and PostgreSQL (prod)

#### 3. Message Queue (`service/queue.py`)
- RQ (Redis Queue) integration
- Enqueue anomaly summarization jobs
- Worker-based async processing
- Job statistics and monitoring

#### 4. Anomaly Detection (`service/anomaly_detector.py`)
- **RRCF-based streaming detector**
- Forest of 50 Robust Random Cut Trees (size 256)
- Custom feature engineering (299-dimensional vectors)
- CoDisp (collusive displacement) scoring
- Hybrid approach: Statistical + rule-based patterns
- Bot filtering (60+ known bots)

#### 5. Background Poller (`service/poller.py`)
- Polls GitHub Events API every 60 seconds
- **ETag caching** for efficient polling (304 Not Modified)
- **Exponential backoff** with jitter for rate limiting
- Handles 403/429/503 errors gracefully
- Deduplicates events using database
- Flow: Poll → Detect → Save → Enqueue

#### 6. GraphQL Enrichment (`service/enrichment.py`)
- Post-detection enrichment of anomalies
- Actor profiles, repo context, workflow status, commit verification
- SQLite caching with TTL (24h/1h/5min)
- Conservative rate limiting and exponential backoff
- Risk-based severity escalation

#### 7. AI Summarization (`service/summarizer.py`)
- RQ worker job: `summarize_anomaly_job()`
- Supports **Anthropic Claude** and **OpenAI GPT**
- Generates structured summaries with enriched context
- **Fallback summaries** when AI unavailable
- Saves to database and broadcasts via SSE

#### 8. FastAPI Application (`service/app.py`)
- **GET /summary**: Query summaries (filtering by time, severity, limit)
- **GET /stream**: Server-Sent Events for real-time updates
- **GET /health**: Service health and statistics
- **GET /stats**: Detailed component stats
- CORS enabled for frontend integration
- Background tasks: poller + summary broadcaster

---

## Features

### RRCF-Based Streaming Anomaly Detection

- Robust Random Cut Forest with 50 trees (size 256 each)
- Proper feature encoding for tree-based models:
  - One-hot hashing for categorical features (event type, actor, repo, org)
  - Character n-gram hashing for text (commits, titles, bodies)
  - Exponential decay for temporal patterns
  - Z-score normalization using Welford's algorithm
- CoDisp (collusive displacement) scoring
- Configurable threshold (default: 40.0)

### Hybrid Detection Pipeline

- RRCF statistical anomaly detection
- Rule-based pattern matching (force pushes, permission changes, etc.)
- Combined scoring for high accuracy
- Bot filtering (removes 60+ known bot accounts)

### GraphQL Enrichment Pipeline

Post-detection enrichment of anomalies with GitHub GraphQL API for enhanced context:

#### Enrichment Data Sources

**Actor Profile Enrichment:**
- Account creation date and age
- Total contributions (commits, PRs, issues)
- Follower/following counts
- Organization memberships
- Site admin status
- Company affiliation

**Repository Context Enrichment:**
- Popularity metrics (stars, forks, watchers)
- Branch protection status
- Required code review settings
- Vulnerability alerts enabled
- Security policy presence
- Primary language and topics

**Workflow/CI Status Enrichment:**
- Check suite conclusions
- Failed/successful job counts
- Individual check run details
- Overall CI/CD health

**Commit Verification Enrichment:**
- GPG signature validation
- Signer identity
- Lines added/deleted
- Files changed
- Author information

#### Caching Strategy

Enrichment data is cached in SQLite with TTL-based expiration:

| Data Type | TTL | Rationale |
|-----------|-----|-----------|
| Actor Profiles | 24 hours | User metadata changes infrequently |
| Repository Context | 1 hour | Repo settings may change during incidents |
| Workflow Status | 5 minutes | CI status updates frequently |
| Commit Verification | Permanent | Commits are immutable |

Cache hit rates typically exceed 70% after warm-up, significantly reducing GraphQL API calls.

#### Risk-Based Severity

Enrichment data influences final severity classification:

- **Critical**: High anomaly score + critical repo (>5000 stars) OR site admin action
- **High**: New account (<30 days) + unsigned commits on protected branch
- **Medium**: Moderate anomaly score + failed workflows
- **Low**: Low anomaly score + no concerning enrichment factors

#### Configuration

Enable GraphQL enrichment by setting environment variables:

```bash
# Required: GitHub token with appropriate scopes
SERVICE_GITHUB_GRAPHQL_TOKEN=ghp_xxxxxxxxxxxx  # Requires: repo, read:org, read:user

# Optional: Feature flag
SERVICE_ENRICHMENT_ENABLED=true  # Default: true

# Optional: Performance tuning
SERVICE_ENRICHMENT_BATCH_SIZE=10         # Default: 10
SERVICE_ENRICHMENT_TIMEOUT_MS=5000       # Default: 5000
```

### Real-time Processing

- Streaming architecture with minimal latency
- ETag caching for efficient polling
- Background job queue for AI processing
- Server-Sent Events for instant UI updates

### AI-Powered Summarization

Generates structured summaries with:
- Title (200 chars max)
- Severity (low/medium/high/critical)
- Root Cause (3-5 bullets)
- Impact (3-5 bullets)
- Next Steps (3-5 bullets)
- Tags for categorization

---

## API Reference

### GET /summary

Query anomaly summaries from database.

**Query Parameters:**
- `since`: ISO timestamp (e.g., `2024-01-01T00:00:00Z`)
- `limit`: Max results (1-500, default: 50)
- `severity`: Filter by `low`, `medium`, `high`, `critical`

**Example:**
```bash
curl "http://localhost:8000/summary?limit=10&severity=high"
```

**Response:**
```json
{
  "count": 2,
  "summaries": [
    {
      "id": 1,
      "event_id": "12345",
      "title": "Force push to main branch on critical-repo",
      "severity": "high",
      "root_cause": [
        "Actor john-doe performed force push to main branch",
        "Anomaly score: 85.3 (threshold: 40.0)",
        "Pattern: Force push detected"
      ],
      "impact": [
        "Potential code history rewrite",
        "May affect other contributors",
        "Could indicate compromised account"
      ],
      "next_steps": [
        "Investigate john-doe's recent activity",
        "Review branch protection rules",
        "Contact repository maintainers"
      ],
      "suspicious_patterns": ["Force push detected"],
      "anomaly_score": 85.3,
      "event_type": "PushEvent",
      "actor_login": "john-doe",
      "repo_name": "org/critical-repo",
      "event_timestamp": "2024-01-15T10:30:00Z",
      "created_at": "2024-01-15T10:30:15Z",
      "tags": ["pushevent", "high", "code-change"]
    }
  ]
}
```

### GET /stream

Server-Sent Events stream for real-time anomaly notifications.

**Example:**
```bash
curl http://localhost:8000/stream
```

**Connection Flow:**
1. Client connects → receives `connected` message immediately
2. New anomalies → receives `anomaly` messages as they're detected
3. Connection persists indefinitely (until client disconnects)

**Message Types:**

1. **Connected Message** - Initial connection acknowledgment
```json
{
  "type": "connected",
  "message": "Stream connected"
}
```

2. **Anomaly Message** - Real-time anomaly notification
```json
{
  "type": "anomaly",
  "data": {
    "id": 42,
    "title": "Suspicious Force Push to Main Branch by New Account",
    "severity": "high",
    "root_cause": [...],
    "impact": [...],
    "next_steps": [...],
    "anomaly_score": 12.3,
    "event_type": "PushEvent",
    "actor_login": "suspicious-user-123",
    "repo_name": "acme/critical-infrastructure",
    "raw_event": {...},
    "tags": ["security", "malicious-activity", "risk:high"]
  }
}
```

**Frontend Integration:**
```javascript
const eventSource = new EventSource('http://localhost:8000/stream');

eventSource.onmessage = (event) => {
  const message = JSON.parse(event.data);

  if (message.type === 'connected') {
    console.log('Connected to anomaly stream');
  } else if (message.type === 'anomaly') {
    displayIncident(message.data);
  }
};
```

### GET /health

Service health and component status.

**Example:**
```bash
curl http://localhost:8000/health
```

Returns: Poller status, detector statistics, queue statistics, SSE client count

### GET /stats

Detailed statistics for all components.

**Example:**
```bash
curl http://localhost:8000/stats
```

---

## GitHub Client Library

The project includes a robust, type-safe Python client for accessing GitHub's public events REST API.

### Features

- **Async/Await Support**: Built on `httpx` for high-performance async HTTP requests
- **Type Safety**: Full Pydantic v2 models for request/response validation
- **ETag Caching**: Implements GitHub's ETag caching mechanism to optimize polling
- **Poll Interval Awareness**: Automatically parses and respects `X-Poll-Interval` headers
- **Comprehensive Error Handling**: Custom exceptions for different API error scenarios
- **OOP Design**: Clean class-based architecture with context manager support
- **Pagination Support**: Easy pagination through event results

### Basic Usage

```python
import asyncio
from github_client import GitHubEventsClient

async def main():
    async with GitHubEventsClient() as client:
        response = await client.list_public_events(per_page=10)

        print(f"Fetched {len(response.events)} events")

        for event in response.events:
            print(f"{event.type} by {event.actor.login} on {event.repo.name}")

asyncio.run(main())
```

### ETag Caching Example

```python
async with GitHubEventsClient() as client:
    # First request
    response = await client.list_public_events(per_page=5)
    etag = response.etag

    # Second request with ETag
    response2 = await client.list_public_events(etag=etag, per_page=5)

    if not response2.events:
        print("No new events (304 Not Modified)")
        # Your rate limit was not affected!
```

### GH Archive Support

The client supports loading and analyzing historical GitHub event data from [GH Archive](https://www.gharchive.org/).

```python
from github_client import GHArchiveLoader

# Load events from a local archive file
loader = GHArchiveLoader("data/2015-01-01-15.json.gz")

# Iterate through events (memory-efficient)
for event in loader.iter_events():
    print(f"{event.type} by {event.actor.login} on {event.repo.name}")

# Get processing statistics
stats = loader.get_stats()
print(f"Processed {stats.valid_events} valid events out of {stats.total_events} total")
```

### Error Handling

```python
from github_client import (
    GitHubAPIError,
    ForbiddenError,
    ServiceUnavailableError,
    RateLimitError,
)

async with GitHubEventsClient() as client:
    try:
        response = await client.list_public_events()
    except ForbiddenError:
        print("Rate limit exceeded")
    except ServiceUnavailableError:
        print("GitHub API is down")
    except GitHubAPIError as e:
        print(f"API error: {e.message} (status: {e.status_code})")
```

---

## Configuration

All settings can be configured via environment variables with the `SERVICE_` prefix:

### GitHub API
- `SERVICE_GITHUB_TOKEN`: Personal access token (optional, increases rate limit from 60 to 5000 req/hr)
- `SERVICE_GITHUB_GRAPHQL_TOKEN`: Token for GraphQL enrichment (requires: repo, read:org, read:user)

### Redis
- `SERVICE_REDIS_URL`: Redis connection URL (default: `redis://localhost:6379`)

### Database
- `SERVICE_DATABASE_URL`: SQLAlchemy connection string
  - SQLite: `sqlite+aiosqlite:///./github_events.db`
  - PostgreSQL: `postgresql+asyncpg://user:pass@host/db`

### Polling
- `SERVICE_POLLING_INTERVAL`: Default poll interval in seconds (default: 60)
- `SERVICE_MAX_BACKOFF_SECONDS`: Max backoff for rate limiting (default: 300)
- `SERVICE_MAX_EVENTS_PER_FETCH`: Events per API call (default: 100)

### Anomaly Detection
- `SERVICE_ANOMALY_THRESHOLD`: CoDisp threshold (default: 40.0)
- `SERVICE_TREE_SIZE`: RRCF tree size (default: 256)
- `SERVICE_NUM_TREES`: Number of trees in forest (default: 50)
- `SERVICE_ENABLE_BOT_FILTERING`: Filter known bots (default: true)

### GraphQL Enrichment
- `SERVICE_ENRICHMENT_ENABLED`: Enable enrichment (default: true)
- `SERVICE_ENRICHMENT_BATCH_SIZE`: Batch size for queries (default: 10)
- `SERVICE_ENRICHMENT_TIMEOUT_MS`: Query timeout (default: 5000)

### AI Summarization
- `SERVICE_AI_PROVIDER`: `anthropic` or `openai` (default: anthropic)
- `SERVICE_AI_MODEL`: Model name (default: claude-3-5-sonnet-20241022)
- `SERVICE_ANTHROPIC_API_KEY`: Anthropic API key
- `SERVICE_OPENAI_API_KEY`: OpenAI API key

### Workers
- `SERVICE_WORKER_COUNT`: Number of RQ workers (default: 2)
- `SERVICE_QUEUE_NAME`: RQ queue name (default: github-anomalies)

---

## Database Schema

### GitHubEvent
Stores all processed GitHub events:
- `id`: GitHub event ID (primary key)
- `event_type`: Event type (PushEvent, IssuesEvent, etc.)
- `actor_login`, `actor_id`: Actor information
- `repo_name`, `repo_id`: Repository information
- `org_login`: Organization (if applicable)
- `created_at`: Event timestamp
- `payload`: Full event payload (JSON)
- `processed`: Processing flag
- `is_anomaly`: Anomaly detection result
- `anomaly_score`: CoDisp score

### AnomalySummary
Stores AI-generated summaries for anomalies:
- `id`: Auto-increment primary key
- `event_id`: Reference to GitHubEvent
- `title`: Short incident description
- `severity`: low, medium, high, critical
- `root_cause`: Array of 3-5 bullet points
- `impact`: Array of 3-5 bullet points
- `next_steps`: Array of 3-5 bullet points
- `suspicious_patterns`: Detected patterns
- `anomaly_score`: CoDisp score
- `event_type`, `actor_login`, `repo_name`: Event context
- `raw_event`: Full event data (JSON)
- `event_timestamp`, `created_at`: Timestamps
- `tags`: Array of tags

### Enrichment Cache Tables
- `actor_profile_cache`: Actor profile data (TTL: 24h)
- `repository_context_cache`: Repository context (TTL: 1h)
- `workflow_status_cache`: Workflow/CI status (TTL: 5min)
- `commit_verification_cache`: Commit verification (permanent)

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests (fast, mocked)
pytest -m unit

# Run only integration tests (slow, real API)
pytest -m integration

# Skip integration tests
pytest -m "not integration"

# Run with coverage report
pytest --cov=github_client --cov=service --cov-report=html

# Run tests verbosely
pytest -v
```

### Test Organization

**Unit Tests** (`github_client/tests/unit/`, `tests/unit/`):
- `test_models.py`: Tests for Pydantic models
- `test_config.py`: Tests for configuration settings
- `test_client.py`: Tests for client with mocked HTTP
- `test_sse_models.py`: Tests for SSE message models
- `test_multiforest.py`: Tests for multi-forest detector
- `test_velocity_fix.py`: Tests for velocity tracking

**Integration Tests** (`github_client/tests/integration/`, `tests/integration/`):
- `test_github_api.py`: Tests hitting real GitHub API
- `test_service.py`: Service integration tests
- `test_event_parsing.py`: Event parsing tests

---

## Deployment

### Tuning

#### Lower threshold for more anomalies
```bash
# In .env
SERVICE_ANOMALY_THRESHOLD=20.0
```

#### Increase workers for faster processing
```bash
python run_service.py --workers 4
```

#### Custom port
```bash
python run_service.py --port 8080
```

### Running the Service

```bash
# Run everything (API server + workers)
python run_service.py

# Run with custom worker count
python run_service.py --workers 4

# Run on custom host/port
python run_service.py --host 0.0.0.0 --port 8080

# Run only workers (no API)
python run_service.py --workers-only

# Run only API (no workers)
python run_service.py --api-only
```

---

## Implementation Details

### File Structure

```
conway/
├── service/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── database.py            # SQLAlchemy models
│   ├── queue.py               # RQ integration
│   ├── anomaly_detector.py    # RRCF detector
│   ├── poller.py              # Background polling
│   ├── enrichment.py          # GraphQL enrichment
│   ├── summarizer.py          # AI worker
│   ├── app.py                 # FastAPI endpoints
│   ├── worker.py              # RQ worker entry
│   └── sse_models.py          # SSE message models
├── github_client/             # GitHub client library
│   ├── client.py              # GitHubEventsClient
│   ├── models.py              # Pydantic models
│   ├── feature_extractor.py   # Feature engineering
│   ├── gharchive.py           # GH Archive loader
│   └── tests/                 # Client tests
├── tests/                     # Service tests
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── run_service.py             # Main runner
├── test_service.py            # Setup verification
├── .env.example               # Config template
├── pyproject.toml             # Dependencies
└── README.md                  # This file
```

### Key Features Implemented

- Background polling with ETag caching
- Exponential backoff (60s → 300s max)
- RRCF streaming algorithm
- Proper feature encoding (one-hot hashing, n-grams, exponential decay)
- 299-dimensional feature vectors
- Bot filtering reduces noise
- GraphQL enrichment pipeline
- Database persistence with deduplication
- Redis-backed RQ for job processing
- RESTful query endpoint
- SSE streaming for live updates
- Health checks and metrics
- AI summarization (Claude/GPT)
- Structured JSON output
- Fallback to rule-based summaries
- Environment-based configuration
- Graceful shutdown
- Error handling and retries
- Multiprocessing support

---

## License

This repo is licensed under AGPL-3.0.
