"""Comprehensive unit tests for enrichment service orchestrator.

Tests include:
- Enrichment orchestration (enabled/disabled)
- Cache-first strategy for all data types
- Event-specific enrichment (PushEvent, PullRequestEvent)
- Error handling and graceful degradation
- Statistics tracking
- Cache cleanup operations
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from github_client.enriched_models import (
    ActorProfile,
    CommitVerification,
    EnrichedEvent,
    RepositoryContext,
    WorkflowStatus,
)
from github_client.models import Actor, Event, Repo
from service.enrichment_service import EnrichmentService


@pytest.fixture
def mock_graphql_client():
    """Mock GitHubGraphQLClient for testing."""
    client = AsyncMock()
    client.rate_limit_info = {"remaining": 5000, "limit": 5000}
    return client


@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    return AsyncMock()


@pytest.fixture
def mock_cache_manager():
    """Mock EnrichmentCacheManager for testing."""
    manager = AsyncMock()
    manager.cache_stats = {
        "hits": 0,
        "misses": 0,
        "total_requests": 0,
        "hit_rate": 0.0,
    }
    return manager


@pytest.fixture
def sample_event():
    """Sample GitHub event for testing."""
    return Event(
        id="12345",
        type="PushEvent",
        actor=Actor(
            id=123,
            login="testuser",
            display_login="testuser",
            gravatar_id="",
            url="https://api.github.com/users/testuser",
            avatar_url="https://avatars.githubusercontent.com/u/123",
        ),
        repo=Repo(
            id=456,
            name="testowner/testrepo",
            url="https://api.github.com/repos/testowner/testrepo",
        ),
        payload={
            "ref": "refs/heads/main",
            "head": "abc123def456",
            "before": "xyz789",
        },
        public=True,
        created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_actor_profile():
    """Sample ActorProfile for testing."""
    return ActorProfile(
        login="testuser",
        account_created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        account_age_days=380,
        follower_count=100,
        following_count=50,
        repository_count=25,
        total_commit_contributions=500,
        total_pr_contributions=150,
        total_issue_contributions=75,
        organizations=["org1"],
        is_site_admin=False,
        cached_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_repository_context():
    """Sample RepositoryContext for testing."""
    return RepositoryContext(
        owner="testowner",
        name="testrepo",
        full_name="testowner/testrepo",
        stargazer_count=1500,
        fork_count=200,
        watcher_count=100,
        primary_language="Python",
        default_branch="main",
        has_security_policy=True,
        is_fork=False,
        is_archived=False,
        topics=["testing"],
        license_name="MIT",
        cached_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_workflow_status():
    """Sample WorkflowStatus for testing."""
    return WorkflowStatus(
        repository="testowner/testrepo",
        commit_sha="abc123def456",
        total_check_suites=2,
        successful_suites=2,
        failed_suites=0,
        pending_suites=0,
        check_runs=[{"name": "test", "conclusion": "SUCCESS"}],
        overall_conclusion="SUCCESS",
        cached_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_commit_verification():
    """Sample CommitVerification for testing."""
    return CommitVerification(
        repository="testowner/testrepo",
        sha="abc123def456",
        is_signed=True,
        signer_login="testuser",
        signature_valid=True,
        additions=100,
        deletions=50,
        changed_files=5,
        message="Test commit",
        author_name="Test User",
        author_email="test@example.com",
        commit_entropy=5.5,
        commit_size=150,
        cached_at=datetime.utcnow(),
    )


@pytest.mark.unit
class TestEnrichmentServiceInitialization:
    """Tests for EnrichmentService initialization."""

    def test_initialization_enabled(self, mock_graphql_client, mock_db_session):
        """Test service initialization with enrichment enabled."""
        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        assert service.graphql_client == mock_graphql_client
        assert service.enabled is True
        assert service._enrichment_count == 0
        assert service._error_count == 0

    def test_initialization_disabled(self, mock_graphql_client, mock_db_session):
        """Test service initialization with enrichment disabled."""
        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=False,
        )

        assert service.enabled is False


@pytest.mark.unit
class TestEnrichAnomaly:
    """Tests for enrich_anomaly method."""

    @pytest.mark.asyncio
    async def test_enrich_anomaly_disabled(
        self, mock_graphql_client, mock_db_session, sample_event
    ):
        """Test enrichment when disabled returns base event."""
        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=False,
        )

        enriched = await service.enrich_anomaly(
            event=sample_event,
            anomaly_score=75.0,
            suspicious_patterns=["high_velocity"],
        )

        assert isinstance(enriched, EnrichedEvent)
        assert enriched.event == sample_event
        assert enriched.anomaly_score == 75.0
        assert enriched.suspicious_patterns == ["high_velocity"]
        assert enriched.actor_profile is None
        assert enriched.repository_context is None

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_anomaly_full_success(
        self,
        mock_cache_class,
        mock_graphql_client,
        mock_db_session,
        sample_event,
        sample_actor_profile,
        sample_repository_context,
        sample_workflow_status,
        sample_commit_verification,
    ):
        """Test successful enrichment with all data sources."""
        # Setup cache manager mock
        mock_cache = AsyncMock()
        mock_cache.get_actor_profile.return_value = None  # Cache miss
        mock_cache.get_repository_context.return_value = None  # Cache miss
        mock_cache.get_workflow_status.return_value = None  # Cache miss
        mock_cache.get_commit_verification.return_value = None  # Cache miss
        mock_cache.cache_stats = {
            "hits": 0,
            "misses": 4,
            "total_requests": 4,
            "hit_rate": 0.0,
        }
        mock_cache_class.return_value = mock_cache

        # Setup GraphQL client responses
        mock_graphql_client.get_actor_profile.return_value = sample_actor_profile
        mock_graphql_client.get_repository_context.return_value = sample_repository_context
        mock_graphql_client.get_workflow_status.return_value = sample_workflow_status
        mock_graphql_client.get_commit_verification.return_value = sample_commit_verification

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        enriched = await service.enrich_anomaly(
            event=sample_event,
            anomaly_score=80.0,
            suspicious_patterns=["new_account", "high_velocity"],
        )

        assert enriched.event == sample_event
        assert enriched.anomaly_score == 80.0
        assert enriched.suspicious_patterns == ["new_account", "high_velocity"]
        assert enriched.actor_profile == sample_actor_profile
        assert enriched.repository_context == sample_repository_context
        assert enriched.workflow_status == sample_workflow_status
        assert enriched.commit_verification == sample_commit_verification
        assert service._enrichment_count == 1
        assert service._error_count == 0

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_anomaly_actor_profile_fails(
        self, mock_cache_class, mock_graphql_client, mock_db_session, sample_event
    ):
        """Test enrichment continues gracefully when actor profile fails."""
        # Setup cache manager mock
        mock_cache = AsyncMock()
        mock_cache.get_actor_profile.side_effect = Exception("API Error")
        mock_cache.get_repository_context.return_value = None
        mock_cache.cache_stats = {"hits": 0, "misses": 0, "total_requests": 0, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        enriched = await service.enrich_anomaly(
            event=sample_event, anomaly_score=75.0, suspicious_patterns=[]
        )

        assert enriched.actor_profile is None  # Failed, but event still enriched
        assert service._error_count == 1
        assert service._enrichment_count == 1

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_anomaly_repository_context_fails(
        self, mock_cache_class, mock_graphql_client, mock_db_session, sample_event
    ):
        """Test enrichment continues gracefully when repository context fails."""
        # Setup cache manager mock
        mock_cache = AsyncMock()
        mock_cache.get_actor_profile.return_value = None
        mock_cache.get_repository_context.side_effect = Exception("API Error")
        mock_cache.cache_stats = {"hits": 0, "misses": 0, "total_requests": 0, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        mock_graphql_client.get_actor_profile.return_value = None

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        enriched = await service.enrich_anomaly(
            event=sample_event, anomaly_score=75.0, suspicious_patterns=[]
        )

        assert enriched.repository_context is None
        assert service._error_count == 1


@pytest.mark.unit
class TestActorProfileEnrichment:
    """Tests for _enrich_actor_profile method."""

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_actor_profile_cache_hit(
        self,
        mock_cache_class,
        mock_graphql_client,
        mock_db_session,
        sample_actor_profile,
    ):
        """Test actor profile enrichment with cache hit."""
        # Setup cache manager mock
        mock_cache = AsyncMock()
        mock_cache.get_actor_profile.return_value = sample_actor_profile
        mock_cache.cache_stats = {"hits": 1, "misses": 0, "total_requests": 1, "hit_rate": 1.0}
        mock_cache_class.return_value = mock_cache

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        profile = await service._enrich_actor_profile("testuser")

        assert profile == sample_actor_profile
        mock_cache.get_actor_profile.assert_called_once_with("testuser")
        mock_graphql_client.get_actor_profile.assert_not_called()  # Should not fetch from API

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_actor_profile_cache_miss(
        self,
        mock_cache_class,
        mock_graphql_client,
        mock_db_session,
        sample_actor_profile,
    ):
        """Test actor profile enrichment with cache miss."""
        # Setup cache manager mock
        mock_cache = AsyncMock()
        mock_cache.get_actor_profile.return_value = None  # Cache miss
        mock_cache.cache_stats = {"hits": 0, "misses": 1, "total_requests": 1, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        mock_graphql_client.get_actor_profile.return_value = sample_actor_profile

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        profile = await service._enrich_actor_profile("testuser")

        assert profile == sample_actor_profile
        mock_cache.get_actor_profile.assert_called_once_with("testuser")
        mock_graphql_client.get_actor_profile.assert_called_once_with("testuser")
        mock_cache.set_actor_profile.assert_called_once_with(sample_actor_profile)

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_actor_profile_returns_none(
        self, mock_cache_class, mock_graphql_client, mock_db_session
    ):
        """Test actor profile enrichment when API returns None."""
        # Setup cache manager mock
        mock_cache = AsyncMock()
        mock_cache.get_actor_profile.return_value = None
        mock_cache.cache_stats = {"hits": 0, "misses": 1, "total_requests": 1, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        mock_graphql_client.get_actor_profile.return_value = None

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        profile = await service._enrich_actor_profile("nonexistent")

        assert profile is None
        mock_cache.set_actor_profile.assert_not_called()  # Don't cache None


@pytest.mark.unit
class TestRepositoryContextEnrichment:
    """Tests for _enrich_repository_context method."""

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_repository_context_cache_hit(
        self,
        mock_cache_class,
        mock_graphql_client,
        mock_db_session,
        sample_repository_context,
    ):
        """Test repository context enrichment with cache hit."""
        # Setup cache manager mock
        mock_cache = AsyncMock()
        mock_cache.get_repository_context.return_value = sample_repository_context
        mock_cache.cache_stats = {"hits": 1, "misses": 0, "total_requests": 1, "hit_rate": 1.0}
        mock_cache_class.return_value = mock_cache

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        context = await service._enrich_repository_context("testowner", "testrepo")

        assert context == sample_repository_context
        mock_cache.get_repository_context.assert_called_once_with("testowner", "testrepo")
        mock_graphql_client.get_repository_context.assert_not_called()

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_repository_context_cache_miss(
        self,
        mock_cache_class,
        mock_graphql_client,
        mock_db_session,
        sample_repository_context,
    ):
        """Test repository context enrichment with cache miss."""
        # Setup cache manager mock
        mock_cache = AsyncMock()
        mock_cache.get_repository_context.return_value = None
        mock_cache.cache_stats = {"hits": 0, "misses": 1, "total_requests": 1, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        mock_graphql_client.get_repository_context.return_value = sample_repository_context

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        context = await service._enrich_repository_context("testowner", "testrepo")

        assert context == sample_repository_context
        mock_graphql_client.get_repository_context.assert_called_once_with(
            "testowner", "testrepo"
        )
        mock_cache.set_repository_context.assert_called_once_with(sample_repository_context)


@pytest.mark.unit
class TestPushEventEnrichment:
    """Tests for _enrich_push_event method."""

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_push_event_success(
        self,
        mock_cache_class,
        mock_graphql_client,
        mock_db_session,
        sample_event,
        sample_workflow_status,
        sample_commit_verification,
    ):
        """Test PushEvent enrichment with valid head SHA."""
        # Setup cache manager mock
        mock_cache = AsyncMock()
        mock_cache.get_workflow_status.return_value = None
        mock_cache.get_commit_verification.return_value = None
        mock_cache.cache_stats = {"hits": 0, "misses": 2, "total_requests": 2, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        mock_graphql_client.get_workflow_status.return_value = sample_workflow_status
        mock_graphql_client.get_commit_verification.return_value = sample_commit_verification

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        enriched = EnrichedEvent(
            event=sample_event, anomaly_score=75.0, suspicious_patterns=[]
        )

        await service._enrich_push_event(sample_event, enriched)

        assert enriched.workflow_status == sample_workflow_status
        assert enriched.commit_verification == sample_commit_verification

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_push_event_missing_head_sha(
        self, mock_cache_class, mock_graphql_client, mock_db_session
    ):
        """Test PushEvent enrichment with missing head SHA."""
        mock_cache = AsyncMock()
        mock_cache.cache_stats = {"hits": 0, "misses": 0, "total_requests": 0, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        event = Event(
            id="12345",
            type="PushEvent",
            actor=Actor(
                id=123,
                login="testuser",
                display_login="testuser",
                gravatar_id="",
                url="https://api.github.com/users/testuser",
                avatar_url="https://avatars.githubusercontent.com/u/123",
            ),
            repo=Repo(
                id=456,
                name="testowner/testrepo",
                url="https://api.github.com/repos/testowner/testrepo",
            ),
            payload={"ref": "refs/heads/main"},  # No head SHA
            public=True,
            created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        enriched = EnrichedEvent(event=event, anomaly_score=75.0, suspicious_patterns=[])

        await service._enrich_push_event(event, enriched)

        assert enriched.workflow_status is None
        assert enriched.commit_verification is None
        mock_graphql_client.get_workflow_status.assert_not_called()

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_push_event_workflow_status_fails(
        self,
        mock_cache_class,
        mock_graphql_client,
        mock_db_session,
        sample_event,
        sample_commit_verification,
    ):
        """Test PushEvent enrichment when workflow status fails."""
        mock_cache = AsyncMock()
        mock_cache.get_workflow_status.side_effect = Exception("API Error")
        mock_cache.get_commit_verification.return_value = None
        mock_cache.cache_stats = {"hits": 0, "misses": 1, "total_requests": 1, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        mock_graphql_client.get_commit_verification.return_value = sample_commit_verification

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        enriched = EnrichedEvent(
            event=sample_event, anomaly_score=75.0, suspicious_patterns=[]
        )

        await service._enrich_push_event(sample_event, enriched)

        assert enriched.workflow_status is None  # Failed
        assert enriched.commit_verification == sample_commit_verification  # Succeeded


@pytest.mark.unit
class TestPullRequestEventEnrichment:
    """Tests for _enrich_pull_request_event method."""

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_pull_request_event_success(
        self,
        mock_cache_class,
        mock_graphql_client,
        mock_db_session,
        sample_workflow_status,
    ):
        """Test PullRequestEvent enrichment with valid head SHA."""
        mock_cache = AsyncMock()
        mock_cache.get_workflow_status.return_value = None
        mock_cache.cache_stats = {"hits": 0, "misses": 1, "total_requests": 1, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        mock_graphql_client.get_workflow_status.return_value = sample_workflow_status

        event = Event(
            id="12345",
            type="PullRequestEvent",
            actor=Actor(
                id=123,
                login="testuser",
                display_login="testuser",
                gravatar_id="",
                url="https://api.github.com/users/testuser",
                avatar_url="https://avatars.githubusercontent.com/u/123",
            ),
            repo=Repo(
                id=456,
                name="testowner/testrepo",
                url="https://api.github.com/repos/testowner/testrepo",
            ),
            payload={
                "action": "opened",
                "pull_request": {"head": {"sha": "abc123def456"}},
            },
            public=True,
            created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        enriched = EnrichedEvent(event=event, anomaly_score=75.0, suspicious_patterns=[])

        await service._enrich_pull_request_event(event, enriched)

        assert enriched.workflow_status == sample_workflow_status

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_pull_request_event_missing_head_sha(
        self, mock_cache_class, mock_graphql_client, mock_db_session
    ):
        """Test PullRequestEvent enrichment with missing head SHA."""
        mock_cache = AsyncMock()
        mock_cache.cache_stats = {"hits": 0, "misses": 0, "total_requests": 0, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        event = Event(
            id="12345",
            type="PullRequestEvent",
            actor=Actor(
                id=123,
                login="testuser",
                display_login="testuser",
                gravatar_id="",
                url="https://api.github.com/users/testuser",
                avatar_url="https://avatars.githubusercontent.com/u/123",
            ),
            repo=Repo(
                id=456,
                name="testowner/testrepo",
                url="https://api.github.com/repos/testowner/testrepo",
            ),
            payload={"action": "opened", "pull_request": {}},  # No head SHA
            public=True,
            created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        enriched = EnrichedEvent(event=event, anomaly_score=75.0, suspicious_patterns=[])

        await service._enrich_pull_request_event(event, enriched)

        assert enriched.workflow_status is None
        mock_graphql_client.get_workflow_status.assert_not_called()


@pytest.mark.unit
class TestWorkflowStatusEnrichment:
    """Tests for _enrich_workflow_status method."""

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_workflow_status_cache_hit(
        self,
        mock_cache_class,
        mock_graphql_client,
        mock_db_session,
        sample_workflow_status,
    ):
        """Test workflow status enrichment with cache hit."""
        mock_cache = AsyncMock()
        mock_cache.get_workflow_status.return_value = sample_workflow_status
        mock_cache.cache_stats = {"hits": 1, "misses": 0, "total_requests": 1, "hit_rate": 1.0}
        mock_cache_class.return_value = mock_cache

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        status = await service._enrich_workflow_status("testowner", "testrepo", "abc123")

        assert status == sample_workflow_status
        mock_graphql_client.get_workflow_status.assert_not_called()

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_workflow_status_cache_miss(
        self,
        mock_cache_class,
        mock_graphql_client,
        mock_db_session,
        sample_workflow_status,
    ):
        """Test workflow status enrichment with cache miss."""
        mock_cache = AsyncMock()
        mock_cache.get_workflow_status.return_value = None
        mock_cache.cache_stats = {"hits": 0, "misses": 1, "total_requests": 1, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        mock_graphql_client.get_workflow_status.return_value = sample_workflow_status

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        status = await service._enrich_workflow_status("testowner", "testrepo", "abc123")

        assert status == sample_workflow_status
        mock_graphql_client.get_workflow_status.assert_called_once_with(
            "testowner", "testrepo", "abc123"
        )
        mock_cache.set_workflow_status.assert_called_once_with(sample_workflow_status)


@pytest.mark.unit
class TestCommitVerificationEnrichment:
    """Tests for _enrich_commit_verification method."""

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_commit_verification_cache_hit(
        self,
        mock_cache_class,
        mock_graphql_client,
        mock_db_session,
        sample_commit_verification,
    ):
        """Test commit verification enrichment with cache hit."""
        mock_cache = AsyncMock()
        mock_cache.get_commit_verification.return_value = sample_commit_verification
        mock_cache.cache_stats = {"hits": 1, "misses": 0, "total_requests": 1, "hit_rate": 1.0}
        mock_cache_class.return_value = mock_cache

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        verification = await service._enrich_commit_verification(
            "testowner", "testrepo", "abc123"
        )

        assert verification == sample_commit_verification
        mock_graphql_client.get_commit_verification.assert_not_called()

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_enrich_commit_verification_cache_miss(
        self,
        mock_cache_class,
        mock_graphql_client,
        mock_db_session,
        sample_commit_verification,
    ):
        """Test commit verification enrichment with cache miss."""
        mock_cache = AsyncMock()
        mock_cache.get_commit_verification.return_value = None
        mock_cache.cache_stats = {"hits": 0, "misses": 1, "total_requests": 1, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        mock_graphql_client.get_commit_verification.return_value = sample_commit_verification

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        verification = await service._enrich_commit_verification(
            "testowner", "testrepo", "abc123"
        )

        assert verification == sample_commit_verification
        mock_graphql_client.get_commit_verification.assert_called_once_with(
            "testowner", "testrepo", "abc123"
        )
        mock_cache.set_commit_verification.assert_called_once_with(sample_commit_verification)


@pytest.mark.unit
class TestCacheCleanupAndStats:
    """Tests for cache cleanup and statistics."""

    @pytest.mark.asyncio
    @patch("service.enrichment_service.EnrichmentCacheManager")
    async def test_cleanup_cache(self, mock_cache_class, mock_graphql_client, mock_db_session):
        """Test cache cleanup operation."""
        mock_cache = AsyncMock()
        mock_cache.cleanup_expired.return_value = 42
        mock_cache.cache_stats = {"hits": 0, "misses": 0, "total_requests": 0, "hit_rate": 0.0}
        mock_cache_class.return_value = mock_cache

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )

        count = await service.cleanup_cache()

        assert count == 42
        mock_cache.cleanup_expired.assert_called_once()

    @patch("service.enrichment_service.EnrichmentCacheManager")
    def test_stats_property(self, mock_cache_class, mock_graphql_client, mock_db_session):
        """Test stats property aggregation."""
        mock_cache = AsyncMock()
        mock_cache.cache_stats = {
            "hits": 10,
            "misses": 5,
            "total_requests": 15,
            "hit_rate": 0.667,
        }
        mock_cache_class.return_value = mock_cache

        mock_graphql_client.rate_limit_info = {"remaining": 4800, "limit": 5000}

        service = EnrichmentService(
            graphql_client=mock_graphql_client,
            db_session=mock_db_session,
            enabled=True,
        )
        service._enrichment_count = 20
        service._error_count = 3

        stats = service.stats

        assert stats["enabled"] is True
        assert stats["enrichment_count"] == 20
        assert stats["error_count"] == 3
        assert stats["cache_hits"] == 10
        assert stats["cache_misses"] == 5
        assert stats["cache_hit_rate"] == 0.667
        assert stats["graphql_rate_limit"] == {"remaining": 4800, "limit": 5000}
