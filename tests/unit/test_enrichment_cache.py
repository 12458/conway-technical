"""Comprehensive unit tests for enrichment cache layer.

Tests include:
- Cache model conversions (to_model/from_model)
- Get operations (cache hits, misses, expiration)
- Set operations (create, update/merge)
- TTL enforcement
- Cleanup operations
- Statistics tracking
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from github_client.enriched_models import (
    ActorProfile,
    CommitVerification,
    RepositoryContext,
    WorkflowStatus,
)
from github_client.enrichment_cache import (
    ActorProfileCache,
    CommitVerificationCache,
    EnrichmentCacheManager,
    RepositoryContextCache,
    WorkflowStatusCache,
    utc_now,
)


@pytest.fixture
def fixed_time():
    """Fixed datetime for testing."""
    return datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_session():
    """Mock AsyncSession for testing."""
    session = AsyncMock(spec=AsyncSession)
    return session


@pytest.fixture
def sample_actor_profile(fixed_time):
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
        organizations=["org1", "org2"],
        is_site_admin=False,
        company="Test Company",
        location="San Francisco",
        bio="Software engineer",
        cached_at=fixed_time,
    )


@pytest.fixture
def sample_repository_context(fixed_time):
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
        topics=["testing", "python"],
        license_name="MIT",
        cached_at=fixed_time,
    )


@pytest.fixture
def sample_workflow_status(fixed_time):
    """Sample WorkflowStatus for testing."""
    return WorkflowStatus(
        repository="testowner/testrepo",
        commit_sha="abc123def456",
        total_check_suites=3,
        successful_suites=2,
        failed_suites=1,
        pending_suites=0,
        check_runs=[
            {"name": "test", "conclusion": "SUCCESS"},
            {"name": "lint", "conclusion": "FAILURE"},
        ],
        overall_conclusion="FAILURE",
        cached_at=fixed_time,
    )


@pytest.fixture
def sample_commit_verification(fixed_time):
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
        cached_at=fixed_time,
    )


@pytest.mark.unit
class TestActorProfileCache:
    """Tests for ActorProfileCache model."""

    def test_from_model_with_all_fields(self, sample_actor_profile):
        """Test converting ActorProfile to cache entity with all fields."""
        cache_entity = ActorProfileCache.from_model(sample_actor_profile)

        assert cache_entity.login == "testuser"
        assert cache_entity.follower_count == 100
        assert cache_entity.following_count == 50
        assert cache_entity.repository_count == 25
        assert cache_entity.total_commit_contributions == 500
        assert cache_entity.total_pr_contributions == 150
        assert cache_entity.total_issue_contributions == 75
        assert cache_entity.organizations == '["org1", "org2"]'
        assert cache_entity.is_site_admin is False
        assert cache_entity.company == "Test Company"
        assert cache_entity.location == "San Francisco"
        assert cache_entity.bio == "Software engineer"

    def test_from_model_with_minimal_fields(self, fixed_time):
        """Test converting ActorProfile with minimal fields."""
        minimal_profile = ActorProfile(
            login="minimaluser",
            account_created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            account_age_days=200,
            follower_count=0,
            following_count=0,
            repository_count=0,
            total_commit_contributions=0,
            total_pr_contributions=0,
            total_issue_contributions=0,
            organizations=[],
            is_site_admin=None,
            company=None,
            location=None,
            bio=None,
            cached_at=fixed_time,
        )

        cache_entity = ActorProfileCache.from_model(minimal_profile)

        assert cache_entity.login == "minimaluser"
        assert cache_entity.follower_count == 0
        assert cache_entity.organizations == "[]"
        assert cache_entity.is_site_admin is None
        assert cache_entity.company is None

    def test_to_model(self, sample_actor_profile, fixed_time):
        """Test converting cache entity back to ActorProfile."""
        cache_entity = ActorProfileCache.from_model(sample_actor_profile)
        restored_profile = cache_entity.to_model()

        assert restored_profile.login == sample_actor_profile.login
        assert restored_profile.follower_count == sample_actor_profile.follower_count
        assert restored_profile.organizations == sample_actor_profile.organizations
        assert restored_profile.is_site_admin == sample_actor_profile.is_site_admin
        assert restored_profile.company == sample_actor_profile.company
        assert restored_profile.cached_at == fixed_time

    def test_empty_organizations_json(self, fixed_time):
        """Test handling empty organizations list."""
        profile = ActorProfile(
            login="user",
            account_created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            account_age_days=100,
            follower_count=0,
            following_count=0,
            repository_count=0,
            total_commit_contributions=0,
            total_pr_contributions=0,
            total_issue_contributions=0,
            organizations=[],
            cached_at=fixed_time,
        )

        cache_entity = ActorProfileCache.from_model(profile)
        restored = cache_entity.to_model()

        assert restored.organizations == []


@pytest.mark.unit
class TestRepositoryContextCache:
    """Tests for RepositoryContextCache model."""

    def test_from_model_with_all_fields(self, sample_repository_context):
        """Test converting RepositoryContext to cache entity."""
        cache_entity = RepositoryContextCache.from_model(sample_repository_context)

        assert cache_entity.owner == "testowner"
        assert cache_entity.name == "testrepo"
        assert cache_entity.full_name == "testowner/testrepo"
        assert cache_entity.stargazer_count == 1500
        assert cache_entity.fork_count == 200
        assert cache_entity.primary_language == "Python"
        assert cache_entity.has_security_policy is True
        assert cache_entity.is_archived is False
        assert cache_entity.topics == '["testing", "python"]'
        assert cache_entity.license_name == "MIT"

    def test_from_model_with_minimal_fields(self, fixed_time):
        """Test converting RepositoryContext with minimal fields."""
        minimal_context = RepositoryContext(
            owner="owner",
            name="repo",
            full_name="owner/repo",
            stargazer_count=0,
            fork_count=0,
            watcher_count=0,
            primary_language=None,
            default_branch=None,
            has_security_policy=False,
            is_fork=False,
            is_archived=False,
            topics=[],
            license_name=None,
            cached_at=fixed_time,
        )

        cache_entity = RepositoryContextCache.from_model(minimal_context)

        assert cache_entity.owner == "owner"
        assert cache_entity.stargazer_count == 0
        assert cache_entity.topics == "[]"
        assert cache_entity.primary_language is None
        assert cache_entity.license_name is None

    def test_to_model(self, sample_repository_context):
        """Test converting cache entity back to RepositoryContext."""
        cache_entity = RepositoryContextCache.from_model(sample_repository_context)
        restored_context = cache_entity.to_model()

        assert restored_context.owner == sample_repository_context.owner
        assert restored_context.name == sample_repository_context.name
        assert (
            restored_context.stargazer_count
            == sample_repository_context.stargazer_count
        )
        assert restored_context.topics == sample_repository_context.topics
        assert (
            restored_context.has_security_policy
            == sample_repository_context.has_security_policy
        )


@pytest.mark.unit
class TestWorkflowStatusCache:
    """Tests for WorkflowStatusCache model."""

    def test_from_model_with_all_fields(self, sample_workflow_status):
        """Test converting WorkflowStatus to cache entity."""
        cache_entity = WorkflowStatusCache.from_model(sample_workflow_status)

        assert cache_entity.repository == "testowner/testrepo"
        assert cache_entity.commit_sha == "abc123def456"
        assert cache_entity.total_check_suites == 3
        assert cache_entity.successful_suites == 2
        assert cache_entity.failed_suites == 1
        assert cache_entity.pending_suites == 0
        check_runs = json.loads(cache_entity.check_runs)
        assert len(check_runs) == 2
        assert check_runs[0]["name"] == "test"
        assert cache_entity.overall_conclusion == "FAILURE"

    def test_to_model(self, sample_workflow_status):
        """Test converting cache entity back to WorkflowStatus."""
        cache_entity = WorkflowStatusCache.from_model(sample_workflow_status)
        restored_status = cache_entity.to_model()

        assert restored_status.repository == sample_workflow_status.repository
        assert restored_status.commit_sha == sample_workflow_status.commit_sha
        assert (
            restored_status.total_check_suites
            == sample_workflow_status.total_check_suites
        )
        assert restored_status.check_runs == sample_workflow_status.check_runs
        assert (
            restored_status.overall_conclusion
            == sample_workflow_status.overall_conclusion
        )

    def test_empty_check_runs(self, fixed_time):
        """Test handling empty check runs list."""
        status = WorkflowStatus(
            repository="owner/repo",
            commit_sha="sha123",
            total_check_suites=0,
            successful_suites=0,
            failed_suites=0,
            pending_suites=0,
            check_runs=[],
            overall_conclusion=None,
            cached_at=fixed_time,
        )

        cache_entity = WorkflowStatusCache.from_model(status)
        restored = cache_entity.to_model()

        assert restored.check_runs == []


@pytest.mark.unit
class TestCommitVerificationCache:
    """Tests for CommitVerificationCache model."""

    def test_from_model_with_all_fields(self, sample_commit_verification):
        """Test converting CommitVerification to cache entity."""
        cache_entity = CommitVerificationCache.from_model(sample_commit_verification)

        assert cache_entity.repository == "testowner/testrepo"
        assert cache_entity.sha == "abc123def456"
        assert cache_entity.is_signed is True
        assert cache_entity.signer_login == "testuser"
        assert cache_entity.signature_valid is True
        assert cache_entity.additions == 100
        assert cache_entity.deletions == 50
        assert cache_entity.changed_files == 5
        assert cache_entity.message == "Test commit"
        assert cache_entity.author_name == "Test User"
        assert cache_entity.commit_entropy == 5.5
        assert cache_entity.commit_size == 150

    def test_to_model(self, sample_commit_verification):
        """Test converting cache entity back to CommitVerification."""
        cache_entity = CommitVerificationCache.from_model(sample_commit_verification)
        restored_verification = cache_entity.to_model()

        assert restored_verification.repository == sample_commit_verification.repository
        assert restored_verification.sha == sample_commit_verification.sha
        assert restored_verification.is_signed == sample_commit_verification.is_signed
        assert restored_verification.additions == sample_commit_verification.additions
        assert (
            restored_verification.commit_entropy
            == sample_commit_verification.commit_entropy
        )


@pytest.mark.unit
class TestEnrichmentCacheManager:
    """Tests for EnrichmentCacheManager."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_session):
        """Test cache manager initialization."""
        manager = EnrichmentCacheManager(mock_session)

        assert manager.session == mock_session
        assert manager._cache_hits == 0
        assert manager._cache_misses == 0

    @pytest.mark.asyncio
    async def test_get_actor_profile_cache_hit(
        self, mock_session, sample_actor_profile, fixed_time
    ):
        """Test getting actor profile from cache (cache hit)."""
        # Arrange
        cache_entity = ActorProfileCache.from_model(sample_actor_profile)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entity
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        with patch("github_client.enrichment_cache.utc_now", return_value=fixed_time):
            profile = await manager.get_actor_profile("testuser")

        # Assert
        assert profile is not None
        assert profile.login == "testuser"
        assert manager._cache_hits == 1
        assert manager._cache_misses == 0

    @pytest.mark.asyncio
    async def test_get_actor_profile_cache_miss(self, mock_session):
        """Test getting actor profile when not in cache."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        profile = await manager.get_actor_profile("nonexistent")

        # Assert
        assert profile is None
        assert manager._cache_hits == 0
        assert manager._cache_misses == 1

    @pytest.mark.asyncio
    async def test_get_actor_profile_expired(
        self, mock_session, sample_actor_profile, fixed_time
    ):
        """Test getting actor profile when cache is expired."""
        # Arrange
        old_time = fixed_time - timedelta(hours=25)  # Expired (TTL = 24 hours)
        sample_actor_profile.cached_at = old_time
        cache_entity = ActorProfileCache.from_model(sample_actor_profile)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entity
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        with patch("github_client.enrichment_cache.utc_now", return_value=fixed_time):
            profile = await manager.get_actor_profile("testuser")

        # Assert
        assert profile is None  # Expired, should return None
        assert manager._cache_hits == 0
        assert manager._cache_misses == 1

    @pytest.mark.asyncio
    async def test_set_actor_profile(self, mock_session, sample_actor_profile):
        """Test storing actor profile in cache."""
        # Arrange
        manager = EnrichmentCacheManager(mock_session)

        # Act
        await manager.set_actor_profile(sample_actor_profile)

        # Assert
        mock_session.merge.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_repository_context_cache_hit(
        self, mock_session, sample_repository_context, fixed_time
    ):
        """Test getting repository context from cache (cache hit)."""
        # Arrange
        cache_entity = RepositoryContextCache.from_model(sample_repository_context)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entity
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        with patch("github_client.enrichment_cache.utc_now", return_value=fixed_time):
            context = await manager.get_repository_context("testowner", "testrepo")

        # Assert
        assert context is not None
        assert context.owner == "testowner"
        assert context.name == "testrepo"
        assert manager._cache_hits == 1
        assert manager._cache_misses == 0

    @pytest.mark.asyncio
    async def test_get_repository_context_cache_miss(self, mock_session):
        """Test getting repository context when not in cache."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        context = await manager.get_repository_context("owner", "repo")

        # Assert
        assert context is None
        assert manager._cache_hits == 0
        assert manager._cache_misses == 1

    @pytest.mark.asyncio
    async def test_get_repository_context_expired(
        self, mock_session, sample_repository_context, fixed_time
    ):
        """Test getting repository context when cache is expired."""
        # Arrange
        old_time = fixed_time - timedelta(hours=2)  # Expired (TTL = 1 hour)
        sample_repository_context.cached_at = old_time
        cache_entity = RepositoryContextCache.from_model(sample_repository_context)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entity
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        with patch("github_client.enrichment_cache.utc_now", return_value=fixed_time):
            context = await manager.get_repository_context("testowner", "testrepo")

        # Assert
        assert context is None  # Expired
        assert manager._cache_hits == 0
        assert manager._cache_misses == 1

    @pytest.mark.asyncio
    async def test_set_repository_context(
        self, mock_session, sample_repository_context
    ):
        """Test storing repository context in cache."""
        # Arrange
        manager = EnrichmentCacheManager(mock_session)

        # Act
        await manager.set_repository_context(sample_repository_context)

        # Assert
        mock_session.merge.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_workflow_status_cache_hit(
        self, mock_session, sample_workflow_status, fixed_time
    ):
        """Test getting workflow status from cache (cache hit)."""
        # Arrange
        cache_entity = WorkflowStatusCache.from_model(sample_workflow_status)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entity
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        with patch("github_client.enrichment_cache.utc_now", return_value=fixed_time):
            status = await manager.get_workflow_status(
                "testowner/testrepo", "abc123def456"
            )

        # Assert
        assert status is not None
        assert status.repository == "testowner/testrepo"
        assert status.commit_sha == "abc123def456"
        assert manager._cache_hits == 1
        assert manager._cache_misses == 0

    @pytest.mark.asyncio
    async def test_get_workflow_status_expired(
        self, mock_session, sample_workflow_status, fixed_time
    ):
        """Test getting workflow status when cache is expired."""
        # Arrange
        old_time = fixed_time - timedelta(minutes=6)  # Expired (TTL = 5 minutes)
        sample_workflow_status.cached_at = old_time
        cache_entity = WorkflowStatusCache.from_model(sample_workflow_status)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entity
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        with patch("github_client.enrichment_cache.utc_now", return_value=fixed_time):
            status = await manager.get_workflow_status(
                "testowner/testrepo", "abc123def456"
            )

        # Assert
        assert status is None  # Expired
        assert manager._cache_hits == 0
        assert manager._cache_misses == 1

    @pytest.mark.asyncio
    async def test_set_workflow_status(self, mock_session, sample_workflow_status):
        """Test storing workflow status in cache."""
        # Arrange
        manager = EnrichmentCacheManager(mock_session)

        # Act
        await manager.set_workflow_status(sample_workflow_status)

        # Assert
        mock_session.merge.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_commit_verification_cache_hit(
        self, mock_session, sample_commit_verification
    ):
        """Test getting commit verification from cache (no expiration)."""
        # Arrange
        cache_entity = CommitVerificationCache.from_model(sample_commit_verification)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entity
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        verification = await manager.get_commit_verification(
            "testowner/testrepo", "abc123def456"
        )

        # Assert
        assert verification is not None
        assert verification.repository == "testowner/testrepo"
        assert verification.sha == "abc123def456"
        assert manager._cache_hits == 1
        assert manager._cache_misses == 0

    @pytest.mark.asyncio
    async def test_get_commit_verification_cache_miss(self, mock_session):
        """Test getting commit verification when not in cache."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        verification = await manager.get_commit_verification("owner/repo", "sha123")

        # Assert
        assert verification is None
        assert manager._cache_hits == 0
        assert manager._cache_misses == 1

    @pytest.mark.asyncio
    async def test_get_commit_verification_no_expiration(
        self, mock_session, sample_commit_verification, fixed_time
    ):
        """Test that commit verification cache never expires."""
        # Arrange - Very old cached time
        old_time = fixed_time - timedelta(days=365)  # 1 year old
        sample_commit_verification.cached_at = old_time
        cache_entity = CommitVerificationCache.from_model(sample_commit_verification)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entity
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        verification = await manager.get_commit_verification(
            "testowner/testrepo", "abc123def456"
        )

        # Assert - Should still be valid despite being very old
        assert verification is not None
        assert manager._cache_hits == 1
        assert manager._cache_misses == 0

    @pytest.mark.asyncio
    async def test_set_commit_verification(
        self, mock_session, sample_commit_verification
    ):
        """Test storing commit verification in cache."""
        # Arrange
        manager = EnrichmentCacheManager(mock_session)

        # Act
        await manager.set_commit_verification(sample_commit_verification)

        # Assert
        mock_session.merge.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_expired_actors_only(self, mock_session, fixed_time):
        """Test cleanup removes only expired actor profiles."""
        # Arrange
        expired_actor = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value = [expired_actor]
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        with patch("github_client.enrichment_cache.utc_now", return_value=fixed_time):
            count = await manager.cleanup_expired()

        # Assert
        assert count == 3  # One from each category (actors, repos, workflows)
        assert mock_session.delete.call_count == 3
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_expired_nothing_to_clean(self, mock_session):
        """Test cleanup when nothing is expired."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalars.return_value = []
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        count = await manager.cleanup_expired()

        # Assert
        assert count == 0
        mock_session.delete.assert_not_called()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_stats_initial(self, mock_session):
        """Test cache stats with no activity."""
        # Arrange
        manager = EnrichmentCacheManager(mock_session)

        # Act
        stats = manager.cache_stats

        # Assert
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["total_requests"] == 0
        assert stats["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_cache_stats_after_hits(
        self, mock_session, sample_actor_profile, fixed_time
    ):
        """Test cache stats after cache hits."""
        # Arrange
        cache_entity = ActorProfileCache.from_model(sample_actor_profile)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entity
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        with patch("github_client.enrichment_cache.utc_now", return_value=fixed_time):
            await manager.get_actor_profile("testuser")
            await manager.get_actor_profile("testuser")

        stats = manager.cache_stats

        # Assert
        assert stats["hits"] == 2
        assert stats["misses"] == 0
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_cache_stats_mixed_hits_and_misses(self, mock_session, fixed_time):
        """Test cache stats with mixed hits and misses."""
        # Arrange
        manager = EnrichmentCacheManager(mock_session)

        # Simulate 3 hits and 2 misses
        manager._cache_hits = 3
        manager._cache_misses = 2

        # Act
        stats = manager.cache_stats

        # Assert
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["total_requests"] == 5
        assert stats["hit_rate"] == 0.6

    @pytest.mark.asyncio
    async def test_ttl_boundary_exactly_expired(
        self, mock_session, sample_actor_profile, fixed_time
    ):
        """Test TTL boundary condition - exactly at expiration time."""
        # Arrange - Cached exactly 24 hours and 1 second ago (just expired)
        old_time = fixed_time - timedelta(hours=24, seconds=1)
        sample_actor_profile.cached_at = old_time
        cache_entity = ActorProfileCache.from_model(sample_actor_profile)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entity
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        with patch("github_client.enrichment_cache.utc_now", return_value=fixed_time):
            profile = await manager.get_actor_profile("testuser")

        # Assert - Just past boundary, should be considered expired (> TTL)
        assert profile is None
        assert manager._cache_misses == 1

    @pytest.mark.asyncio
    async def test_ttl_boundary_just_valid(
        self, mock_session, sample_actor_profile, fixed_time
    ):
        """Test TTL boundary condition - just before expiration."""
        # Arrange - Cached 23 hours and 59 minutes ago (just valid)
        old_time = fixed_time - timedelta(hours=23, minutes=59)
        sample_actor_profile.cached_at = old_time
        cache_entity = ActorProfileCache.from_model(sample_actor_profile)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = cache_entity
        mock_session.execute.return_value = mock_result

        manager = EnrichmentCacheManager(mock_session)

        # Act
        with patch("github_client.enrichment_cache.utc_now", return_value=fixed_time):
            profile = await manager.get_actor_profile("testuser")

        # Assert - Should still be valid
        assert profile is not None
        assert manager._cache_hits == 1
