"""Comprehensive unit tests for enhanced AI summarization.

Tests include:
- Full summarization workflow
- Context building with various enrichment combinations
- OpenAI API integration (mocked)
- Database persistence
- Response formatting and field truncation
- Error handling (API failures, unsupported providers)
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
from service.enhanced_summarizer import (
    EnhancedSummaryResponse,
    _build_enhanced_context,
    _generate_with_openai,
    generate_enhanced_summary,
    summarize_enriched_anomaly,
)
from service.sse_models import Severity


@pytest.fixture
def sample_event():
    """Sample GitHub event for testing."""
    return Event(
        id="test-event-123",
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
        payload={"ref": "refs/heads/main", "head": "abc123", "commits": []},
        public=True,
        created_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_enriched_event_minimal(sample_event):
    """Minimal enriched event without enrichment data."""
    return EnrichedEvent(
        event=sample_event,
        anomaly_score=85.0,
        suspicious_patterns=["high_velocity", "new_account"],
    )


@pytest.fixture
def sample_enriched_event_full(sample_event):
    """Fully enriched event with all enrichment data."""
    return EnrichedEvent(
        event=sample_event,
        anomaly_score=95.0,
        suspicious_patterns=["high_velocity", "new_account", "unsigned_commit"],
        actor_profile=ActorProfile(
            login="testuser",
            account_created_at=datetime(2025, 1, 10, tzinfo=timezone.utc),
            account_age_days=5,  # New account
            follower_count=2,
            following_count=1,
            repository_count=1,
            total_commit_contributions=3,
            total_pr_contributions=0,
            total_issue_contributions=0,
            organizations=[],
            is_site_admin=False,
            company=None,
            location=None,
            bio=None,
            cached_at=datetime.utcnow(),
        ),
        repository_context=RepositoryContext(
            owner="testowner",
            name="testrepo",
            full_name="testowner/testrepo",
            stargazer_count=6000,  # Critical repo
            fork_count=500,
            watcher_count=300,
            primary_language="Python",
            default_branch="main",
            has_security_policy=True,
            is_fork=False,
            is_archived=False,
            topics=["security", "critical"],
            license_name="MIT",
            cached_at=datetime.utcnow(),
        ),
        workflow_status=WorkflowStatus(
            repository="testowner/testrepo",
            commit_sha="abc123",
            total_check_suites=3,
            successful_suites=1,
            failed_suites=2,  # Failures detected
            pending_suites=0,
            check_runs=[
                {"name": "test", "conclusion": "SUCCESS"},
                {"name": "security-scan", "conclusion": "FAILURE"},
                {"name": "lint", "conclusion": "FAILURE"},
            ],
            overall_conclusion="FAILURE",
            cached_at=datetime.utcnow(),
        ),
        commit_verification=CommitVerification(
            repository="testowner/testrepo",
            sha="abc123",
            is_signed=False,  # Unsigned
            signer_login=None,
            signature_valid=False,
            additions=1200,  # Large commit
            deletions=300,
            changed_files=50,
            message="Major refactor",
            author_name="Test User",
            author_email="test@example.com",
            commit_entropy=7.5,  # High entropy - suspicious
            commit_size=1500,
            cached_at=datetime.utcnow(),
        ),
    )


@pytest.mark.unit
class TestBuildEnhancedContext:
    """Tests for _build_enhanced_context function."""

    def test_build_context_minimal(self, sample_enriched_event_minimal):
        """Test context building with minimal enrichment."""
        context = _build_enhanced_context(sample_enriched_event_minimal)

        # Check base event information
        assert "Event Type: PushEvent" in context
        assert "Actor: testuser" in context
        assert "Repository: testowner/testrepo" in context
        assert "Anomaly Score: 85.00" in context

        # Check suspicious patterns
        assert "Suspicious Patterns Detected:" in context
        assert "high_velocity" in context
        assert "new_account" in context

        # Check instructions are included
        assert "- title: A concise incident description" in context
        assert "- severity: Choose ONE of: low, medium, high, or critical" in context

    def test_build_context_with_actor_profile_new_account(
        self, sample_enriched_event_full
    ):
        """Test context includes actor profile data (new account)."""
        context = _build_enhanced_context(sample_enriched_event_full)

        assert "ACTOR PROFILE:" in context
        assert "Account Age: 5 days" in context
        assert "NEW ACCOUNT" in context
        assert "Total Contributions: 3" in context
        assert "Low activity" in context
        assert "Followers: 2" in context

    def test_build_context_with_actor_profile_established(self, sample_event):
        """Test context with established account."""
        enriched = EnrichedEvent(
            event=sample_event,
            anomaly_score=50.0,
            suspicious_patterns=[],
            actor_profile=ActorProfile(
                login="established",
                account_created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
                account_age_days=1800,  # 5 years
                follower_count=500,
                following_count=100,
                repository_count=50,
                total_commit_contributions=2000,
                total_pr_contributions=500,
                total_issue_contributions=300,
                organizations=["bigcorp", "opensource-foundation"],
                is_site_admin=True,
                company="Big Corp",
                location="San Francisco",
                bio="Senior engineer",
                cached_at=datetime.utcnow(),
            ),
        )

        context = _build_enhanced_context(enriched)

        assert "Account Age: 1800 days" in context
        assert "Established" in context
        assert "Total Contributions: 2800" in context
        assert "Active" in context
        assert "Organizations: bigcorp, opensource-foundation" in context
        assert "SITE ADMINISTRATOR" in context
        assert "Company: Big Corp" in context

    def test_build_context_with_repository_critical(self, sample_enriched_event_full):
        """Test context includes critical repository data."""
        context = _build_enhanced_context(sample_enriched_event_full)

        assert "REPOSITORY CONTEXT:" in context
        assert "Popularity: 6000 stars" in context
        assert "CRITICAL" in context
        assert "Security Score: 1/1" in context
        assert "Security policy present" in context
        assert "Primary Language: Python" in context
        assert "Topics: security, critical" in context

    def test_build_context_with_repository_archived(self, sample_event):
        """Test context with archived repository."""
        enriched = EnrichedEvent(
            event=sample_event,
            anomaly_score=40.0,
            suspicious_patterns=[],
            repository_context=RepositoryContext(
                owner="owner",
                name="repo",
                full_name="owner/repo",
                stargazer_count=100,
                fork_count=10,
                watcher_count=5,
                primary_language="JavaScript",
                default_branch="main",
                has_security_policy=False,
                is_fork=False,
                is_archived=True,  # Archived
                topics=[],
                license_name=None,
                cached_at=datetime.utcnow(),
            ),
        )

        context = _build_enhanced_context(enriched)

        assert "Repository is ARCHIVED" in context
        assert "Standard" in context  # Not popular

    def test_build_context_with_workflow_failures(self, sample_enriched_event_full):
        """Test context includes workflow failure information."""
        context = _build_enhanced_context(sample_enriched_event_full)

        assert "CI/CD WORKFLOW STATUS:" in context
        assert "Status: FAILURE" in context
        assert "FAILURES DETECTED: 2 failed suites" in context
        assert "Failed checks:" in context
        assert "security-scan" in context

    def test_build_context_with_workflow_all_passed(self, sample_event):
        """Test context with all workflows passing."""
        enriched = EnrichedEvent(
            event=sample_event,
            anomaly_score=50.0,
            suspicious_patterns=[],
            workflow_status=WorkflowStatus(
                repository="owner/repo",
                commit_sha="abc123",
                total_check_suites=3,
                successful_suites=3,
                failed_suites=0,
                pending_suites=0,
                check_runs=[{"name": "test", "conclusion": "SUCCESS"}],
                overall_conclusion="SUCCESS",
                cached_at=datetime.utcnow(),
            ),
        )

        context = _build_enhanced_context(enriched)

        assert "All checks passed" in context

    def test_build_context_with_commit_verification_unsigned(
        self, sample_enriched_event_full
    ):
        """Test context includes unsigned commit information."""
        context = _build_enhanced_context(sample_enriched_event_full)

        assert "COMMIT VERIFICATION:" in context
        assert "Signature: NOT SIGNED" in context
        assert "Commit Size: 1500 lines changed" in context
        assert "[LARGE COMMIT]" in context
        assert "Code Entropy: 7.50" in context
        assert "SUSPICIOUS - POSSIBLE OBFUSCATED CODE" in context
        assert "Author: Test User <test@example.com>" in context

    def test_build_context_with_commit_verification_signed(self, sample_event):
        """Test context with signed, verified commit."""
        enriched = EnrichedEvent(
            event=sample_event,
            anomaly_score=30.0,
            suspicious_patterns=[],
            commit_verification=CommitVerification(
                repository="owner/repo",
                sha="abc123",
                is_signed=True,
                signer_login="trusted-dev",
                signature_valid=True,
                additions=50,
                deletions=30,
                changed_files=3,
                message="Fix bug",
                author_name="Trusted Dev",
                author_email="dev@example.com",
                commit_entropy=4.5,  # Normal
                commit_size=80,
                cached_at=datetime.utcnow(),
            ),
        )

        context = _build_enhanced_context(enriched)

        assert "Signature: Signed" in context
        assert "Signer: trusted-dev" in context
        assert "Code Entropy: 4.50" in context
        assert "Normal" in context
        assert "SUSPICIOUS" not in context

    def test_build_context_payload_truncation(self, sample_event):
        """Test that payload is truncated to 1500 characters."""
        # Create event with large payload
        large_payload = {"data": "x" * 2000}
        sample_event.payload = large_payload

        enriched = EnrichedEvent(
            event=sample_event, anomaly_score=50.0, suspicious_patterns=[]
        )

        context = _build_enhanced_context(enriched)

        # Find the payload section
        payload_section_start = context.find("Event Payload (sample):")
        assert payload_section_start >= 0

        # The payload JSON is truncated with [:1500] in the code
        # Verify truncation happens by checking the "data" field doesn't contain all x's
        assert '"data": "xxxx' in context  # Start of data
        # Full 2000 x's should not be there (truncated)
        assert ("x" * 2000) not in context

    def test_build_context_no_enrichment(self, sample_enriched_event_minimal):
        """Test context with no enrichment data at all."""
        context = _build_enhanced_context(sample_enriched_event_minimal)

        # Should still have base information
        assert "Event Type: PushEvent" in context
        assert "Actor: testuser" in context

        # Should not have enrichment sections
        assert "ACTOR PROFILE:" not in context
        assert "REPOSITORY CONTEXT:" not in context
        assert "CI/CD WORKFLOW STATUS:" not in context
        assert "COMMIT VERIFICATION:" not in context


@pytest.mark.unit
class TestGenerateWithOpenAI:
    """Tests for _generate_with_openai function."""

    @pytest.mark.asyncio
    @patch("service.enhanced_summarizer.AsyncOpenAI")
    @patch("service.enhanced_summarizer.service_settings")
    async def test_generate_with_openai_success(self, mock_settings, mock_openai_class):
        """Test successful OpenAI API call."""
        # Setup mocks
        mock_settings.openai_api_key = "test-api-key"
        mock_settings.ai_model = "gpt-4o"

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.output_parsed = EnhancedSummaryResponse(
            title="Test Incident Title",
            severity=Severity.HIGH,
            severity_reasoning="High severity due to new account and critical repo",
            root_cause=["New account activity", "Unsigned commit", "Failed CI checks"],
            impact=["Potential security breach", "Data exposure risk"],
            next_steps=["Review commit", "Contact user", "Rollback changes"],
            tags=["new-account", "unsigned-commit", "critical-repo"],
        )

        mock_client = AsyncMock()
        mock_client.responses.parse.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Execute
        context = "Test context for OpenAI"
        result = await _generate_with_openai(context)

        # Assertions
        assert result["title"] == "Test Incident Title"
        assert result["severity"] == "high"
        assert result["severity_reasoning"] == "High severity due to new account and critical repo"
        assert len(result["root_cause"]) == 3
        assert len(result["impact"]) == 2
        assert len(result["next_steps"]) == 3
        assert len(result["tags"]) == 3

        # Verify OpenAI was called correctly
        mock_openai_class.assert_called_once_with(api_key="test-api-key")
        mock_client.responses.parse.assert_called_once()

    @pytest.mark.asyncio
    @patch("service.enhanced_summarizer.service_settings")
    async def test_generate_with_openai_missing_api_key(self, mock_settings):
        """Test OpenAI call fails when API key is missing."""
        mock_settings.openai_api_key = None

        context = "Test context"

        with pytest.raises(Exception, match="OpenAI API key not configured"):
            await _generate_with_openai(context)

    @pytest.mark.asyncio
    @patch("service.enhanced_summarizer.AsyncOpenAI")
    @patch("service.enhanced_summarizer.service_settings")
    async def test_generate_with_openai_field_truncation(
        self, mock_settings, mock_openai_class
    ):
        """Test that response fields are truncated to limits."""
        # Setup mocks
        mock_settings.openai_api_key = "test-api-key"
        mock_settings.ai_model = "gpt-4o"

        # Create response with fields exceeding limits
        long_title = "x" * 300  # Exceeds 200 char limit
        long_reasoning = "y" * 600  # Exceeds 500 char limit

        mock_response = MagicMock()
        mock_response.output_parsed = EnhancedSummaryResponse(
            title=long_title,
            severity=Severity.CRITICAL,
            severity_reasoning=long_reasoning,
            root_cause=["item" + str(i) for i in range(10)],  # Exceeds 5 item limit
            impact=["impact" + str(i) for i in range(8)],  # Exceeds 5 item limit
            next_steps=["step" + str(i) for i in range(7)],  # Exceeds 5 item limit
            tags=["tag" + str(i) for i in range(15)],  # Exceeds 10 item limit
        )

        mock_client = AsyncMock()
        mock_client.responses.parse.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Execute
        result = await _generate_with_openai("context")

        # Assertions - check truncation
        assert len(result["title"]) == 200  # Truncated
        assert len(result["severity_reasoning"]) == 500  # Truncated
        assert len(result["root_cause"]) == 5  # Truncated
        assert len(result["impact"]) == 5  # Truncated
        assert len(result["next_steps"]) == 5  # Truncated
        assert len(result["tags"]) == 10  # Truncated

    @pytest.mark.asyncio
    @patch("service.enhanced_summarizer.AsyncOpenAI")
    @patch("service.enhanced_summarizer.service_settings")
    async def test_generate_with_openai_all_severity_levels(
        self, mock_settings, mock_openai_class
    ):
        """Test all severity levels are correctly converted."""
        mock_settings.openai_api_key = "test-api-key"
        mock_settings.ai_model = "gpt-4o"

        for severity in [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]:
            mock_response = MagicMock()
            mock_response.output_parsed = EnhancedSummaryResponse(
                title="Test",
                severity=severity,
                severity_reasoning="Test reasoning",
                root_cause=["test"],
                impact=["test"],
                next_steps=["test"],
                tags=["test"],
            )

            mock_client = AsyncMock()
            mock_client.responses.parse.return_value = mock_response
            mock_openai_class.return_value = mock_client

            result = await _generate_with_openai("context")

            assert result["severity"] == severity.value


@pytest.mark.unit
class TestGenerateEnhancedSummary:
    """Tests for generate_enhanced_summary function."""

    @pytest.mark.asyncio
    @patch("service.enhanced_summarizer._generate_with_openai")
    @patch("service.enhanced_summarizer._build_enhanced_context")
    @patch("service.enhanced_summarizer.service_settings")
    async def test_generate_enhanced_summary_openai(
        self, mock_settings, mock_build_context, mock_generate_openai, sample_enriched_event_full
    ):
        """Test summary generation with OpenAI provider."""
        mock_settings.ai_provider = "openai"
        mock_build_context.return_value = "Enhanced context"
        mock_generate_openai.return_value = {
            "title": "Test",
            "severity": "high",
            "severity_reasoning": "Test",
            "root_cause": ["test"],
            "impact": ["test"],
            "next_steps": ["test"],
            "tags": ["test"],
        }

        result = await generate_enhanced_summary(sample_enriched_event_full)

        assert result["title"] == "Test"
        assert result["severity"] == "high"
        mock_build_context.assert_called_once_with(sample_enriched_event_full)
        mock_generate_openai.assert_called_once_with("Enhanced context")

    @pytest.mark.asyncio
    @patch("service.enhanced_summarizer.service_settings")
    async def test_generate_enhanced_summary_unsupported_provider(
        self, mock_settings, sample_enriched_event_full
    ):
        """Test summary generation fails with unsupported provider."""
        mock_settings.ai_provider = "anthropic"  # Not supported yet

        with pytest.raises(Exception, match="Unsupported AI provider"):
            await generate_enhanced_summary(sample_enriched_event_full)


@pytest.mark.unit
class TestSummarizeEnrichedAnomaly:
    """Tests for summarize_enriched_anomaly function."""

    @pytest.mark.asyncio
    @patch("service.enhanced_summarizer.generate_enhanced_summary")
    @patch("service.enhanced_summarizer.AsyncSessionLocal")
    @patch("service.enhanced_summarizer.AnomalySummary")
    async def test_summarize_enriched_anomaly_success(
        self,
        mock_anomaly_summary_class,
        mock_session_local,
        mock_generate_summary,
        sample_enriched_event_full,
    ):
        """Test full summarization workflow."""
        # Mock summary generation
        mock_generate_summary.return_value = {
            "title": "Critical Security Incident",
            "severity": "critical",
            "severity_reasoning": "New account on critical repo with failures",
            "root_cause": ["New account", "Unsigned commit", "Failed security scan"],
            "impact": ["Security breach risk", "Data exposure"],
            "next_steps": ["Investigate immediately", "Lock repository"],
            "tags": ["critical", "new-account", "unsigned"],
        }

        # Mock database summary object
        mock_db_summary = MagicMock()
        mock_db_summary.to_dict.return_value = {
            "id": 1,
            "event_id": "test-event-123",
            "title": "Critical Security Incident",
            "severity": "critical",
            "severity_reasoning": "New account on critical repo with failures",
            "root_cause": ["New account", "Unsigned commit", "Failed security scan"],
            "impact": ["Security breach risk", "Data exposure"],
            "next_steps": ["Investigate immediately", "Lock repository"],
            "tags": ["critical", "new-account", "unsigned"],
            "created_at": "2025-01-15T12:00:00",
        }
        mock_anomaly_summary_class.return_value = mock_db_summary

        # Mock database session with proper async support
        mock_session = AsyncMock()
        mock_session.add = MagicMock()  # add is sync
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        # Mock the context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session_local.return_value = mock_context

        # Execute
        result = await summarize_enriched_anomaly(sample_enriched_event_full)

        # Assertions
        assert result["title"] == "Critical Security Incident"
        assert result["severity"] == "critical"
        assert result["event_id"] == "test-event-123"
        mock_generate_summary.assert_called_once_with(sample_enriched_event_full)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    @patch("service.enhanced_summarizer.generate_enhanced_summary")
    async def test_summarize_enriched_anomaly_generation_fails(
        self, mock_generate_summary, sample_enriched_event_full
    ):
        """Test that generation errors are propagated."""
        mock_generate_summary.side_effect = Exception("OpenAI API error")

        with pytest.raises(Exception, match="OpenAI API error"):
            await summarize_enriched_anomaly(sample_enriched_event_full)

    @pytest.mark.asyncio
    @patch("service.enhanced_summarizer.generate_enhanced_summary")
    @patch("service.enhanced_summarizer.AsyncSessionLocal")
    @patch("service.enhanced_summarizer.AnomalySummary")
    async def test_summarize_enriched_anomaly_database_commit_fails(
        self,
        mock_anomaly_summary_class,
        mock_session_local,
        mock_generate_summary,
        sample_enriched_event_full,
    ):
        """Test that database errors are propagated."""
        mock_generate_summary.return_value = {
            "title": "Test",
            "severity": "low",
            "severity_reasoning": "Test",
            "root_cause": ["test"],
            "impact": ["test"],
            "next_steps": ["test"],
            "tags": ["test"],
        }

        # Mock database summary object
        mock_db_summary = MagicMock()
        mock_anomaly_summary_class.return_value = mock_db_summary

        # Mock database session that fails on commit
        mock_session = AsyncMock()
        mock_session.add = MagicMock()  # add is sync
        mock_session.commit = AsyncMock(side_effect=Exception("Database error"))

        # Mock the context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_session)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_session_local.return_value = mock_context

        with pytest.raises(Exception, match="Database error"):
            await summarize_enriched_anomaly(sample_enriched_event_full)
