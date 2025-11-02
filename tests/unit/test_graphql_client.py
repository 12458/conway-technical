"""Comprehensive unit tests for GitHubGraphQLClient.

Tests include:
- Shannon entropy calculation
- Client initialization and lifecycle
- Rate limit handling
- Query execution with retries
- Actor profile enrichment
- Repository context enrichment
- Workflow status fetching
- Commit verification with entropy analysis
- Error handling and edge cases
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gql.transport.exceptions import TransportQueryError

from github_client.graphql_client import GitHubGraphQLClient, calculate_shannon_entropy
from github_client.enriched_models import ActorProfile, RepositoryContext


@pytest.mark.unit
class TestShannonEntropy:
    """Tests for Shannon entropy calculation."""

    def test_entropy_empty_string(self):
        """Test entropy of empty string is zero."""
        assert calculate_shannon_entropy("") == 0.0

    def test_entropy_single_character(self):
        """Test entropy of single repeated character."""
        # "aaaa" has only one character, so entropy is 0
        entropy = calculate_shannon_entropy("aaaa")
        assert entropy == 0.0

    def test_entropy_uniform_distribution(self):
        """Test entropy of uniformly distributed characters."""
        # All different characters has maximum entropy
        text = "abcdefgh"  # 8 unique characters
        entropy = calculate_shannon_entropy(text)

        # For 8 unique characters, max entropy is log2(8) = 3.0
        assert entropy == pytest.approx(3.0, abs=0.01)

    def test_entropy_obfuscated_code(self):
        """Test entropy of obfuscated/random code is high."""
        # Simulate obfuscated code with random-looking characters
        obfuscated = "xK9vL2mN8qR5tY7pW3zA1bC4dE6fG0hJ"
        entropy = calculate_shannon_entropy(obfuscated)

        # Obfuscated code typically has entropy > 5
        assert entropy > 4.0

    def test_entropy_normal_code(self):
        """Test entropy of normal code."""
        normal_code = "function hello() { return 'world'; }"
        entropy = calculate_shannon_entropy(normal_code)

        # Normal code typically has entropy 4-6
        assert 3.0 < entropy < 6.5

    def test_entropy_calculation_formula(self):
        """Test that entropy calculation follows Shannon's formula."""
        text = "aabb"
        # 2 'a', 2 'b' -> p(a)=0.5, p(b)=0.5
        # H = -(0.5*log2(0.5) + 0.5*log2(0.5)) = -(-0.5 + -0.5) = 1.0
        entropy = calculate_shannon_entropy(text)
        assert entropy == pytest.approx(1.0, abs=0.01)

    def test_entropy_with_spaces(self):
        """Test entropy calculation includes spaces."""
        text_with_spaces = "hello world"
        entropy = calculate_shannon_entropy(text_with_spaces)

        # Should be non-zero and reasonable
        assert entropy > 0
        assert entropy < 5


@pytest.mark.unit
class TestGraphQLClientInitialization:
    """Tests for GitHubGraphQLClient initialization."""

    def test_client_initialization(self):
        """Test client initializes with token."""
        token = "ghp_test_token_123"
        client = GitHubGraphQLClient(token)

        assert client.token == token
        assert client.base_url == "https://api.github.com/graphql"
        assert client._client is None
        assert client._rate_limit_remaining == 5000
        assert client._rate_limit_reset_at is None
        assert client._consecutive_errors == 0
        assert client._max_consecutive_errors == 3

    def test_client_initialization_custom_url(self):
        """Test client initializes with custom URL."""
        token = "ghp_test_token"
        custom_url = "https://github.enterprise.com/api/graphql"
        client = GitHubGraphQLClient(token, base_url=custom_url)

        assert client.base_url == custom_url

    @pytest.mark.asyncio
    async def test_client_connect(self):
        """Test client connection establishes transport."""
        client = GitHubGraphQLClient("ghp_test_token")

        with patch("github_client.graphql_client.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            await client.connect()

            assert client._client == mock_client
            mock_client_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_close(self):
        """Test client close cleans up resources."""
        client = GitHubGraphQLClient("ghp_test_token")

        # Set up mock client with transport
        mock_transport = AsyncMock()
        mock_client = MagicMock()
        mock_client.transport = mock_transport
        client._client = mock_client

        await client.close()

        mock_transport.close.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager lifecycle."""
        client = GitHubGraphQLClient("ghp_test_token")

        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(client, "close", new_callable=AsyncMock) as mock_close:
                async with client:
                    pass

                mock_connect.assert_called_once()
                mock_close.assert_called_once()


@pytest.mark.unit
class TestQueryExecution:
    """Tests for GraphQL query execution."""

    @pytest.mark.asyncio
    async def test_execute_query_success(self):
        """Test successful query execution."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {"data": "test"}
        mock_client = MagicMock()
        mock_client.execute_async = AsyncMock(return_value=mock_result)
        client._client = mock_client

        result = await client._execute_query("query { test }", {})

        assert result == mock_result
        assert client._consecutive_errors == 0

    @pytest.mark.asyncio
    async def test_execute_query_auto_connect(self):
        """Test query execution auto-connects if not connected."""
        client = GitHubGraphQLClient("ghp_test_token")

        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            mock_client = MagicMock()
            mock_client.execute_async = AsyncMock(return_value={})

            async def set_client():
                client._client = mock_client

            mock_connect.side_effect = set_client

            await client._execute_query("query { test }", {})

            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_query_rate_limit_wait(self):
        """Test query waits when rate limit is low."""
        client = GitHubGraphQLClient("ghp_test_token")
        client._rate_limit_remaining = 5  # Very low
        client._rate_limit_reset_at = datetime.now(timezone.utc) + timedelta(seconds=2)

        mock_client = MagicMock()
        mock_client.execute_async = AsyncMock(return_value={})
        client._client = mock_client

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await client._execute_query("query { test }", {})

            # Should have slept waiting for rate limit
            mock_sleep.assert_called()
            call_args = mock_sleep.call_args[0][0]
            assert 0 < call_args < 5  # Should wait a few seconds

    @pytest.mark.asyncio
    async def test_execute_query_retry_on_transport_error(self):
        """Test query retries on TransportQueryError."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_client = MagicMock()
        # First call fails, second succeeds
        mock_client.execute_async = AsyncMock(
            side_effect=[TransportQueryError("Temporary error"), {"data": "success"}]
        )
        client._client = mock_client

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client._execute_query("query { test }", {}, retries=3)

        assert result == {"data": "success"}
        assert mock_client.execute_async.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_query_rate_limit_backoff(self):
        """Test exponential backoff on rate limit error."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_client = MagicMock()
        mock_client.execute_async = AsyncMock(
            side_effect=TransportQueryError("rate limit exceeded")
        )
        client._client = mock_client

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client._execute_query("query { test }", {}, retries=3)

        assert result is None
        # Should have slept with exponential backoff: 5s, 10s, 20s
        assert mock_sleep.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_query_max_consecutive_errors(self):
        """Test that consecutive errors are tracked."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_client = MagicMock()
        mock_client.execute_async = AsyncMock(side_effect=TransportQueryError("error"))
        client._client = mock_client

        with patch("asyncio.sleep", new_callable=AsyncMock):
            # First error
            await client._execute_query("query { test }", {}, retries=1)
            assert client._consecutive_errors == 1

            # Second error
            await client._execute_query("query { test }", {}, retries=1)
            assert client._consecutive_errors == 2

            # Third error triggers critical log
            await client._execute_query("query { test }", {}, retries=1)
            assert client._consecutive_errors == 3

    @pytest.mark.asyncio
    async def test_execute_query_unexpected_exception(self):
        """Test handling of unexpected exceptions."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_client = MagicMock()
        mock_client.execute_async = AsyncMock(side_effect=RuntimeError("Unexpected"))
        client._client = mock_client

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await client._execute_query("query { test }", {}, retries=2)

        assert result is None


@pytest.mark.unit
class TestActorProfileEnrichment:
    """Tests for actor profile enrichment."""

    @pytest.mark.asyncio
    async def test_get_actor_profile_success(self):
        """Test successful actor profile retrieval."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {
            "user": {
                "createdAt": "2020-01-01T00:00:00Z",
                "followers": {"totalCount": 100},
                "following": {"totalCount": 50},
                "repositories": {"totalCount": 25},
                "contributionsCollection": {
                    "totalCommitContributions": 500,
                    "totalPullRequestContributions": 50,
                    "totalIssueContributions": 30,
                },
                "organizations": {
                    "nodes": [
                        {"login": "org1"},
                        {"login": "org2"},
                    ]
                },
                "isSiteAdmin": False,
                "company": "Acme Corp",
                "location": "San Francisco",
                "bio": "Software developer",
            },
            "rateLimit": {
                "remaining": 4500,
                "resetAt": "2025-01-01T12:00:00Z",
            },
        }

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            profile = await client.get_actor_profile("testuser")

        assert profile is not None
        assert profile.login == "testuser"
        assert profile.follower_count == 100
        assert profile.following_count == 50
        assert profile.repository_count == 25
        assert profile.total_commit_contributions == 500
        assert profile.organizations == ["org1", "org2"]
        assert profile.company == "Acme Corp"
        assert profile.location == "San Francisco"
        assert profile.bio == "Software developer"

        # Check rate limit was updated
        assert client._rate_limit_remaining == 4500

    @pytest.mark.asyncio
    async def test_get_actor_profile_bot_filtered(self):
        """Test that bot accounts are filtered."""
        client = GitHubGraphQLClient("ghp_test_token")

        profile = await client.get_actor_profile("github-actions[bot]")
        assert profile is None

        profile = await client.get_actor_profile("renovate-bot")
        assert profile is None

    @pytest.mark.asyncio
    async def test_get_actor_profile_no_user_found(self):
        """Test handling when user is not found."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {"user": None}

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            profile = await client.get_actor_profile("nonexistent")

        assert profile is None

    @pytest.mark.asyncio
    async def test_get_actor_profile_query_error(self):
        """Test handling of query errors."""
        client = GitHubGraphQLClient("ghp_test_token")

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=None
        ):
            profile = await client.get_actor_profile("testuser")

        assert profile is None

    @pytest.mark.asyncio
    async def test_get_actor_profile_account_age_calculation(self):
        """Test that account age is correctly calculated."""
        client = GitHubGraphQLClient("ghp_test_token")

        # Create account 30 days ago
        created_at = datetime.now(timezone.utc) - timedelta(days=30)

        mock_result = {
            "user": {
                "createdAt": created_at.isoformat().replace("+00:00", "Z"),
                "followers": {"totalCount": 0},
                "following": {"totalCount": 0},
                "repositories": {"totalCount": 0},
                "contributionsCollection": {
                    "totalCommitContributions": 0,
                    "totalPullRequestContributions": 0,
                    "totalIssueContributions": 0,
                },
                "organizations": {"nodes": []},
            },
            "rateLimit": {"remaining": 5000, "resetAt": "2025-01-01T12:00:00Z"},
        }

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            profile = await client.get_actor_profile("newuser")

        assert profile is not None
        assert 29 <= profile.account_age_days <= 31  # Allow 1 day tolerance


@pytest.mark.unit
class TestRepositoryContextEnrichment:
    """Tests for repository context enrichment."""

    @pytest.mark.asyncio
    async def test_get_repository_context_success(self):
        """Test successful repository context retrieval."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {
            "repository": {
                "stargazerCount": 1000,
                "forkCount": 200,
                "watchers": {"totalCount": 50},
                "primaryLanguage": {"name": "Python"},
                "defaultBranchRef": {"name": "main"},
                "securityPolicyUrl": "https://github.com/owner/repo/security",
                "isFork": False,
                "isArchived": False,
                "repositoryTopics": {
                    "nodes": [
                        {"topic": {"name": "python"}},
                        {"topic": {"name": "machine-learning"}},
                    ]
                },
                "licenseInfo": {"name": "MIT License"},
            },
            "rateLimit": {"remaining": 4500, "resetAt": "2025-01-01T12:00:00Z"},
        }

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            context = await client.get_repository_context("owner", "repo")

        assert context is not None
        assert context.full_name == "owner/repo"
        assert context.stargazer_count == 1000
        assert context.fork_count == 200
        assert context.watcher_count == 50
        assert context.primary_language == "Python"
        assert context.default_branch == "main"
        assert context.has_security_policy is True
        assert context.is_fork is False
        assert context.is_archived is False
        assert context.topics == ["python", "machine-learning"]
        assert context.license_name == "MIT License"

    @pytest.mark.asyncio
    async def test_get_repository_context_no_repo_found(self):
        """Test handling when repository is not found."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {"repository": None}

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            context = await client.get_repository_context("owner", "nonexistent")

        assert context is None

    @pytest.mark.asyncio
    async def test_get_repository_context_private_repo(self):
        """Test repository context for private repo with minimal data."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {
            "repository": {
                "stargazerCount": 0,
                "forkCount": 0,
                "watchers": {"totalCount": 1},
                "primaryLanguage": None,
                "defaultBranchRef": None,
                "securityPolicyUrl": None,
                "isFork": False,
                "isArchived": False,
                "repositoryTopics": {"nodes": []},
                "licenseInfo": None,
            },
            "rateLimit": {"remaining": 4500, "resetAt": "2025-01-01T12:00:00Z"},
        }

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            context = await client.get_repository_context("owner", "private-repo")

        assert context is not None
        assert context.primary_language is None
        assert context.default_branch is None
        assert context.has_security_policy is False
        assert context.topics == []
        assert context.license_name is None


@pytest.mark.unit
class TestRateLimitHandling:
    """Tests for rate limit tracking and handling."""

    def test_rate_limit_initialization(self):
        """Test rate limit starts at 5000."""
        client = GitHubGraphQLClient("ghp_test_token")
        assert client._rate_limit_remaining == 5000
        assert client._rate_limit_reset_at is None

    @pytest.mark.asyncio
    async def test_rate_limit_updated_from_response(self):
        """Test rate limit is updated from GraphQL response."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {
            "user": {
                "createdAt": "2020-01-01T00:00:00Z",
                "followers": {"totalCount": 0},
                "following": {"totalCount": 0},
                "repositories": {"totalCount": 0},
                "contributionsCollection": {
                    "totalCommitContributions": 0,
                    "totalPullRequestContributions": 0,
                    "totalIssueContributions": 0,
                },
                "organizations": {"nodes": []},
            },
            "rateLimit": {
                "remaining": 3000,
                "resetAt": "2025-01-01T15:00:00Z",
            },
        }

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            await client.get_actor_profile("testuser")

        assert client._rate_limit_remaining == 3000
        assert client._rate_limit_reset_at is not None

    @pytest.mark.asyncio
    async def test_low_rate_limit_warning(self):
        """Test warning is logged when rate limit is low."""
        client = GitHubGraphQLClient("ghp_test_token")
        client._rate_limit_remaining = 50  # Low but not critical

        mock_client = MagicMock()
        mock_client.execute_async = AsyncMock(return_value={})
        client._client = mock_client

        with patch("github_client.graphql_client.logger") as mock_logger:
            await client._execute_query("query { test }", {})

            # Should log warning
            assert any(
                "Low rate limit" in str(call)
                for call in mock_logger.warning.call_args_list
            )


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_get_actor_profile_minimal_data(self):
        """Test actor profile with minimal/missing optional fields."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {
            "user": {
                "createdAt": "2020-01-01T00:00:00Z",
                "followers": {"totalCount": 0},
                "following": {"totalCount": 0},
                "repositories": {"totalCount": 0},
                "contributionsCollection": {
                    "totalCommitContributions": 0,
                    "totalPullRequestContributions": 0,
                    "totalIssueContributions": 0,
                },
                "organizations": {"nodes": []},
                # Optional fields missing
            },
            "rateLimit": {"remaining": 5000, "resetAt": "2025-01-01T12:00:00Z"},
        }

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            profile = await client.get_actor_profile("minimaluser")

        assert profile is not None
        assert profile.is_site_admin is None
        assert profile.company is None
        assert profile.location is None
        assert profile.bio is None

    @pytest.mark.asyncio
    async def test_consecutive_errors_reset_on_success(self):
        """Test that consecutive error counter resets on success."""
        client = GitHubGraphQLClient("ghp_test_token")
        client._consecutive_errors = 2

        mock_client = MagicMock()
        mock_client.execute_async = AsyncMock(return_value={"data": "success"})
        client._client = mock_client

        await client._execute_query("query { test }", {})

        assert client._consecutive_errors == 0

    def test_entropy_very_long_text(self):
        """Test entropy calculation with very long text."""
        long_text = "a" * 10000 + "b" * 10000
        entropy = calculate_shannon_entropy(long_text)

        # Should be exactly 1.0 (two equally likely characters)
        assert entropy == pytest.approx(1.0, abs=0.01)

    def test_entropy_special_characters(self):
        """Test entropy with special characters."""
        special_text = "!@#$%^&*()_+-=[]{}|;:',.<>?/"
        entropy = calculate_shannon_entropy(special_text)

        assert entropy > 0
        assert entropy < 6


@pytest.mark.unit
class TestWorkflowStatusEnrichment:
    """Tests for workflow status enrichment."""

    @pytest.mark.asyncio
    async def test_get_workflow_status_multiple_check_suites(self):
        """Test workflow status with multiple check suites."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {
            "repository": {
                "object": {
                    "checkSuites": {
                        "nodes": [
                            {
                                "conclusion": "SUCCESS",
                                "status": "COMPLETED",
                                "checkRuns": {
                                    "nodes": [
                                        {"name": "test", "conclusion": "SUCCESS", "status": "COMPLETED"},
                                        {"name": "lint", "conclusion": "SUCCESS", "status": "COMPLETED"},
                                    ]
                                },
                            },
                            {
                                "conclusion": "FAILURE",
                                "status": "COMPLETED",
                                "checkRuns": {
                                    "nodes": [
                                        {"name": "build", "conclusion": "FAILURE", "status": "COMPLETED"}
                                    ]
                                },
                            },
                        ]
                    }
                }
            },
            "rateLimit": {"remaining": 4500, "resetAt": "2025-01-01T12:00:00Z"},
        }

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            status = await client.get_workflow_status("owner", "repo", "abc123")

        assert status is not None
        assert status.repository == "owner/repo"
        assert status.commit_sha == "abc123"
        assert status.total_check_suites == 2
        assert status.successful_suites == 1
        assert status.failed_suites == 1
        assert status.overall_conclusion == "FAILURE"  # Failed > 0
        assert len(status.check_runs) == 3

    @pytest.mark.asyncio
    async def test_get_workflow_status_pending(self):
        """Test workflow status with pending check suites."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {
            "repository": {
                "object": {
                    "checkSuites": {
                        "nodes": [
                            {
                                "conclusion": None,
                                "status": "IN_PROGRESS",
                                "checkRuns": {
                                    "nodes": [
                                        {"name": "test", "conclusion": None, "status": "IN_PROGRESS"}
                                    ]
                                },
                            }
                        ]
                    }
                }
            },
            "rateLimit": {"remaining": 4500, "resetAt": "2025-01-01T12:00:00Z"},
        }

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            status = await client.get_workflow_status("owner", "repo", "abc123")

        assert status is not None
        assert status.pending_suites == 1
        assert status.overall_conclusion == "PENDING"

    @pytest.mark.asyncio
    async def test_get_workflow_status_empty_check_suites(self):
        """Test workflow status with no check suites."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {
            "repository": {"object": {}},  # No checkSuites key
            "rateLimit": {"remaining": 4500, "resetAt": "2025-01-01T12:00:00Z"},
        }

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            status = await client.get_workflow_status("owner", "repo", "abc123")

        assert status is not None
        assert status.total_check_suites == 0
        assert status.successful_suites == 0
        assert status.failed_suites == 0
        assert status.overall_conclusion is None

    @pytest.mark.asyncio
    async def test_get_workflow_status_no_repository(self):
        """Test workflow status when repository is not found."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {"repository": None}

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            status = await client.get_workflow_status("owner", "nonexistent", "abc123")

        assert status is None


@pytest.mark.unit
class TestCommitVerification:
    """Tests for commit verification enrichment."""

    @pytest.mark.asyncio
    async def test_get_commit_verification_signed_commit(self):
        """Test commit verification for signed commit."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {
            "repository": {
                "object": {
                    "signature": {
                        "isValid": True,
                        "signer": {"login": "signer-user"},
                    },
                    "additions": 100,
                    "deletions": 50,
                    "changedFiles": 5,
                    "message": "Add new feature",
                    "author": {"name": "Developer", "email": "dev@example.com"},
                }
            },
            "rateLimit": {"remaining": 4500, "resetAt": "2025-01-01T12:00:00Z"},
        }

        # Mock patch fetching and entropy
        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            with patch.object(
                client, "_fetch_commit_patch_rest", new_callable=AsyncMock, return_value="diff content"
            ):
                verification = await client.get_commit_verification("owner", "repo", "abc123")

        assert verification is not None
        assert verification.repository == "owner/repo"
        assert verification.sha == "abc123"
        assert verification.is_signed is True
        assert verification.signature_valid is True
        assert verification.signer_login == "signer-user"
        assert verification.additions == 100
        assert verification.deletions == 50
        assert verification.changed_files == 5
        assert verification.commit_entropy is not None  # Entropy was calculated

    @pytest.mark.asyncio
    async def test_get_commit_verification_unsigned_commit(self):
        """Test commit verification for unsigned commit."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {
            "repository": {
                "object": {
                    "signature": None,  # No signature
                    "additions": 10,
                    "deletions": 5,
                    "changedFiles": 2,
                    "message": "Fix bug",
                    "author": {"name": "Developer", "email": "dev@example.com"},
                }
            },
            "rateLimit": {"remaining": 4500, "resetAt": "2025-01-01T12:00:00Z"},
        }

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            with patch.object(
                client, "_fetch_commit_patch_rest", new_callable=AsyncMock, return_value="small diff"
            ):
                verification = await client.get_commit_verification("owner", "repo", "def456")

        assert verification is not None
        assert verification.is_signed is False
        assert verification.signature_valid is False
        assert verification.signer_login is None

    @pytest.mark.asyncio
    async def test_get_commit_verification_high_entropy(self):
        """Test commit verification detects high entropy (obfuscated code)."""
        client = GitHubGraphQLClient("ghp_test_token")

        # Create high entropy content (random-looking)
        high_entropy_patch = "xK9vL2mN8qR5tY7pW3zA1bC4dE6fG0hJ" * 100

        mock_result = {
            "repository": {
                "object": {
                    "signature": None,
                    "additions": 500,
                    "deletions": 0,
                    "changedFiles": 1,
                    "message": "Update file",
                    "author": {"name": "Dev", "email": "dev@test.com"},
                }
            },
            "rateLimit": {"remaining": 4500, "resetAt": "2025-01-01T12:00:00Z"},
        }

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            with patch.object(
                client, "_fetch_commit_patch_rest", new_callable=AsyncMock, return_value=high_entropy_patch
            ):
                verification = await client.get_commit_verification("owner", "repo", "xyz789")

        assert verification is not None
        assert verification.commit_entropy is not None
        # High entropy content should have entropy > 4.0
        assert verification.commit_entropy > 4.0

    @pytest.mark.asyncio
    async def test_get_commit_verification_patch_fetch_error(self):
        """Test commit verification handles patch fetch errors gracefully."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {
            "repository": {
                "object": {
                    "signature": None,
                    "additions": 10,
                    "deletions": 5,
                    "changedFiles": 1,
                    "message": "Update",
                    "author": {"name": "Dev", "email": "dev@test.com"},
                }
            },
            "rateLimit": {"remaining": 4500, "resetAt": "2025-01-01T12:00:00Z"},
        }

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            with patch.object(
                client, "_fetch_commit_patch_rest", new_callable=AsyncMock, side_effect=Exception("Timeout")
            ):
                verification = await client.get_commit_verification("owner", "repo", "error123")

        assert verification is not None
        assert verification.commit_entropy is None  # Entropy calculation failed gracefully

    @pytest.mark.asyncio
    async def test_get_commit_verification_no_commit_found(self):
        """Test commit verification when commit is not found."""
        client = GitHubGraphQLClient("ghp_test_token")

        mock_result = {"repository": {"object": None}}

        with patch.object(
            client, "_execute_query", new_callable=AsyncMock, return_value=mock_result
        ):
            verification = await client.get_commit_verification("owner", "repo", "notfound")

        assert verification is None
