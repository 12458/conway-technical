"""GitHub GraphQL API client using gql library."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.exceptions import TransportQueryError

from github_client.enriched_models import (
    ActorProfile,
    CommitVerification,
    RepositoryContext,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


class GitHubGraphQLClient:
    """Async GitHub GraphQL API client with rate limiting and error handling."""

    def __init__(self, token: str, base_url: str = "https://api.github.com/graphql"):
        """Initialize GraphQL client.

        Args:
            token: GitHub personal access token (needs repo, read:org, read:user scopes)
            base_url: GraphQL API endpoint URL
        """
        self.token = token
        self.base_url = base_url
        self._client: Client | None = None
        self._rate_limit_remaining = 5000
        self._rate_limit_reset_at: datetime | None = None
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self):
        """Establish connection to GraphQL API."""
        transport = AIOHTTPTransport(
            url=self.base_url,
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self._client = Client(transport=transport, fetch_schema_from_transport=False)

    async def close(self):
        """Close GraphQL client connection."""
        if self._client and self._client.transport:
            await self._client.transport.close()
        self._client = None

    async def _execute_query(
        self, query: str, variables: dict[str, Any], retries: int = 3
    ) -> dict[str, Any] | None:
        """Execute GraphQL query with retry logic and rate limit handling.

        Args:
            query: GraphQL query string
            variables: Query variables
            retries: Number of retry attempts

        Returns:
            Query result dict or None on error
        """
        if not self._client:
            await self.connect()

        # Check rate limit
        if self._rate_limit_remaining < 100:
            logger.warning(
                f"Low rate limit: {self._rate_limit_remaining} remaining, "
                f"resets at {self._rate_limit_reset_at}"
            )
            # If we're very low and reset time is soon, wait
            if self._rate_limit_remaining < 10 and self._rate_limit_reset_at:
                wait_seconds = (
                    self._rate_limit_reset_at - datetime.now(timezone.utc)
                ).total_seconds()
                if 0 < wait_seconds < 300:  # Max 5 min wait
                    logger.info(f"Waiting {wait_seconds:.0f}s for rate limit reset")
                    await asyncio.sleep(wait_seconds)

        for attempt in range(retries):
            try:
                result = await self._client.execute_async(
                    gql(query), variable_values=variables
                )
                self._consecutive_errors = 0
                return result
            except TransportQueryError as e:
                logger.error(f"GraphQL query error (attempt {attempt + 1}): {e}")
                if "rate limit" in str(e).lower():
                    # Rate limit hit, back off exponentially
                    wait = 2 ** attempt * 5
                    logger.warning(f"Rate limited, waiting {wait}s before retry")
                    await asyncio.sleep(wait)
                elif attempt < retries - 1:
                    # Other errors, exponential backoff
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)
                else:
                    self._consecutive_errors += 1
                    if self._consecutive_errors >= self._max_consecutive_errors:
                        logger.critical(
                            f"Too many consecutive errors ({self._consecutive_errors})"
                        )
                    return None
            except Exception as e:
                logger.error(f"Unexpected error in GraphQL query: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None

        return None

    async def get_actor_profile(self, login: str) -> ActorProfile | None:
        """Get enriched actor profile from GraphQL.

        Args:
            login: GitHub username

        Returns:
            ActorProfile or None on error
        """
        # Skip bot accounts - they don't have user profiles
        if login.endswith("[bot]") or login.endswith("-bot"):
            logger.debug(f"Skipping bot account: {login}")
            return None

        query = """
        query GetActorProfile($login: String!) {
          user(login: $login) {
            createdAt
            followers { totalCount }
            following { totalCount }
            repositories { totalCount }
            contributionsCollection {
              totalCommitContributions
              totalPullRequestContributions
              totalIssueContributions
            }
            organizations(first: 10) {
              nodes {
                login
              }
            }
            isSiteAdmin
            company
            location
            bio
          }
          rateLimit {
            remaining
            resetAt
          }
        }
        """

        result = await self._execute_query(query, {"login": login})
        if not result or "user" not in result or not result["user"]:
            logger.debug(f"No user data found for login: {login}")
            return None

        # Update rate limit info
        if "rateLimit" in result:
            self._rate_limit_remaining = result["rateLimit"]["remaining"]
            self._rate_limit_reset_at = datetime.fromisoformat(
                result["rateLimit"]["resetAt"].replace("Z", "+00:00")
            )

        user = result["user"]
        created_at = datetime.fromisoformat(user["createdAt"].replace("Z", "+00:00"))
        account_age_days = (datetime.now(timezone.utc) - created_at).days

        return ActorProfile(
            login=login,
            account_created_at=created_at,
            account_age_days=account_age_days,
            follower_count=user["followers"]["totalCount"],
            following_count=user["following"]["totalCount"],
            repository_count=user["repositories"]["totalCount"],
            total_commit_contributions=user["contributionsCollection"][
                "totalCommitContributions"
            ],
            total_pr_contributions=user["contributionsCollection"][
                "totalPullRequestContributions"
            ],
            total_issue_contributions=user["contributionsCollection"][
                "totalIssueContributions"
            ],
            organizations=[org["login"] for org in user["organizations"]["nodes"]],
            is_site_admin=user.get("isSiteAdmin"),
            company=user.get("company"),
            location=user.get("location"),
            bio=user.get("bio"),
        )

    async def get_repository_context(
        self, owner: str, name: str
    ) -> RepositoryContext | None:
        """Get enriched repository context from GraphQL.

        Args:
            owner: Repository owner
            name: Repository name

        Returns:
            RepositoryContext or None on error
        """
        query = """
        query GetRepoContext($owner: String!, $name: String!) {
          repository(owner: $owner, name: $name) {
            stargazerCount
            forkCount
            watchers { totalCount }
            primaryLanguage { name }
            defaultBranchRef {
              name
            }
            securityPolicyUrl
            isFork
            isArchived
            repositoryTopics(first: 10) {
              nodes {
                topic { name }
              }
            }
            licenseInfo { name }
          }
          rateLimit {
            remaining
            resetAt
          }
        }
        """

        result = await self._execute_query(query, {"owner": owner, "name": name})
        if not result or "repository" not in result or not result["repository"]:
            logger.debug(f"No repository data found for: {owner}/{name}")
            return None

        # Update rate limit info
        if "rateLimit" in result:
            self._rate_limit_remaining = result["rateLimit"]["remaining"]
            self._rate_limit_reset_at = datetime.fromisoformat(
                result["rateLimit"]["resetAt"].replace("Z", "+00:00")
            )

        repo = result["repository"]
        default_branch_ref = repo.get("defaultBranchRef")

        return RepositoryContext(
            owner=owner,
            name=name,
            full_name=f"{owner}/{name}",
            stargazer_count=repo["stargazerCount"],
            fork_count=repo["forkCount"],
            watcher_count=repo["watchers"]["totalCount"],
            primary_language=(
                repo["primaryLanguage"]["name"] if repo.get("primaryLanguage") else None
            ),
            default_branch=(
                default_branch_ref["name"] if default_branch_ref else None
            ),
            has_security_policy=repo.get("securityPolicyUrl") is not None,
            is_fork=repo["isFork"],
            is_archived=repo["isArchived"],
            topics=[
                node["topic"]["name"] for node in repo["repositoryTopics"]["nodes"]
            ],
            license_name=repo["licenseInfo"]["name"] if repo.get("licenseInfo") else None,
        )

    async def get_workflow_status(
        self, owner: str, name: str, commit_sha: str
    ) -> WorkflowStatus | None:
        """Get workflow/CI status for a commit from GraphQL.

        Args:
            owner: Repository owner
            name: Repository name
            commit_sha: Commit SHA

        Returns:
            WorkflowStatus or None on error
        """
        query = """
        query GetWorkflowStatus($owner: String!, $name: String!, $sha: GitObjectID!) {
          repository(owner: $owner, name: $name) {
            object(oid: $sha) {
              ... on Commit {
                checkSuites(first: 10) {
                  nodes {
                    conclusion
                    status
                    checkRuns(first: 20) {
                      nodes {
                        name
                        conclusion
                        status
                      }
                    }
                  }
                }
              }
            }
          }
          rateLimit {
            remaining
            resetAt
          }
        }
        """

        result = await self._execute_query(
            query, {"owner": owner, "name": name, "sha": commit_sha}
        )
        if not result or "repository" not in result or not result["repository"]:
            logger.debug(f"No workflow data found for: {owner}/{name}@{commit_sha}")
            return None

        # Update rate limit info
        if "rateLimit" in result:
            self._rate_limit_remaining = result["rateLimit"]["remaining"]
            self._rate_limit_reset_at = datetime.fromisoformat(
                result["rateLimit"]["resetAt"].replace("Z", "+00:00")
            )

        commit_obj = result["repository"].get("object")
        if not commit_obj or "checkSuites" not in commit_obj:
            # No check suites for this commit
            return WorkflowStatus(
                repository=f"{owner}/{name}",
                commit_sha=commit_sha,
                total_check_suites=0,
                successful_suites=0,
                failed_suites=0,
                pending_suites=0,
                check_runs=[],
                overall_conclusion=None,
            )

        check_suites = commit_obj["checkSuites"]["nodes"]
        successful = sum(1 for suite in check_suites if suite.get("conclusion") == "SUCCESS")
        failed = sum(1 for suite in check_suites if suite.get("conclusion") == "FAILURE")
        pending = sum(
            1
            for suite in check_suites
            if suite.get("status") in ["QUEUED", "IN_PROGRESS"]
        )

        # Collect all check runs
        check_runs = []
        for suite in check_suites:
            for run in suite.get("checkRuns", {}).get("nodes", []):
                check_runs.append(
                    {
                        "name": run.get("name"),
                        "conclusion": run.get("conclusion"),
                        "status": run.get("status"),
                    }
                )

        # Determine overall conclusion
        if failed > 0:
            overall = "FAILURE"
        elif pending > 0:
            overall = "PENDING"
        elif successful > 0:
            overall = "SUCCESS"
        else:
            overall = None

        return WorkflowStatus(
            repository=f"{owner}/{name}",
            commit_sha=commit_sha,
            total_check_suites=len(check_suites),
            successful_suites=successful,
            failed_suites=failed,
            pending_suites=pending,
            check_runs=check_runs,
            overall_conclusion=overall,
        )

    async def get_commit_verification(
        self, owner: str, name: str, sha: str
    ) -> CommitVerification | None:
        """Get commit verification and stats from GraphQL.

        Args:
            owner: Repository owner
            name: Repository name
            sha: Commit SHA

        Returns:
            CommitVerification or None on error
        """
        query = """
        query GetCommitVerification($owner: String!, $name: String!, $sha: GitObjectID!) {
          repository(owner: $owner, name: $name) {
            object(oid: $sha) {
              ... on Commit {
                signature {
                  isValid
                  signer {
                    login
                  }
                }
                additions
                deletions
                changedFiles
                message
                author {
                  name
                  email
                }
              }
            }
          }
          rateLimit {
            remaining
            resetAt
          }
        }
        """

        result = await self._execute_query(
            query, {"owner": owner, "name": name, "sha": sha}
        )
        if not result or "repository" not in result or not result["repository"]:
            logger.debug(f"No commit data found for: {owner}/{name}@{sha}")
            return None

        # Update rate limit info
        if "rateLimit" in result:
            self._rate_limit_remaining = result["rateLimit"]["remaining"]
            self._rate_limit_reset_at = datetime.fromisoformat(
                result["rateLimit"]["resetAt"].replace("Z", "+00:00")
            )

        commit_obj = result["repository"].get("object")
        if not commit_obj:
            logger.debug(f"Commit object not found: {owner}/{name}@{sha}")
            return None

        signature = commit_obj.get("signature")
        is_signed = signature is not None
        signature_valid = signature.get("isValid", False) if signature else False
        signer_login = None
        if signature and signature.get("signer"):
            signer_login = signature["signer"].get("login")

        return CommitVerification(
            repository=f"{owner}/{name}",
            sha=sha,
            is_signed=is_signed,
            signer_login=signer_login,
            signature_valid=signature_valid,
            additions=commit_obj.get("additions", 0),
            deletions=commit_obj.get("deletions", 0),
            changed_files=commit_obj.get("changedFiles", 0),
            message=commit_obj.get("message"),
            author_name=(
                commit_obj["author"].get("name") if commit_obj.get("author") else None
            ),
            author_email=(
                commit_obj["author"].get("email") if commit_obj.get("author") else None
            ),
        )

    async def batch_query(
        self, queries: list[tuple[str, dict[str, Any]]]
    ) -> list[dict[str, Any] | None]:
        """Execute multiple queries in sequence with shared rate limit tracking.

        Args:
            queries: List of (query_string, variables) tuples

        Returns:
            List of query results (None for failed queries)
        """
        results = []
        for query, variables in queries:
            result = await self._execute_query(query, variables)
            results.append(result)
            # Small delay between queries to be nice to API
            await asyncio.sleep(0.1)
        return results

    @property
    def rate_limit_info(self) -> dict[str, Any]:
        """Get current rate limit information."""
        return {
            "remaining": self._rate_limit_remaining,
            "reset_at": self._rate_limit_reset_at,
            "consecutive_errors": self._consecutive_errors,
        }
