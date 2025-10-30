"""Custom exceptions for GitHub API client."""


class GitHubAPIError(Exception):
    """Base exception for GitHub API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class RateLimitError(GitHubAPIError):
    """Raised when API rate limit is exceeded."""

    pass


class ForbiddenError(GitHubAPIError):
    """Raised when access is forbidden (403)."""

    pass


class ServiceUnavailableError(GitHubAPIError):
    """Raised when the GitHub API service is unavailable (503)."""

    pass


class ValidationError(GitHubAPIError):
    """Raised when response validation fails."""

    pass
