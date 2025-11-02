"""Unit tests for custom exceptions."""

import pytest

from github_client.exceptions import (
    ForbiddenError,
    GitHubAPIError,
    RateLimitError,
    ServiceUnavailableError,
    ValidationError,
)


@pytest.mark.unit
class TestGitHubAPIError:
    """Tests for base GitHubAPIError exception."""

    def test_exception_with_message_only(self):
        """Test GitHubAPIError with message only."""
        error = GitHubAPIError("Something went wrong")

        assert error.message == "Something went wrong"
        assert error.status_code is None
        assert str(error) == "Something went wrong"

    def test_exception_with_status_code(self):
        """Test GitHubAPIError with message and status code."""
        error = GitHubAPIError("API error", status_code=500)

        assert error.message == "API error"
        assert error.status_code == 500

    def test_exception_is_instance_of_exception(self):
        """Test GitHubAPIError is instance of Exception."""
        error = GitHubAPIError("Error")
        assert isinstance(error, Exception)

    def test_exception_can_be_raised(self):
        """Test GitHubAPIError can be raised and caught."""
        with pytest.raises(GitHubAPIError) as exc_info:
            raise GitHubAPIError("Test error", status_code=400)

        assert exc_info.value.message == "Test error"
        assert exc_info.value.status_code == 400


@pytest.mark.unit
class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_rate_limit_error_inherits_from_base(self):
        """Test RateLimitError inherits from GitHubAPIError."""
        error = RateLimitError("Rate limit exceeded", status_code=429)

        assert isinstance(error, GitHubAPIError)
        assert error.message == "Rate limit exceeded"
        assert error.status_code == 429

    def test_rate_limit_error_can_be_caught_as_base(self):
        """Test RateLimitError can be caught as GitHubAPIError."""
        with pytest.raises(GitHubAPIError):
            raise RateLimitError("Too many requests")


@pytest.mark.unit
class TestForbiddenError:
    """Tests for ForbiddenError exception."""

    def test_forbidden_error(self):
        """Test ForbiddenError exception."""
        error = ForbiddenError("Access forbidden", status_code=403)

        assert isinstance(error, GitHubAPIError)
        assert error.message == "Access forbidden"
        assert error.status_code == 403

    def test_forbidden_error_inheritance(self):
        """Test ForbiddenError can be caught as GitHubAPIError."""
        with pytest.raises(GitHubAPIError):
            raise ForbiddenError("Forbidden")


@pytest.mark.unit
class TestServiceUnavailableError:
    """Tests for ServiceUnavailableError exception."""

    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError exception."""
        error = ServiceUnavailableError("Service down", status_code=503)

        assert isinstance(error, GitHubAPIError)
        assert error.message == "Service down"
        assert error.status_code == 503

    def test_service_unavailable_error_inheritance(self):
        """Test ServiceUnavailableError can be caught as GitHubAPIError."""
        with pytest.raises(GitHubAPIError):
            raise ServiceUnavailableError("Service unavailable")


@pytest.mark.unit
class TestValidationError:
    """Tests for ValidationError exception."""

    def test_validation_error(self):
        """Test ValidationError exception."""
        error = ValidationError("Invalid response", status_code=200)

        assert isinstance(error, GitHubAPIError)
        assert error.message == "Invalid response"
        assert error.status_code == 200

    def test_validation_error_inheritance(self):
        """Test ValidationError can be caught as GitHubAPIError."""
        with pytest.raises(GitHubAPIError):
            raise ValidationError("Validation failed")


@pytest.mark.unit
class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """Test all custom exceptions inherit from GitHubAPIError."""
        exceptions = [
            RateLimitError("test"),
            ForbiddenError("test"),
            ServiceUnavailableError("test"),
            ValidationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, GitHubAPIError)
            assert isinstance(exc, Exception)

    def test_catch_specific_exception(self):
        """Test catching specific exception type."""
        with pytest.raises(ForbiddenError):
            raise ForbiddenError("Forbidden")

        # Should not catch other exception types
        with pytest.raises(ForbiddenError):
            try:
                raise ForbiddenError("Forbidden")
            except RateLimitError:
                pytest.fail("Should not catch RateLimitError")
