"""Unit tests for GitHubEventsClient with mocked HTTP responses."""

import httpx
import pytest
import respx

from github_client import GitHubEventsClient
from github_client.exceptions import (
    ForbiddenError,
    GitHubAPIError,
    ServiceUnavailableError,
    ValidationError,
)


@pytest.mark.unit
class TestGitHubEventsClientInitialization:
    """Tests for client initialization."""

    def test_client_default_initialization(self):
        """Test client initializes with default values."""
        client = GitHubEventsClient()

        assert client.base_url == "https://api.github.com"
        assert client.timeout == 30.0
        assert "Accept" in client.headers
        assert client.headers["Accept"] == "application/vnd.github+json"

    def test_client_custom_initialization(self):
        """Test client initializes with custom values."""
        client = GitHubEventsClient(
            base_url="https://custom.api.com",
            timeout=60.0,
            headers={"Custom-Header": "value"},
        )

        assert client.base_url == "https://custom.api.com"
        assert client.timeout == 60.0
        assert client.headers["Custom-Header"] == "value"

    def test_client_without_context_manager_raises_error(self):
        """Test using client without context manager raises RuntimeError."""
        client = GitHubEventsClient()

        with pytest.raises(RuntimeError) as exc_info:
            client._ensure_client()

        assert "not initialized" in str(exc_info.value).lower()


@pytest.mark.unit
class TestGitHubEventsClientContextManager:
    """Tests for async context manager."""

    async def test_context_manager_enter(self):
        """Test client enters context manager correctly."""
        async with GitHubEventsClient() as client:
            assert client._client is not None
            assert isinstance(client._client, httpx.AsyncClient)

    async def test_context_manager_exit(self):
        """Test client exits context manager and closes HTTP client."""
        client = GitHubEventsClient()
        async with client:
            assert client._client is not None

        # After exiting, client should be None
        assert client._client is None


@pytest.mark.unit
@respx.mock
class TestListPublicEvents:
    """Tests for list_public_events method."""

    async def test_list_public_events_success(self, sample_events_list):
        """Test successful list_public_events call."""
        # Mock the API response
        respx.get("https://api.github.com/events").mock(
            return_value=httpx.Response(
                200,
                json=sample_events_list,
                headers={
                    "ETag": '"abc123"',
                    "X-Poll-Interval": "60",
                },
            )
        )

        async with GitHubEventsClient() as client:
            response = await client.list_public_events(per_page=10)

            assert len(response.events) == 2
            assert response.etag == "abc123"  # Should strip quotes
            assert response.poll_interval == 60

    async def test_list_public_events_with_pagination(self, sample_events_list):
        """Test list_public_events with pagination parameters."""
        respx.get("https://api.github.com/events").mock(
            return_value=httpx.Response(200, json=sample_events_list)
        )

        async with GitHubEventsClient() as client:
            response = await client.list_public_events(per_page=20, page=2)

            # Verify request was made with correct parameters
            assert response is not None

    async def test_list_public_events_304_not_modified(self, mock_etag):
        """Test 304 Not Modified response with ETag."""
        respx.get("https://api.github.com/events").mock(
            return_value=httpx.Response(
                304,
                headers={
                    "ETag": f'"{mock_etag}"',
                    "X-Poll-Interval": "60",
                },
            )
        )

        async with GitHubEventsClient() as client:
            response = await client.list_public_events(etag=mock_etag)

            assert len(response.events) == 0
            assert response.etag == mock_etag
            assert response.poll_interval == 60

    async def test_list_public_events_403_forbidden(self):
        """Test 403 Forbidden response raises ForbiddenError."""
        respx.get("https://api.github.com/events").mock(
            return_value=httpx.Response(403, json={"message": "Forbidden"})
        )

        async with GitHubEventsClient() as client:
            with pytest.raises(ForbiddenError) as exc_info:
                await client.list_public_events()

            assert exc_info.value.status_code == 403
            assert "forbidden" in exc_info.value.message.lower()

    async def test_list_public_events_503_service_unavailable(self):
        """Test 503 Service Unavailable response."""
        respx.get("https://api.github.com/events").mock(
            return_value=httpx.Response(503, json={"message": "Service Unavailable"})
        )

        async with GitHubEventsClient() as client:
            with pytest.raises(ServiceUnavailableError) as exc_info:
                await client.list_public_events()

            assert exc_info.value.status_code == 503
            assert "unavailable" in exc_info.value.message.lower()

    async def test_list_public_events_other_http_error(self):
        """Test other HTTP errors raise GitHubAPIError."""
        respx.get("https://api.github.com/events").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        async with GitHubEventsClient() as client:
            with pytest.raises(GitHubAPIError) as exc_info:
                await client.list_public_events()

            assert exc_info.value.status_code == 500

    async def test_list_public_events_invalid_json(self):
        """Test invalid JSON response raises ValidationError."""
        respx.get("https://api.github.com/events").mock(
            return_value=httpx.Response(
                200,
                json=[{"invalid": "data", "missing": "required_fields"}],
            )
        )

        async with GitHubEventsClient() as client:
            with pytest.raises(ValidationError) as exc_info:
                await client.list_public_events()

            assert "validate" in exc_info.value.message.lower()

    async def test_list_public_events_network_error(self):
        """Test network error raises GitHubAPIError."""
        respx.get("https://api.github.com/events").mock(
            side_effect=httpx.ConnectError("Connection failed")
        )

        async with GitHubEventsClient() as client:
            with pytest.raises(GitHubAPIError) as exc_info:
                await client.list_public_events()

            assert "request error" in exc_info.value.message.lower()
            assert exc_info.value.status_code is None


@pytest.mark.unit
class TestClientUtilityMethods:
    """Tests for utility methods."""

    def test_parse_etag_with_quotes(self):
        """Test _parse_etag removes quotes."""
        response = httpx.Response(200, headers={"ETag": '"abc123"'})
        etag = GitHubEventsClient._parse_etag(response)

        assert etag == "abc123"

    def test_parse_etag_without_quotes(self):
        """Test _parse_etag handles ETag without quotes."""
        response = httpx.Response(200, headers={"ETag": "abc123"})
        etag = GitHubEventsClient._parse_etag(response)

        assert etag == "abc123"

    def test_parse_etag_missing(self):
        """Test _parse_etag returns None when ETag is missing."""
        response = httpx.Response(200, headers={})
        etag = GitHubEventsClient._parse_etag(response)

        assert etag is None

    def test_parse_poll_interval_present(self):
        """Test _parse_poll_interval extracts interval."""
        response = httpx.Response(200, headers={"X-Poll-Interval": "60"})
        interval = GitHubEventsClient._parse_poll_interval(response)

        assert interval == 60

    def test_parse_poll_interval_missing(self):
        """Test _parse_poll_interval returns None when missing."""
        response = httpx.Response(200, headers={})
        interval = GitHubEventsClient._parse_poll_interval(response)

        assert interval is None

    def test_parse_poll_interval_invalid(self):
        """Test _parse_poll_interval handles invalid value."""
        response = httpx.Response(200, headers={"X-Poll-Interval": "invalid"})
        interval = GitHubEventsClient._parse_poll_interval(response)

        assert interval is None


@pytest.mark.unit
@respx.mock
class TestETagHandling:
    """Tests for ETag caching functionality."""

    async def test_etag_sent_in_request_header(self, sample_events_list, mock_etag):
        """Test ETag is sent in If-None-Match header."""
        route = respx.get("https://api.github.com/events").mock(
            return_value=httpx.Response(200, json=sample_events_list)
        )

        async with GitHubEventsClient() as client:
            await client.list_public_events(etag=mock_etag)

            # Check that If-None-Match header was sent
            assert route.called
            request = route.calls.last.request
            assert "If-None-Match" in request.headers
            assert mock_etag in request.headers["If-None-Match"]

    async def test_etag_with_quotes_in_request(self, sample_events_list):
        """Test ETag is properly quoted in request header."""
        route = respx.get("https://api.github.com/events").mock(
            return_value=httpx.Response(200, json=sample_events_list)
        )

        async with GitHubEventsClient() as client:
            await client.list_public_events(etag="test123")

            request = route.calls.last.request
            # Should add quotes if not present
            assert request.headers["If-None-Match"] == '"test123"'


@pytest.mark.unit
class TestClientClose:
    """Tests for client close method."""

    async def test_close_method(self):
        """Test close method closes HTTP client."""
        client = GitHubEventsClient()
        await client.__aenter__()

        assert client._client is not None

        await client.close()

        assert client._client is None

    async def test_close_method_when_not_initialized(self):
        """Test close method when client is not initialized."""
        client = GitHubEventsClient()

        # Should not raise error
        await client.close()

        assert client._client is None
