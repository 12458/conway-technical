"""GitHub Events API client implementation."""

from typing import Any

import httpx
from pydantic import ValidationError as PydanticValidationError

from .config import settings
from .exceptions import (
    ForbiddenError,
    GitHubAPIError,
    ServiceUnavailableError,
    ValidationError,
)
from .models import Event, EventsResponse


class GitHubEventsClient:
    """
    Async client for GitHub Events API.

    Supports ETag caching and respects X-Poll-Interval headers for optimal polling.

    Usage:
        async with GitHubEventsClient() as client:
            response = await client.list_public_events()
            events = response.events
            etag = response.etag

            # Use etag for subsequent requests
            response2 = await client.list_public_events(etag=etag)
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ):
        """
        Initialize the GitHub Events API client.

        Args:
            base_url: Base URL for GitHub API (defaults to settings.github_api_base_url)
            timeout: Request timeout in seconds (defaults to settings.timeout)
            headers: Additional headers to include in requests
        """
        self.base_url = base_url or settings.github_api_base_url
        self.timeout = timeout or settings.timeout

        # Build default headers
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": settings.github_api_version,
            "User-Agent": settings.user_agent,
        }

        # Merge with user-provided headers
        if headers:
            self.headers.update(headers)

        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "GitHubEventsClient":
        """Enter async context manager."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with GitHubEventsClient() as client:' "
                "or call 'await client.__aenter__()'"
            )
        return self._client

    async def list_public_events(
        self,
        per_page: int | None = None,
        page: int = 1,
        etag: str | None = None,
    ) -> EventsResponse:
        """
        List public events from GitHub.

        Note: This API is not built to serve real-time use cases. Depending on the
        time of day, event latency can be anywhere from 30s to 6h.

        Args:
            per_page: Number of results per page (1-100, default from settings)
            page: Page number of results to fetch (default: 1)
            etag: ETag from previous request for caching (optional)

        Returns:
            EventsResponse containing:
                - events: List of Event objects
                - etag: New ETag for subsequent requests
                - poll_interval: Recommended polling interval in seconds

        Raises:
            ForbiddenError: When access is forbidden (403)
            ServiceUnavailableError: When service is unavailable (503)
            RateLimitError: When rate limit is exceeded
            GitHubAPIError: For other API errors
            ValidationError: When response validation fails

        Example:
            >>> async with GitHubEventsClient() as client:
            ...     response = await client.list_public_events(per_page=10)
            ...     for event in response.events:
            ...         print(f"{event.type} by {event.actor.login}")
            ...     # Use ETag for next request
            ...     response2 = await client.list_public_events(etag=response.etag)
            ...     if not response2.events:
            ...         print("No new events (304 Not Modified)")
        """
        client = self._ensure_client()

        # Build query parameters
        params: dict[str, Any] = {
            "page": page,
            "per_page": per_page or settings.default_per_page,
        }

        # Add ETag header if provided
        request_headers = {}
        if etag:
            # GitHub requires quotes around ETag value
            request_headers["If-None-Match"] = (
                f'"{etag}"' if not etag.startswith('"') else etag
            )

        try:
            response = await client.get(
                "/events",
                params=params,
                headers=request_headers,
            )

            # Handle different response status codes
            if response.status_code == 304:
                # Not Modified - no new events since last ETag
                return EventsResponse(
                    events=[],
                    etag=etag,
                    poll_interval=self._parse_poll_interval(response),
                )

            if response.status_code == 403:
                raise ForbiddenError(
                    message="Access forbidden. You may have exceeded the rate limit.",
                    status_code=403,
                )

            if response.status_code == 503:
                raise ServiceUnavailableError(
                    message="GitHub API service is temporarily unavailable.",
                    status_code=503,
                )

            # Raise for other HTTP errors
            response.raise_for_status()

            # Parse response
            events_data = response.json()

            # Validate and parse events
            try:
                events = [Event.model_validate(event) for event in events_data]
            except PydanticValidationError as e:
                raise ValidationError(
                    message=f"Failed to validate response: {str(e)}",
                    status_code=response.status_code,
                )

            # Extract ETag and poll interval from headers
            new_etag = self._parse_etag(response)
            poll_interval = self._parse_poll_interval(response)

            return EventsResponse(
                events=events,
                etag=new_etag,
                poll_interval=poll_interval,
            )

        except httpx.HTTPStatusError as e:
            # Handle unexpected HTTP errors
            raise GitHubAPIError(
                message=f"HTTP error occurred: {e.response.status_code} - {e.response.text}",
                status_code=e.response.status_code,
            )
        except httpx.RequestError as e:
            # Handle network/connection errors
            raise GitHubAPIError(
                message=f"Request error occurred: {str(e)}",
                status_code=None,
            )

    @staticmethod
    def _parse_etag(response: httpx.Response) -> str | None:
        """
        Parse ETag header from response.

        Args:
            response: HTTP response

        Returns:
            ETag value without quotes, or None if not present
        """
        etag = response.headers.get("ETag")
        if etag:
            # Remove quotes from ETag value
            return etag.strip('"')
        return None

    @staticmethod
    def _parse_poll_interval(response: httpx.Response) -> int | None:
        """
        Parse X-Poll-Interval header from response.

        Args:
            response: HTTP response

        Returns:
            Poll interval in seconds, or None if not present
        """
        poll_interval = response.headers.get("X-Poll-Interval")
        if poll_interval:
            try:
                return int(poll_interval)
            except ValueError:
                return None
        return None

    async def close(self) -> None:
        """Close the HTTP client. Alternative to using context manager."""
        if self._client:
            await self._client.aclose()
            self._client = None
