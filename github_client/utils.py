"""Utility functions for GitHub client."""

import datetime

from github_client.models import Event


def extract_timestamp(event: Event) -> float:
    """Extract Unix timestamp from Event object.

    Args:
        event: GitHub Event object

    Returns:
        Unix timestamp (seconds since epoch)

    Example:
        >>> event = Event(...)
        >>> timestamp = extract_timestamp(event)
        >>> print(timestamp)
        1699564800.0
    """
    if isinstance(event.created_at, datetime.datetime):
        return event.created_at.timestamp()

    # Fallback for string timestamps
    return datetime.datetime.fromisoformat(
        event.created_at.replace("Z", "+00:00")
    ).timestamp()
