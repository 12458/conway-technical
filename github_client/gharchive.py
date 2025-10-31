"""GH Archive data loader for historical GitHub events.

This module provides utilities to load and process historical GitHub event data
from GH Archive (https://www.gharchive.org/). GH Archive provides historical
GitHub event data in compressed JSON format.

Example:
    >>> from github_client.gharchive import GHArchiveLoader
    >>>
    >>> loader = GHArchiveLoader("data/2015-01-01-15.json.gz")
    >>> for event in loader.iter_events():
    ...     print(f"{event.type} by {event.actor.login}")
    >>>
    >>> stats = loader.get_stats()
    >>> print(f"Processed {stats.total_events} events, {stats.valid_events} valid")
"""

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, TextIO

from pydantic import ValidationError as PydanticValidationError

from .models import Event

logger = logging.getLogger(__name__)


@dataclass
class LoaderStats:
    """Statistics about archive loading and processing.

    Attributes:
        total_events: Total number of event records encountered
        valid_events: Number of events successfully validated
        invalid_events: Number of events that failed validation
        skipped_events: Number of events skipped due to parse errors
        error_types: Dictionary mapping error types to their counts
    """

    total_events: int = 0
    valid_events: int = 0
    invalid_events: int = 0
    skipped_events: int = 0
    error_types: dict[str, int] = field(default_factory=dict)

    def increment_error(self, error_type: str) -> None:
        """Increment count for a specific error type."""
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1


class GHArchiveLoader:
    """Loader for GH Archive historical event data.

    This class handles loading, decompressing, and parsing GH Archive files.
    Archive files are in NDJSON (newline-delimited JSON) format, compressed
    with gzip.

    Events are validated using the Event Pydantic model. Invalid events can
    be skipped (lenient mode) or cause an exception (strict mode).

    Args:
        file_path: Path to the .json.gz archive file
        strict: If True, raise exception on validation errors. If False,
                skip invalid events and log warnings (default: False)

    Example:
        >>> loader = GHArchiveLoader("2015-01-01-15.json.gz")
        >>> events = list(loader.iter_events())
        >>> print(f"Loaded {len(events)} events")
        >>>
        >>> stats = loader.get_stats()
        >>> print(f"Valid: {stats.valid_events}, Invalid: {stats.invalid_events}")
    """

    def __init__(self, file_path: str | Path, strict: bool = False):
        """Initialize the GH Archive loader.

        Args:
            file_path: Path to the GH Archive .json.gz file
            strict: Whether to raise exceptions on invalid events

        Raises:
            FileNotFoundError: If the archive file doesn't exist
        """
        self.file_path = Path(file_path)
        self.strict = strict
        self._stats = LoaderStats()

        if not self.file_path.exists():
            raise FileNotFoundError(f"Archive file not found: {self.file_path}")

    def get_stats(self) -> LoaderStats:
        """Get statistics about processed events.

        Returns:
            LoaderStats object with processing statistics
        """
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = LoaderStats()

    def iter_events(self, limit: int | None = None) -> Generator[Event, None, None]:
        """Iterate over events in the archive file.

        This is a generator function that yields validated Event objects
        one at a time, enabling memory-efficient processing of large files.

        Args:
            limit: Optional limit on number of events to yield (default: None)

        Yields:
            Event: Validated Event objects from the archive

        Raises:
            ValidationError: If strict=True and an event fails validation
            JSONDecodeError: If a line cannot be parsed as JSON

        Example:
            >>> loader = GHArchiveLoader("archive.json.gz")
            >>> for event in loader.iter_events(limit=100):
            ...     if event.type == "PushEvent":
            ...         print(f"Push to {event.repo.name}")
        """
        with gzip.open(self.file_path, "rt", encoding="utf-8") as file:
            yield from self._process_file(file, limit)

    def load_events(self, limit: int | None = None) -> list[Event]:
        """Load all events from the archive into a list.

        Note: This loads all events into memory. For large files, consider
        using iter_events() instead.

        Args:
            limit: Optional limit on number of events to load (default: None)

        Returns:
            List of validated Event objects

        Example:
            >>> loader = GHArchiveLoader("small-archive.json.gz")
            >>> events = loader.load_events(limit=1000)
            >>> push_events = [e for e in events if e.type == "PushEvent"]
        """
        return list(self.iter_events(limit=limit))

    def _process_file(
        self, file: TextIO, limit: int | None = None
    ) -> Generator[Event, None, None]:
        """Process events from an open file handle.

        Args:
            file: Open text file handle
            limit: Optional limit on number of events to process

        Yields:
            Event: Validated Event objects
        """
        events_yielded = 0

        for line_num, line in enumerate(file, start=1):
            # Check limit
            if limit is not None and events_yielded >= limit:
                logger.info(f"Reached limit of {limit} events")
                break

            # Skip empty lines
            line = line.strip()
            if not line:
                continue

            self._stats.total_events += 1

            # Parse JSON
            try:
                event_data = json.loads(line)
            except json.JSONDecodeError as e:
                self._stats.skipped_events += 1
                self._stats.increment_error("json_decode_error")
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                if self.strict:
                    raise
                continue

            # Validate with Pydantic model
            try:
                event = Event.model_validate(event_data)
                self._stats.valid_events += 1
                events_yielded += 1
                yield event

            except PydanticValidationError as e:
                self._stats.invalid_events += 1
                self._stats.increment_error("validation_error")
                logger.warning(
                    f"Line {line_num}: Validation error for event {event_data.get('id', 'unknown')}: {e}"
                )
                if self.strict:
                    raise

    def get_file_info(self) -> dict[str, str | int]:
        """Get information about the archive file.

        Returns:
            Dictionary with file metadata (name, size, etc.)
        """
        stat = self.file_path.stat()
        return {
            "file_name": self.file_path.name,
            "file_path": str(self.file_path.absolute()),
            "file_size_bytes": stat.st_size,
            "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
        }
