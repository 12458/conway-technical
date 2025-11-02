"""Unit tests for GH Archive loader."""

import gzip
import json
from pathlib import Path

import pytest

from github_client.gharchive import GHArchiveLoader, LoaderStats


@pytest.mark.unit
class TestLoaderStats:
    """Tests for LoaderStats dataclass."""

    def test_loader_stats_initialization(self):
        """Test LoaderStats initializes with correct defaults."""
        stats = LoaderStats()

        assert stats.total_events == 0
        assert stats.valid_events == 0
        assert stats.invalid_events == 0
        assert stats.skipped_events == 0
        assert stats.error_types == {}

    def test_increment_error(self):
        """Test incrementing error counts."""
        stats = LoaderStats()

        stats.increment_error("validation_error")
        assert stats.error_types["validation_error"] == 1

        stats.increment_error("validation_error")
        assert stats.error_types["validation_error"] == 2

        stats.increment_error("json_decode_error")
        assert stats.error_types["json_decode_error"] == 1


@pytest.mark.unit
class TestGHArchiveLoader:
    """Tests for GHArchiveLoader class."""

    def test_loader_initialization_with_nonexistent_file(self):
        """Test loader raises FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            GHArchiveLoader("/path/to/nonexistent/file.json.gz")

        assert "not found" in str(exc_info.value).lower()

    def test_loader_initialization_with_existing_file(self, sample_gharchive_file):
        """Test loader initializes successfully with existing file."""
        loader = GHArchiveLoader(sample_gharchive_file)

        assert loader.file_path == Path(sample_gharchive_file)
        assert loader.strict is False
        assert isinstance(loader.get_stats(), LoaderStats)

    def test_loader_strict_mode(self, sample_gharchive_file):
        """Test loader can be initialized in strict mode."""
        loader = GHArchiveLoader(sample_gharchive_file, strict=True)

        assert loader.strict is True

    def test_get_file_info(self, sample_gharchive_file):
        """Test getting file information."""
        loader = GHArchiveLoader(sample_gharchive_file)
        info = loader.get_file_info()

        assert "file_name" in info
        assert "file_path" in info
        assert "file_size_bytes" in info
        assert "file_size_mb" in info
        assert info["file_size_bytes"] > 0

    def test_iter_events_with_valid_events(self, sample_gharchive_file):
        """Test iterating over valid events."""
        loader = GHArchiveLoader(sample_gharchive_file)

        events = list(loader.iter_events())

        assert len(events) > 0
        stats = loader.get_stats()
        assert stats.total_events == len(events)
        assert stats.valid_events == len(events)
        assert stats.invalid_events == 0

        # Verify first event structure
        first_event = events[0]
        assert hasattr(first_event, "id")
        assert hasattr(first_event, "type")
        assert hasattr(first_event, "actor")
        assert hasattr(first_event, "repo")

    def test_iter_events_with_limit(self, sample_gharchive_file):
        """Test limiting number of events returned."""
        loader = GHArchiveLoader(sample_gharchive_file)

        events = list(loader.iter_events(limit=2))

        assert len(events) == 2
        stats = loader.get_stats()
        assert stats.valid_events == 2

    def test_load_events(self, sample_gharchive_file):
        """Test loading all events into a list."""
        loader = GHArchiveLoader(sample_gharchive_file)

        events = loader.load_events()

        assert isinstance(events, list)
        assert len(events) > 0
        assert all(hasattr(e, "type") for e in events)

    def test_reset_stats(self, sample_gharchive_file):
        """Test resetting statistics."""
        loader = GHArchiveLoader(sample_gharchive_file)

        # Process some events
        list(loader.iter_events(limit=2))
        assert loader.get_stats().total_events > 0

        # Reset stats
        loader.reset_stats()
        stats = loader.get_stats()
        assert stats.total_events == 0
        assert stats.valid_events == 0

    def test_iter_events_with_invalid_json(self, tmp_path):
        """Test handling of invalid JSON in lenient mode."""
        # Create archive with invalid JSON
        archive_file = tmp_path / "invalid.json.gz"
        with gzip.open(archive_file, "wt", encoding="utf-8") as f:
            f.write('{"id": "1", "type": "WatchEvent"}\n')  # Valid
            f.write("invalid json here\n")  # Invalid
            f.write('{"id": "2", "type": "WatchEvent"}\n')  # Valid

        loader = GHArchiveLoader(archive_file, strict=False)
        events = list(loader.iter_events())

        # Should skip invalid line but continue
        assert len(events) < 3  # Some events may fail validation
        stats = loader.get_stats()
        assert stats.total_events == 3
        assert stats.skipped_events >= 1

    def test_iter_events_with_invalid_json_strict_mode(self, tmp_path):
        """Test that strict mode raises exception on invalid JSON."""
        # Create archive with invalid JSON
        archive_file = tmp_path / "invalid.json.gz"
        with gzip.open(archive_file, "wt", encoding="utf-8") as f:
            f.write("invalid json\n")

        loader = GHArchiveLoader(archive_file, strict=True)

        with pytest.raises(json.JSONDecodeError):
            list(loader.iter_events())

    def test_iter_events_with_validation_errors(
        self, sample_gharchive_file_with_invalid_events
    ):
        """Test handling events that fail Pydantic validation."""
        loader = GHArchiveLoader(
            sample_gharchive_file_with_invalid_events, strict=False
        )

        events = list(loader.iter_events())

        stats = loader.get_stats()
        # Some events should be valid, some invalid
        assert stats.total_events > 0
        assert stats.valid_events >= 0
        assert stats.invalid_events > 0

    def test_iter_events_skips_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        archive_file = tmp_path / "empty_lines.json.gz"
        with gzip.open(archive_file, "wt", encoding="utf-8") as f:
            f.write("\n")  # Empty line
            f.write("  \n")  # Whitespace line
            f.write("\n")  # Empty line

        loader = GHArchiveLoader(archive_file)
        events = list(loader.iter_events())

        assert len(events) == 0
        # Empty lines shouldn't count as total events
        assert loader.get_stats().total_events == 0

    def test_multiple_iterations_accumulate_stats(self, sample_gharchive_file):
        """Test that multiple iterations accumulate statistics."""
        loader = GHArchiveLoader(sample_gharchive_file)

        # First iteration
        list(loader.iter_events(limit=2))
        first_count = loader.get_stats().total_events

        # Second iteration
        list(loader.iter_events(limit=2))
        second_count = loader.get_stats().total_events

        # Stats should accumulate
        assert second_count == first_count * 2

    def test_event_types_are_preserved(self, sample_gharchive_file):
        """Test that different event types are correctly loaded."""
        loader = GHArchiveLoader(sample_gharchive_file)

        events = list(loader.iter_events())

        # Get all event types
        event_types = {event.type for event in events}

        # Should have multiple event types in the sample data
        assert len(event_types) >= 1
        # Verify common event types exist (at least one)
        common_types = {"WatchEvent", "PushEvent", "CreateEvent", "IssuesEvent"}
        assert any(t in event_types for t in common_types)

    def test_event_timestamps_are_parsed(self, sample_gharchive_file):
        """Test that event timestamps are correctly parsed."""
        loader = GHArchiveLoader(sample_gharchive_file)

        events = list(loader.iter_events(limit=1))

        assert len(events) > 0
        event = events[0]
        # Verify created_at is a datetime object
        from datetime import datetime

        assert isinstance(event.created_at, datetime)
