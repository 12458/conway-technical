#!/usr/bin/env python3
"""Example script demonstrating GH Archive data analysis.

This script shows how to:
1. Load historical GitHub events from GH Archive files
2. Process events using the Event models
3. Perform simple analysis (event counts, top actors, etc.)
4. Handle errors and display statistics

Usage:
    python examples/analyze_gharchive.py <path-to-archive.json.gz>

Example:
    # Download a sample archive first:
    # wget https://data.gharchive.org/2015-01-01-15.json.gz

    python examples/analyze_gharchive.py data/2015-01-01-15.json.gz
"""

import sys
from collections import Counter
from pathlib import Path

from github_client import GHArchiveLoader


def analyze_archive(archive_path: str) -> None:
    """Analyze a GH Archive file and display statistics.

    Args:
        archive_path: Path to the .json.gz archive file
    """
    print(f"Analyzing GH Archive: {archive_path}")
    print("=" * 70)

    # Create loader
    loader = GHArchiveLoader(archive_path, strict=False)

    # Display file information
    file_info = loader.get_file_info()
    print("\nFile Information:")
    print(f"  Name: {file_info['file_name']}")
    print(f"  Size: {file_info['file_size_mb']} MB ({file_info['file_size_bytes']:,} bytes)")

    # Initialize counters
    event_type_counter = Counter()
    actor_counter = Counter()
    repo_counter = Counter()
    org_counter = Counter()

    # Process events
    print("\nProcessing events...")
    for event in loader.iter_events():
        # Count event types
        event_type_counter[event.type] += 1

        # Count actors (users)
        actor_counter[event.actor.login] += 1

        # Count repositories
        repo_counter[event.repo.name] += 1

        # Count organizations (if present)
        if event.org:
            org_counter[event.org.login] += 1

    # Display statistics
    stats = loader.get_stats()
    print("\nProcessing Statistics:")
    print(f"  Total events processed: {stats.total_events:,}")
    print(f"  Valid events: {stats.valid_events:,}")
    print(f"  Invalid events: {stats.invalid_events:,}")
    print(f"  Skipped events: {stats.skipped_events:,}")

    if stats.error_types:
        print("\n  Error breakdown:")
        for error_type, count in stats.error_types.items():
            print(f"    {error_type}: {count:,}")

    # Display event type distribution
    print("\nEvent Type Distribution:")
    print(f"  Total unique event types: {len(event_type_counter)}")
    for event_type, count in event_type_counter.most_common(10):
        percentage = (count / stats.valid_events * 100) if stats.valid_events > 0 else 0
        print(f"  {event_type:30s} {count:6,} ({percentage:5.1f}%)")

    # Display top actors
    print("\nTop 10 Most Active Users:")
    for actor, count in actor_counter.most_common(10):
        print(f"  {actor:30s} {count:6,} events")

    # Display top repositories
    print("\nTop 10 Most Active Repositories:")
    for repo, count in repo_counter.most_common(10):
        print(f"  {repo:50s} {count:6,} events")

    # Display top organizations (if any)
    if org_counter:
        print("\nTop 10 Most Active Organizations:")
        for org, count in org_counter.most_common(10):
            print(f"  {org:30s} {count:6,} events")

    print("\n" + "=" * 70)
    print("Analysis complete!")


def analyze_specific_event_types(archive_path: str) -> None:
    """Demonstrate analyzing specific event types.

    Args:
        archive_path: Path to the .json.gz archive file
    """
    print("\n\nAnalyzing Specific Event Types")
    print("=" * 70)

    loader = GHArchiveLoader(archive_path, strict=False)

    # Track specific event types
    push_events = []
    watch_events = []
    issue_events = []
    pr_events = []

    for event in loader.iter_events():
        if event.type == "PushEvent":
            push_events.append(event)
        elif event.type == "WatchEvent":
            watch_events.append(event)
        elif event.type == "IssuesEvent":
            issue_events.append(event)
        elif event.type == "PullRequestEvent":
            pr_events.append(event)

    # Analyze push events
    if push_events:
        print(f"\nPush Events: {len(push_events)}")
        branches = Counter()
        for event in push_events:
            # Access payload data
            if "ref" in event.payload:
                ref = event.payload["ref"]
                # Extract branch name from refs/heads/branch-name
                if ref.startswith("refs/heads/"):
                    branch = ref.replace("refs/heads/", "")
                    branches[branch] += 1

        if branches:
            print("  Top branches:")
            for branch, count in branches.most_common(5):
                print(f"    {branch:30s} {count:6,} pushes")

    # Analyze watch events (stars)
    if watch_events:
        print(f"\nWatch Events (Stars): {len(watch_events)}")
        starred_repos = Counter(event.repo.name for event in watch_events)
        print("  Most starred repositories:")
        for repo, count in starred_repos.most_common(5):
            print(f"    {repo:50s} {count:6,} stars")

    # Analyze issue events
    if issue_events:
        print(f"\nIssue Events: {len(issue_events)}")
        actions = Counter()
        for event in issue_events:
            if "action" in event.payload:
                actions[event.payload["action"]] += 1
        if actions:
            print("  Actions breakdown:")
            for action, count in actions.most_common():
                print(f"    {action:20s} {count:6,}")

    # Analyze PR events
    if pr_events:
        print(f"\nPull Request Events: {len(pr_events)}")
        actions = Counter()
        for event in pr_events:
            if "action" in event.payload:
                actions[event.payload["action"]] += 1
        if actions:
            print("  Actions breakdown:")
            for action, count in actions.most_common():
                print(f"    {action:20s} {count:6,}")


def main() -> None:
    """Main entry point for the analysis script."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_gharchive.py <archive-file.json.gz>")
        print()
        print("Example:")
        print("  # First download an archive:")
        print("  wget https://data.gharchive.org/2015-01-01-15.json.gz")
        print()
        print("  # Then analyze it:")
        print("  python examples/analyze_gharchive.py 2015-01-01-15.json.gz")
        sys.exit(1)

    archive_path = sys.argv[1]

    # Check if file exists
    if not Path(archive_path).exists():
        print(f"Error: File not found: {archive_path}")
        sys.exit(1)

    try:
        # Run basic analysis
        analyze_archive(archive_path)

        # Run detailed event type analysis
        analyze_specific_event_types(archive_path)

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
