#!/usr/bin/env python3
"""Detect malicious GitHub events using RRCF with real Pydantic model features.

This script uses ONLY features that actually exist in your Event models.
It demonstrates practical anomaly detection for security monitoring.

Usage:
    python examples/rrcf_malicious_detection.py <archive.json.gz> [threshold]

Example:
    # Download sample data
    wget https://data.gharchive.org/2015-01-01-15.json.gz

    # Run detection
    python examples/rrcf_malicious_detection.py 2015-01-01-15.json.gz
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import newrrcf as rrcf

from github_client import GHArchiveLoader
from github_client.feature_extractor import (
    GitHubFeatureExtractor,
    get_suspicious_patterns,
)


class RRCFMaliciousEventDetector:
    """RRCF-based detector for malicious GitHub events."""

    def __init__(
        self, num_trees: int = 40, tree_size: int = 256, filter_bots: bool = True
    ):
        """Initialize the detector.

        Args:
            num_trees: Number of trees in forest (more = better accuracy, slower)
            tree_size: Max points per tree (larger = more memory, better context)
            filter_bots: Whether to filter out known bot accounts (default: True)
        """
        self.num_trees = num_trees
        self.tree_size = tree_size

        # Initialize RRCF forest
        self.forest = {i: rrcf.RCTree() for i in range(num_trees)}

        # Feature extractor
        self.feature_extractor = GitHubFeatureExtractor(filter_bots=filter_bots)

        # Tracking
        self.index = 0
        self.all_scores = []
        self.bots_filtered = 0

    def detect(self, event) -> tuple[float, np.ndarray] | None:
        """Detect if event is anomalous.

        Args:
            event: GitHub Event object

        Returns:
            Tuple of (anomaly_score, feature_vector), or None if bot was filtered
        """
        # Extract features using actual model fields
        features = self.feature_extractor.extract_features(event)

        # Skip if bot was filtered
        if features is None:
            self.bots_filtered += 1
            return None

        # Calculate anomaly score (CoDisp)
        avg_codisp = 0.0

        for tree_idx, tree in self.forest.items():
            # Maintain sliding window
            if len(tree.leaves) >= self.tree_size:
                oldest = self.index - self.tree_size
                if oldest >= 0:
                    try:
                        tree.forget_point(oldest)
                    except KeyError:
                        pass

            # Insert point
            tree.insert_point(features, index=self.index)

            # Calculate displacement
            if len(tree.leaves) > 1:
                avg_codisp += tree.codisp(self.index)

        avg_codisp /= self.num_trees
        self.all_scores.append(avg_codisp)
        self.index += 1

        return avg_codisp, features

    def get_adaptive_threshold(self, percentile: float = 99.0) -> float:
        """Get adaptive threshold based on score history.

        Args:
            percentile: Percentile for threshold (e.g., 99.0)

        Returns:
            Threshold value
        """
        if len(self.all_scores) < 20:
            return 2.0  # Default threshold
        return np.percentile(self.all_scores, percentile)


def analyze_for_malicious_activity(
    archive_path: str,
    percentile_threshold: float = 99.0,
    top_n: int = 20,
    filter_bots: bool = True,
) -> None:
    """Analyze archive for malicious events.

    Args:
        archive_path: Path to GH Archive file
        percentile_threshold: Percentile for anomaly threshold (99.0 = top 1%)
        top_n: Number of top anomalies to display
        filter_bots: Whether to filter out known bot accounts (default: True)
    """
    print(f"Analyzing: {archive_path}")
    print(f"Anomaly threshold: {percentile_threshold}th percentile")
    print(f"Bot filtering: {'enabled' if filter_bots else 'disabled'}")
    print("=" * 80)

    # Initialize
    loader = GHArchiveLoader(archive_path, strict=False)
    detector = RRCFMaliciousEventDetector(
        num_trees=40, tree_size=256, filter_bots=filter_bots
    )

    # Track anomalies
    all_events = []
    event_count = 0

    # First pass: collect scores
    print("\nPass 1: Processing events and collecting scores...")
    MAX_EVENTS = 20_000
    for event in loader.iter_events():
        event_count += 1
        result = detector.detect(event)

        # Skip bots (if filtered)
        if result is None:
            continue

        score, features = result
        all_events.append((score, event, features))

        if event_count % 1000 == 0:
            print(f"  Processed {event_count:,} events...")

        if event_count >= MAX_EVENTS:
            print(f"  Reached maximum event limit ({MAX_EVENTS:,}), stopping...")
            break

    # Calculate adaptive threshold
    threshold = detector.get_adaptive_threshold(percentile_threshold)

    # Filter anomalies
    anomalies = [(s, e, f) for s, e, f in all_events if s > threshold]

    # Calculate statistics
    processed_events = len(all_events)
    total_events = event_count

    print(f"\nProcessed {total_events:,} total events")
    if filter_bots and detector.bots_filtered > 0:
        print(
            f"  - {detector.bots_filtered:,} bot events filtered ({detector.bots_filtered / total_events * 100:.1f}%)"
        )
        print(
            f"  - {processed_events:,} non-bot events analyzed ({processed_events / total_events * 100:.1f}%)"
        )
    print(f"Adaptive threshold: {threshold:.3f}")
    print(
        f"Found {len(anomalies):,} anomalies above threshold ({len(anomalies) / processed_events * 100:.2f}%)"
    )

    # Statistics
    scores_array = np.array(detector.all_scores)
    print(f"\nScore Statistics:")
    print(f"  Mean:   {scores_array.mean():.3f}")
    print(f"  Median: {np.median(scores_array):.3f}")
    print(f"  StdDev: {scores_array.std():.3f}")
    print(f"  Min:    {scores_array.min():.3f}")
    print(f"  Max:    {scores_array.max():.3f}")
    print(f"  95th:   {np.percentile(scores_array, 95):.3f}")
    print(f"  99th:   {np.percentile(scores_array, 99):.3f}")
    print(f"  99.9th: {np.percentile(scores_array, 99.9):.3f}")

    # Display top anomalies
    if anomalies:
        print(f"\n{'=' * 80}")
        print(f"Top {top_n} Most Suspicious Events (Potential Malicious Activity)")
        print(f"{'=' * 80}")

        # Sort by score
        anomalies.sort(key=lambda x: x[0], reverse=True)

        for i, (score, event, features) in enumerate(anomalies[:top_n], 1):
            # Calculate severity
            z_score = (score - scores_array.mean()) / (scores_array.std() + 1e-10)
            if z_score > 5:
                severity = "🚨 CRITICAL"
            elif z_score > 3:
                severity = "🔴 HIGH"
            elif z_score > 2:
                severity = "🔶 MEDIUM"
            else:
                severity = "⚠️  LOW"

            print(f"\n#{i} {severity} - Score: {score:.2f} (z-score: {z_score:.1f})")
            print(f"  Event:  {event.type}")
            print(f"  Actor:  {event.actor.login} (ID: {event.actor.id})")
            print(f"  Repo:   {event.repo.name}")
            print(f"  Time:   {event.created_at}")

            if event.org:
                print(f"  Org:    {event.org.login}")

            # Show suspicious patterns
            patterns = get_suspicious_patterns(event, detector.feature_extractor)
            if patterns:
                print("  🚩 Suspicious Patterns:")
                for pattern in patterns:
                    print(f"     • {pattern}")

            # Show key payload details
            if event.type == "PushEvent":
                ref = event.payload.get("ref", "")
                print(f"  Details: push to {ref}")
            elif event.type == "MemberEvent":
                action = event.payload.get("action", "")
                print(f"  Details: Member {action}")
            elif event.type == "DeleteEvent":
                ref_type = event.payload.get("ref_type", "")
                ref = event.payload.get("ref", "")
                print(f"  Details: Deleted {ref_type} '{ref}'")

        # Pattern analysis
        print(f"\n{'=' * 80}")
        print("Malicious Pattern Analysis")
        print(f"{'=' * 80}")

        # Event types in anomalies
        from collections import Counter

        event_types = Counter(e.type for _, e, _ in anomalies)
        print("\nMost suspicious event types:")
        for event_type, count in event_types.most_common(10):
            pct = count / len(anomalies) * 100
            print(f"  {event_type:30s} {count:5,} ({pct:5.1f}%)")

        # Actors in anomalies
        actors = Counter(e.actor.login for _, e, _ in anomalies)
        print("\nMost suspicious actors:")
        for actor, count in actors.most_common(10):
            total = detector.feature_extractor.actor_event_counts[actor]
            anomaly_rate = count / total * 100
            print(
                f"  {actor:30s} {count:5,} anomalies / {total:5,} total ({anomaly_rate:5.1f}%)"
            )

        # Repos in anomalies
        repos = Counter(e.repo.name for _, e, _ in anomalies)
        print("\nMost targeted repositories:")
        for repo, count in repos.most_common(10):
            print(f"  {repo:50s} {count:5,} anomalies")

        # Security-critical events
        critical_types = ["MemberEvent", "DeleteEvent", "PublicEvent"]
        critical_anomalies = [
            (s, e, f) for s, e, f in anomalies if e.type in critical_types
        ]
        if critical_anomalies:
            print(
                f"\n⚠️  SECURITY ALERT: {len(critical_anomalies)} anomalous security-critical events!"
            )
            print("  Event type breakdown:")
            for event_type in critical_types:
                count = sum(1 for _, e, _ in critical_anomalies if e.type == event_type)
                if count > 0:
                    print(f"    {event_type:20s} {count:5,}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect malicious GitHub events using RRCF anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  wget https://data.gharchive.org/2015-01-01-15.json.gz
  python examples/rrcf_malicious_detection.py 2015-01-01-15.json.gz

  # Stricter threshold (fewer alerts)
  python examples/rrcf_malicious_detection.py 2015-01-01-15.json.gz --percentile 99.9

  # Include bots in analysis
  python examples/rrcf_malicious_detection.py 2015-01-01-15.json.gz --include-bots
        """,
    )

    parser.add_argument(
        "archive",
        help="Path to GH Archive file (.json.gz)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Anomaly threshold percentile (default: 99.0). 99.0 = top 1%%, 99.9 = top 0.1%%",
    )
    parser.add_argument(
        "--include-bots",
        action="store_true",
        help="Include bot accounts in analysis (by default, known bots are filtered)",
    )

    args = parser.parse_args()

    if not Path(args.archive).exists():
        print(f"Error: File not found: {args.archive}")
        sys.exit(1)

    if not (90.0 <= args.percentile <= 99.99):
        print("Error: Percentile must be between 90.0 and 99.99")
        sys.exit(1)

    try:
        analyze_for_malicious_activity(
            args.archive,
            percentile_threshold=args.percentile,
            filter_bots=not args.include_bots,
        )
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
