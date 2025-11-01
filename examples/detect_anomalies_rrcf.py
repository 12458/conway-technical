#!/usr/bin/env python3
"""Detect malicious GitHub events using RRCF (Robust Random Cut Forest).

This script demonstrates how to use RRCF for anomaly detection on GitHub events.
RRCF is particularly effective at detecting outliers in streaming data.

Usage:
    python examples/detect_anomalies_rrcf.py <path-to-archive.json.gz>

Example:
    # Download sample data first:
    # wget https://data.gharchive.org/2015-01-01-15.json.gz

    python examples/detect_anomalies_rrcf.py data/2015-01-01-15.json.gz
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import newrrcf as rrcf

from github_client import GHArchiveLoader
from github_client.feature_extractor import GitHubFeatureExtractor


class RRCFAnomalyDetector:
    """RRCF-based anomaly detector for GitHub events."""

    def __init__(
        self, num_trees: int = 100, tree_size: int = 256, shingle_size: int = 1
    ):
        """Initialize the RRCF detector.

        Args:
            num_trees: Number of trees in the forest (more trees = better detection)
            tree_size: Maximum number of points in each tree
            shingle_size: Size of sliding window (1 = no shingling)
        """
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.shingle_size = shingle_size

        # Initialize forest
        self.forest = {}
        for i in range(num_trees):
            self.forest[i] = rrcf.RCTree()

        self.index = 0
        # Use improved RRCF-optimized feature extractor
        self.feature_extractor = GitHubFeatureExtractor()

    def detect_anomaly(self, event: Any) -> tuple[float, np.ndarray]:
        """Detect if an event is anomalous.

        Args:
            event: GitHub Event object

        Returns:
            Tuple of (anomaly_score, features)
            - anomaly_score: Higher values indicate more anomalous events
            - features: The feature vector used for detection
        """
        # Extract features
        features = self.feature_extractor.extract_features(event)

        # Compute anomaly score (CoDisp - Collusive Displacement)
        avg_codisp = 0.0

        # Insert point into each tree and compute displacement
        for tree_index in self.forest:
            tree = self.forest[tree_index]

            # If tree is at capacity, remove oldest point
            if len(tree.leaves) >= self.tree_size:
                # Find oldest point to remove
                oldest_index = self.index - self.tree_size
                if oldest_index >= 0:
                    try:
                        tree.forget_point(oldest_index)
                    except KeyError:
                        pass  # Point already removed

            # Insert new point
            tree.insert_point(features, index=self.index)

            # Compute collusive displacement (anomaly score)
            if len(tree.leaves) > 1:
                codisp = tree.codisp(self.index)
                avg_codisp += codisp

        # Average across all trees
        avg_codisp /= self.num_trees

        self.index += 1

        return avg_codisp, features


def analyze_archive_for_anomalies(
    archive_path: str, anomaly_threshold: float = 2.0, top_n: int = 20
) -> None:
    """Analyze a GH Archive file for anomalous events using RRCF.

    Args:
        archive_path: Path to the .json.gz archive file
        anomaly_threshold: Threshold for flagging anomalies (higher = fewer anomalies)
        top_n: Number of top anomalies to display
    """
    print(f"Analyzing GH Archive for anomalies: {archive_path}")
    print(f"Anomaly threshold: {anomaly_threshold}")
    print("=" * 80)

    # Create loader and detector
    loader = GHArchiveLoader(archive_path, strict=False)
    detector = RRCFAnomalyDetector(num_trees=100, tree_size=256)

    # Track anomalies
    anomalies = []  # List of (score, event, features)
    all_scores = []

    # Process events
    print("\nProcessing events...")
    event_count = 0
    for event in loader.iter_events():
        event_count += 1

        # Detect anomaly
        anomaly_score, features = detector.detect_anomaly(event)
        all_scores.append(anomaly_score)

        # Store if above threshold
        if anomaly_score > anomaly_threshold:
            anomalies.append((anomaly_score, event, features))

        if event_count % 1000 == 0:
            print(f"  Processed {event_count:,} events...")

    # Display statistics
    print(f"\nProcessed {event_count:,} total events")
    print(f"Found {len(anomalies):,} anomalies above threshold")

    if all_scores:
        all_scores_array = np.array(all_scores)
        print(f"\nAnomaly Score Statistics:")
        print(f"  Mean: {all_scores_array.mean():.3f}")
        print(f"  Std Dev: {all_scores_array.std():.3f}")
        print(f"  Min: {all_scores_array.min():.3f}")
        print(f"  Max: {all_scores_array.max():.3f}")
        print(f"  95th percentile: {np.percentile(all_scores_array, 95):.3f}")
        print(f"  99th percentile: {np.percentile(all_scores_array, 99):.3f}")

    # Display top anomalies
    if anomalies:
        print(f"\n{'=' * 80}")
        print(f"Top {top_n} Most Anomalous Events:")
        print(f"{'=' * 80}")

        # Sort by score descending
        anomalies.sort(key=lambda x: x[0], reverse=True)

        for i, (score, event, features) in enumerate(anomalies[:top_n], 1):
            print(f"\n#{i} - Anomaly Score: {score:.2f}")
            print(f"  Event ID: {event.id}")
            print(f"  Event Type: {event.type}")
            print(f"  Actor: {event.actor.login} (ID: {event.actor.id})")
            print(f"  Repository: {event.repo.name}")
            print(f"  Timestamp: {event.created_at}")
            print(f"  Payload keys: {list(event.payload.keys())}")

            # Show specific suspicious patterns
            suspicious_patterns = []

            # Check for high activity (using decayed counts)
            actor_stats = detector.feature_extractor.get_actor_stats(event.actor.login)
            actor_count = actor_stats["total_events"]
            if actor_count > 50:
                suspicious_patterns.append(f"High activity: {actor_count:.1f} decayed events")

            # Check for delete events (potentially destructive)
            if event.type in ["DeleteEvent", "DestroyEvent"]:
                suspicious_patterns.append("Destructive event type")

            # Check for force push
            if event.type == "PushEvent" and "forced" in event.payload:
                if event.payload.get("forced"):
                    suspicious_patterns.append("Force push detected")

            if suspicious_patterns:
                print("  🚨 Suspicious patterns:")
                for pattern in suspicious_patterns:
                    print(f"    - {pattern}")

    # Analyze anomaly patterns
    if anomalies:
        separator = "=" * 80
        print(f"\n{separator}")
        print("Anomaly Pattern Analysis:")
        print(separator)

        # Event type distribution among anomalies
        anomaly_types = Counter(event.type for _, event, _ in anomalies)
        print("\nMost common event types in anomalies:")
        for event_type, count in anomaly_types.most_common(10):
            percentage = (count / len(anomalies)) * 100
            print(f"  {event_type:30s} {count:5,} ({percentage:5.1f}%)")

        # Actor distribution among anomalies
        anomaly_actors = Counter(event.actor.login for _, event, _ in anomalies)
        print("\nMost frequent actors in anomalies:")
        for actor, count in anomaly_actors.most_common(10):
            print(f"  {actor:30s} {count:5,} anomalies")

        # Repository distribution among anomalies
        anomaly_repos = Counter(event.repo.name for _, event, _ in anomalies)
        print("\nMost targeted repositories in anomalies:")
        for repo, count in anomaly_repos.most_common(10):
            print(f"  {repo:50s} {count:5,} anomalies")


def main() -> None:
    """Main entry point for the anomaly detection script."""
    if len(sys.argv) < 2:
        print("Usage: python detect_anomalies_rrcf.py <archive-file.json.gz> [threshold]")
        print()
        print("Arguments:")
        print("  archive-file.json.gz  Path to GH Archive file")
        print("  threshold             Optional: Anomaly score threshold (default: 2.0)")
        print()
        print("Example:")
        print("  # Download an archive:")
        print("  wget https://data.gharchive.org/2015-01-01-15.json.gz")
        print()
        print("  # Detect anomalies:")
        print("  python examples/detect_anomalies_rrcf.py 2015-01-01-15.json.gz")
        print()
        print("  # With custom threshold:")
        print("  python examples/detect_anomalies_rrcf.py 2015-01-01-15.json.gz 3.0")
        sys.exit(1)

    archive_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    # Check if file exists
    if not Path(archive_path).exists():
        print(f"Error: File not found: {archive_path}")
        sys.exit(1)

    try:
        analyze_archive_for_anomalies(archive_path, anomaly_threshold=threshold)

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
