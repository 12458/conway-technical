#!/usr/bin/env python3
"""Real-time GitHub event anomaly detection using RRCF with streaming.

This script demonstrates how to detect anomalies in real-time as events stream in.
It includes adaptive thresholding and maintains a sliding window of recent events.

Features:
- Streaming anomaly detection
- Adaptive threshold based on recent history
- Windowed statistics
- Alert system for high-severity anomalies

Usage:
    python examples/realtime_anomaly_detection.py <path-to-archive.json.gz>
"""

import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import rrcf

from github_client import GHArchiveLoader


class StreamingRRCFDetector:
    """Real-time streaming anomaly detector using RRCF."""

    def __init__(
        self,
        num_trees: int = 40,
        tree_size: int = 256,
        window_size: int = 100,
        percentile_threshold: float = 99.0,
    ):
        """Initialize streaming detector.

        Args:
            num_trees: Number of trees in the forest
            tree_size: Maximum points per tree
            window_size: Size of sliding window for adaptive thresholding
            percentile_threshold: Percentile for adaptive threshold (e.g., 99.0 = 99th percentile)
        """
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.window_size = window_size
        self.percentile_threshold = percentile_threshold

        # Initialize forest
        self.forest = {i: rrcf.RCTree() for i in range(num_trees)}

        # Sliding window of recent scores for adaptive thresholding
        self.score_window = deque(maxlen=window_size)

        # Feature tracking
        self.actor_stats = {}  # actor -> stats dict
        self.repo_stats = {}  # repo -> stats dict
        self.event_type_ids = {}

        self.index = 0

    def _update_actor_stats(self, actor_login: str, event_type: str) -> dict:
        """Update and return actor statistics."""
        if actor_login not in self.actor_stats:
            self.actor_stats[actor_login] = {
                "total_events": 0,
                "repos": set(),
                "event_types": [],
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
            }

        stats = self.actor_stats[actor_login]
        stats["total_events"] += 1
        stats["event_types"].append(event_type)
        stats["last_seen"] = datetime.now()

        return stats

    def _update_repo_stats(self, repo_name: str) -> dict:
        """Update and return repository statistics."""
        if repo_name not in self.repo_stats:
            self.repo_stats[repo_name] = {
                "total_events": 0,
                "actors": set(),
                "event_types": [],
            }

        stats = self.repo_stats[repo_name]
        stats["total_events"] += 1

        return stats

    def _extract_features(self, event: Any) -> np.ndarray:
        """Extract features from event with streaming context."""
        features = []

        # Actor features
        actor_login = event.actor.login
        actor_stats = self._update_actor_stats(actor_login, event.type)
        actor_stats["repos"].add(event.repo.name)

        features.extend(
            [
                actor_stats["total_events"],
                len(actor_stats["repos"]),
                event.actor.id % 10000,
            ]
        )

        # Repo features
        repo_name = event.repo.name
        repo_stats = self._update_repo_stats(repo_name)
        repo_stats["actors"].add(actor_login)

        features.extend([repo_stats["total_events"], len(repo_stats["actors"])])

        # Event type
        event_type = event.type
        if event_type not in self.event_type_ids:
            self.event_type_ids[event_type] = len(self.event_type_ids)
        features.append(self.event_type_ids[event_type])

        # Event velocity (events in last N seconds)
        recent_events = len(
            [t for t in actor_stats["event_types"][-10:] if t == event_type]
        )
        features.append(recent_events)

        # Payload complexity
        features.append(len(str(event.payload)))

        return np.array(features, dtype=float)

    def process_event(self, event: Any) -> dict:
        """Process event and return anomaly detection results.

        Returns:
            Dictionary with:
            - anomaly_score: The CoDisp score
            - is_anomaly: Whether event exceeds adaptive threshold
            - threshold: Current adaptive threshold
            - severity: Anomaly severity (low/medium/high/critical)
            - features: Feature vector
        """
        # Extract features
        features = self._extract_features(event)

        # Calculate anomaly score
        avg_codisp = 0.0

        for tree_idx, tree in self.forest.items():
            # Remove oldest point if at capacity
            if len(tree.leaves) >= self.tree_size:
                oldest = self.index - self.tree_size
                if oldest >= 0:
                    try:
                        tree.forget_point(oldest)
                    except KeyError:
                        pass

            # Insert new point
            tree.insert_point(features, index=self.index)

            # Calculate displacement
            if len(tree.leaves) > 1:
                avg_codisp += tree.codisp(self.index)

        avg_codisp /= self.num_trees

        # Update score window
        self.score_window.append(avg_codisp)

        # Calculate adaptive threshold
        if len(self.score_window) >= 20:  # Need minimum samples
            threshold = np.percentile(
                list(self.score_window), self.percentile_threshold
            )
        else:
            threshold = 2.0  # Default threshold

        # Determine if anomaly
        is_anomaly = avg_codisp > threshold

        # Calculate severity
        if not is_anomaly:
            severity = "normal"
        else:
            # Calculate how many standard deviations above threshold
            if len(self.score_window) >= 20:
                mean = np.mean(list(self.score_window))
                std = np.std(list(self.score_window))
                z_score = (avg_codisp - mean) / (std + 1e-10)

                if z_score > 5:
                    severity = "critical"
                elif z_score > 3:
                    severity = "high"
                elif z_score > 2:
                    severity = "medium"
                else:
                    severity = "low"
            else:
                severity = "low"

        self.index += 1

        return {
            "anomaly_score": avg_codisp,
            "is_anomaly": is_anomaly,
            "threshold": threshold,
            "severity": severity,
            "features": features,
        }

    def get_suspicious_patterns(self, event: Any, result: dict) -> list[str]:
        """Identify specific suspicious patterns in an anomalous event."""
        patterns = []

        actor_login = event.actor.login
        actor_stats = self.actor_stats.get(actor_login, {})

        # High velocity
        if actor_stats.get("total_events", 0) > 100:
            patterns.append(
                f"High velocity: {actor_stats['total_events']} events from {actor_login}"
            )

        # Rapid repo switching
        if len(actor_stats.get("repos", set())) > 20:
            patterns.append(
                f"Repo hopping: {len(actor_stats['repos'])} different repos"
            )

        # Destructive actions
        destructive_events = ["DeleteEvent", "DestroyEvent"]
        if event.type in destructive_events:
            patterns.append(f"Destructive action: {event.type}")

        # Note: Force push and large push detection removed because GitHub Events API
        # does not provide 'forced' or 'size' fields for PushEvent

        # Member changes
        if event.type == "MemberEvent":
            action = event.payload.get("action", "")
            patterns.append(f"Permission change: {action}")

        # Public/Private changes
        if event.type == "PublicEvent":
            patterns.append("Repository made public")

        return patterns


def monitor_stream(archive_path: str, alert_threshold: str = "medium") -> None:
    """Monitor event stream for anomalies with real-time alerts.

    Args:
        archive_path: Path to archive file
        alert_threshold: Minimum severity for alerts (low/medium/high/critical)
    """
    severity_order = ["normal", "low", "medium", "high", "critical"]
    min_severity_idx = severity_order.index(alert_threshold)

    print(f"Monitoring GitHub events from: {archive_path}")
    print(f"Alert threshold: {alert_threshold}")
    print("=" * 80)

    loader = GHArchiveLoader(archive_path, strict=False)
    detector = StreamingRRCFDetector(
        num_trees=40, tree_size=256, window_size=100, percentile_threshold=99.0
    )

    event_count = 0
    anomaly_count = 0
    severity_counts = {s: 0 for s in severity_order}

    print("\nProcessing events (streaming mode)...")
    print("Press Ctrl+C to stop\n")

    try:
        for event in loader.iter_events():
            event_count += 1

            # Detect anomaly
            result = detector.process_event(event)
            severity = result["severity"]
            severity_counts[severity] += 1

            if result["is_anomaly"]:
                anomaly_count += 1

            # Alert if severity meets threshold
            severity_idx = severity_order.index(severity)
            if severity_idx >= min_severity_idx:
                patterns = detector.get_suspicious_patterns(event, result)

                # Color coding for severity
                severity_emoji = {
                    "low": "⚠️ ",
                    "medium": "🔶",
                    "high": "🔴",
                    "critical": "🚨",
                }

                print(
                    f"\n{severity_emoji.get(severity, '')} ALERT - {severity.upper()}"
                )
                print(
                    f"  Score: {result['anomaly_score']:.2f} (threshold: {result['threshold']:.2f})"
                )
                print(f"  Event: {event.type} by {event.actor.login}")
                print(f"  Repo: {event.repo.name}")
                print(f"  Time: {event.created_at}")

                if patterns:
                    print("  Suspicious patterns:")
                    for pattern in patterns:
                        print(f"    • {pattern}")

            # Periodic summary
            if event_count % 1000 == 0:
                anomaly_rate = (anomaly_count / event_count) * 100
                print(f"\n--- Summary after {event_count:,} events ---")
                print(f"  Anomalies detected: {anomaly_count:,} ({anomaly_rate:.2f}%)")
                print(f"  Current threshold: {result['threshold']:.2f}")
                print(f"  Severity breakdown:")
                for sev in ["low", "medium", "high", "critical"]:
                    count = severity_counts[sev]
                    if count > 0:
                        pct = (count / event_count) * 100
                        print(f"    {sev.capitalize():10s}: {count:5,} ({pct:5.2f}%)")

    except KeyboardInterrupt:
        print("\n\nStream monitoring stopped by user")

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total events processed: {event_count:,}")
    print(f"Total anomalies: {anomaly_count:,}")
    print(f"Anomaly rate: {(anomaly_count / event_count * 100):.2f}%")
    print(f"\nSeverity distribution:")
    for severity in ["normal", "low", "medium", "high", "critical"]:
        count = severity_counts[severity]
        pct = (count / event_count) * 100 if event_count > 0 else 0
        print(f"  {severity.capitalize():10s}: {count:6,} ({pct:5.2f}%)")


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python realtime_anomaly_detection.py <archive-file.json.gz> [alert_threshold]"
        )
        print()
        print("Arguments:")
        print("  archive-file.json.gz  Path to GH Archive file")
        print(
            "  alert_threshold       Optional: low/medium/high/critical (default: medium)"
        )
        print()
        print("Example:")
        print(
            "  python examples/realtime_anomaly_detection.py 2015-01-01-15.json.gz high"
        )
        sys.exit(1)

    archive_path = sys.argv[1]
    alert_threshold = sys.argv[2] if len(sys.argv) > 2 else "medium"

    if alert_threshold not in ["low", "medium", "high", "critical"]:
        print(f"Error: Invalid alert threshold '{alert_threshold}'")
        print("Must be one of: low, medium, high, critical")
        sys.exit(1)

    if not Path(archive_path).exists():
        print(f"Error: File not found: {archive_path}")
        sys.exit(1)

    try:
        monitor_stream(archive_path, alert_threshold)
    except Exception as e:
        print(f"\n\nError during monitoring: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
