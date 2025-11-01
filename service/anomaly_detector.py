"""Anomaly detection using RRCF (Robust Random Cut Forest)."""

import logging
from typing import Any

import numpy as np
from newrrcf import RCTree

from github_client.feature_extractor import GitHubFeatureExtractor, get_suspicious_patterns
from github_client.models import Event
from service.config import service_settings

logger = logging.getLogger(__name__)


class StreamingAnomalyDetector:
    """Streaming anomaly detector using RRCF.

    Uses a forest of Robust Random Cut Trees to detect anomalies in real-time
    as GitHub events stream in.
    """

    def __init__(
        self,
        num_trees: int | None = None,
        tree_size: int | None = None,
        shingle_size: int | None = None,
        threshold: float | None = None,
    ):
        """Initialize the anomaly detector.

        Args:
            num_trees: Number of trees in forest
            tree_size: Maximum size of each tree
            shingle_size: Shingle size for streaming
            threshold: CoDisp score threshold for anomaly detection
        """
        self.num_trees = num_trees or service_settings.num_trees
        self.tree_size = tree_size or service_settings.tree_size
        self.shingle_size = shingle_size or service_settings.shingle_size
        self.threshold = threshold or service_settings.anomaly_threshold

        # Initialize forest of trees
        self.forest: dict[int, RCTree] = {}
        for i in range(self.num_trees):
            self.forest[i] = RCTree()

        # Feature extractor with bot filtering
        self.extractor = GitHubFeatureExtractor(
            filter_bots=service_settings.enable_bot_filtering,
        )

        # Point index counter
        self.point_index = 0

        logger.info(
            f"Initialized anomaly detector: {self.num_trees} trees, "
            f"size {self.tree_size}, threshold {self.threshold}"
        )

    def process_event(
        self, event: Event
    ) -> tuple[float | None, list[str], np.ndarray | None, float, bool, str]:
        """Process a single event and return anomaly score.

        Args:
            event: GitHub Event to process

        Returns:
            Tuple of (anomaly_score, suspicious_patterns, features, velocity_score, is_inhuman_speed, velocity_reason)
            - anomaly_score: CoDisp score (None if event filtered or no features)
            - suspicious_patterns: List of detected patterns
            - features: Extracted feature vector (None if filtered)
            - velocity_score: Events per minute velocity score
            - is_inhuman_speed: True if velocity exceeds threshold
            - velocity_reason: Human-readable velocity explanation
        """
        # Extract features
        features = self.extractor.extract_features(event)

        # Check if event was filtered (e.g., bot)
        if features is None:
            logger.debug(f"Event {event.id} filtered (likely bot: {event.actor.login})")
            return None, [], None, 0.0, False, "Event filtered"

        # Extract timestamp for velocity detection
        import datetime
        if isinstance(event.created_at, datetime.datetime):
            event_timestamp = event.created_at.timestamp()
        else:
            # Fallback for string timestamps
            event_timestamp = datetime.datetime.fromisoformat(
                event.created_at.replace('Z', '+00:00')
            ).timestamp()

        # Get velocity-based anomaly score
        velocity_score, is_inhuman_speed, velocity_reason = (
            self.extractor.get_velocity_anomaly_score(event.actor.login, event_timestamp)
        )

        # Get suspicious patterns using rule-based detection
        suspicious_patterns = get_suspicious_patterns(event, self.extractor)

        # Calculate anomaly score using RRCF
        avg_codisp = 0.0

        # Insert point into all trees and compute CoDisp
        for tree_idx in self.forest:
            tree = self.forest[tree_idx]

            # Insert point
            tree.insert_point(features, index=self.point_index)

            # Compute CoDisp (collusive displacement)
            codisp = tree.codisp(self.point_index)
            avg_codisp += codisp

            # Maintain tree size by forgetting oldest points
            if len(tree.leaves) > self.tree_size:
                # Forget oldest point (FIFO)
                oldest_index = self.point_index - self.tree_size
                if oldest_index in tree.leaves:
                    tree.forget_point(oldest_index)

        # Average CoDisp across all trees
        avg_codisp /= self.num_trees

        # Increment point index
        self.point_index += 1

        logger.debug(
            f"Event {event.id} - CoDisp: {avg_codisp:.2f}, "
            f"Patterns: {len(suspicious_patterns)}, Type: {event.type}, "
            f"Velocity: {velocity_score:.1f} events/min, Inhuman: {is_inhuman_speed}"
        )

        return avg_codisp, suspicious_patterns, features, velocity_score, is_inhuman_speed, velocity_reason

    def is_anomaly(
        self, score: float, patterns: list[str], is_inhuman_speed: bool = False
    ) -> bool:
        """Check if event should be flagged as anomaly.

        Args:
            score: CoDisp score
            patterns: Suspicious patterns detected (not used in detection)
            is_inhuman_speed: Whether velocity-based detection flagged inhuman speed

        Returns:
            True if score exceeds threshold (velocity is tracked but does not bypass score requirement)
        """
        # Score threshold is a hard requirement
        # Velocity detection is tracked for context but does not bypass the threshold
        return score >= self.threshold

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics.

        Returns:
            Dictionary with detector stats
        """
        total_points = sum(len(tree.leaves) for tree in self.forest.values())
        avg_tree_size = total_points / self.num_trees if self.num_trees > 0 else 0

        return {
            "num_trees": self.num_trees,
            "tree_size_limit": self.tree_size,
            "threshold": self.threshold,
            "points_processed": self.point_index,
            "avg_tree_size": avg_tree_size,
            "total_actors_tracked": len(self.extractor.actor_event_counts),
            "total_repos_tracked": len(self.extractor.repo_event_counts),
        }


class MultiForestAnomalyDetector:
    """Multi-forest anomaly detector with event-type-specific forests.

    Maintains separate RRCF forests for different event type groups to prevent
    rare event types from being flagged as anomalous due to type imbalance.
    """

    def __init__(
        self,
        num_trees: int | None = None,
        tree_size: int | None = None,
        shingle_size: int | None = None,
        threshold: float | None = None,
    ):
        """Initialize the multi-forest anomaly detector.

        Args:
            num_trees: Number of trees per forest
            tree_size: Maximum size of each tree
            shingle_size: Shingle size for streaming
            threshold: CoDisp score threshold for anomaly detection
        """
        self.num_trees = num_trees or service_settings.num_trees
        self.tree_size = tree_size or service_settings.tree_size
        self.shingle_size = shingle_size or service_settings.shingle_size
        self.threshold = threshold or service_settings.anomaly_threshold

        # Build event type to forest group mapping
        self.event_type_to_group: dict[str, str] = {}
        for group_name, event_types in service_settings.event_type_forest_groups.items():
            for event_type in event_types:
                self.event_type_to_group[event_type] = group_name

        # Initialize one detector per forest group
        self.detectors: dict[str, StreamingAnomalyDetector] = {}
        for group_name in service_settings.event_type_forest_groups.keys():
            self.detectors[group_name] = StreamingAnomalyDetector(
                num_trees=self.num_trees,
                tree_size=self.tree_size,
                shingle_size=self.shingle_size,
                threshold=self.threshold,
            )

        # Shared feature extractor across all detectors (for consistency)
        # Use the extractor from the first detector
        first_detector = next(iter(self.detectors.values()))
        self.extractor = first_detector.extractor

        # Override extractors in all detectors to use shared extractor
        for detector in self.detectors.values():
            detector.extractor = self.extractor

        logger.info(
            f"Initialized multi-forest detector: {len(self.detectors)} forest groups, "
            f"{self.num_trees} trees each, size {self.tree_size}, threshold {self.threshold}"
        )
        logger.info(f"Forest groups: {list(self.detectors.keys())}")

    def _get_forest_group(self, event_type: str) -> str:
        """Get the forest group name for an event type.

        Args:
            event_type: Event type (e.g., 'PushEvent')

        Returns:
            Forest group name (e.g., 'push', 'issues', 'other')
        """
        return self.event_type_to_group.get(event_type, "other")

    def process_event(
        self, event: Event
    ) -> tuple[float | None, list[str], np.ndarray | None, float, bool, str]:
        """Process a single event and return anomaly score.

        Routes the event to the appropriate forest based on event type.

        Args:
            event: GitHub Event to process

        Returns:
            Tuple of (anomaly_score, suspicious_patterns, features, velocity_score, is_inhuman_speed, velocity_reason)
            - anomaly_score: CoDisp score (None if event filtered or no features)
            - suspicious_patterns: List of detected patterns
            - features: Extracted feature vector (None if filtered)
            - velocity_score: Events per minute velocity score
            - is_inhuman_speed: True if velocity exceeds threshold
            - velocity_reason: Human-readable velocity explanation
        """
        # Get appropriate forest group for this event type
        forest_group = self._get_forest_group(event.type)
        detector = self.detectors[forest_group]

        # Extract features using shared extractor
        features = self.extractor.extract_features(event)

        # Check if event was filtered (e.g., bot)
        if features is None:
            logger.debug(f"Event {event.id} filtered (likely bot: {event.actor.login})")
            return None, [], None, 0.0, False, "Event filtered"

        # Extract timestamp for velocity detection
        import datetime
        if isinstance(event.created_at, datetime.datetime):
            event_timestamp = event.created_at.timestamp()
        else:
            # Fallback for string timestamps
            event_timestamp = datetime.datetime.fromisoformat(
                event.created_at.replace('Z', '+00:00')
            ).timestamp()

        # Get velocity-based anomaly score
        velocity_score, is_inhuman_speed, velocity_reason = (
            self.extractor.get_velocity_anomaly_score(event.actor.login, event_timestamp)
        )

        # Get suspicious patterns using rule-based detection
        suspicious_patterns = get_suspicious_patterns(event, self.extractor)

        # Calculate anomaly score using RRCF in the appropriate forest
        avg_codisp = 0.0

        # Insert point into all trees in this forest and compute CoDisp
        for tree_idx in detector.forest:
            tree = detector.forest[tree_idx]

            # Insert point
            tree.insert_point(features, index=detector.point_index)

            # Compute CoDisp (collusive displacement)
            codisp = tree.codisp(detector.point_index)
            avg_codisp += codisp

            # Maintain tree size by forgetting oldest points
            if len(tree.leaves) > self.tree_size:
                # Forget oldest point (FIFO)
                oldest_index = detector.point_index - self.tree_size
                if oldest_index in tree.leaves:
                    tree.forget_point(oldest_index)

        # Average CoDisp across all trees
        avg_codisp /= self.num_trees

        # Increment point index for this detector
        detector.point_index += 1

        logger.debug(
            f"Event {event.id} - Forest: {forest_group}, CoDisp: {avg_codisp:.2f}, "
            f"Patterns: {len(suspicious_patterns)}, Type: {event.type}, "
            f"Velocity: {velocity_score:.1f} events/min, Inhuman: {is_inhuman_speed}"
        )

        return avg_codisp, suspicious_patterns, features, velocity_score, is_inhuman_speed, velocity_reason

    def is_anomaly(
        self, score: float, patterns: list[str], is_inhuman_speed: bool = False
    ) -> bool:
        """Check if event should be flagged as anomaly.

        Args:
            score: CoDisp score
            patterns: Suspicious patterns detected (not used in detection)
            is_inhuman_speed: Whether velocity-based detection flagged inhuman speed

        Returns:
            True if score exceeds threshold (velocity is tracked but does not bypass score requirement)
        """
        # Score threshold is a hard requirement
        # Velocity detection is tracked for context but does not bypass the threshold
        return score >= self.threshold

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics across all forests.

        Returns:
            Dictionary with detector stats including per-forest breakdowns
        """
        forest_stats = {}
        total_points = 0
        total_trees = 0

        for group_name, detector in self.detectors.items():
            group_points = sum(len(tree.leaves) for tree in detector.forest.values())
            total_points += group_points
            total_trees += len(detector.forest)

            forest_stats[group_name] = {
                "points_processed": detector.point_index,
                "avg_tree_size": group_points / len(detector.forest) if detector.forest else 0,
            }

        return {
            "num_forest_groups": len(self.detectors),
            "num_trees_per_group": self.num_trees,
            "total_trees": total_trees,
            "tree_size_limit": self.tree_size,
            "threshold": self.threshold,
            "total_points": total_points,
            "avg_tree_size": total_points / total_trees if total_trees > 0 else 0,
            "total_actors_tracked": len(self.extractor.actor_event_counts),
            "total_repos_tracked": len(self.extractor.repo_event_counts),
            "forest_stats": forest_stats,
        }


# Global detector instance (use multi-forest if enabled, otherwise single forest)
if service_settings.enable_multi_forest:
    detector = MultiForestAnomalyDetector()
    logger.info("Using MultiForestAnomalyDetector with event-type-specific forests")
else:
    detector = StreamingAnomalyDetector()
    logger.info("Using single StreamingAnomalyDetector (multi-forest disabled)")
