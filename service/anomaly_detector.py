"""Anomaly detection using RRCF (Robust Random Cut Forest)."""

import logging
from typing import Any

import numpy as np
from newrrcf import RCTree

from github_client.feature_extractor import GitHubFeatureExtractor
from github_client.models import Event
from service.anomaly_detector_base import BaseAnomalyDetector
from service.config import service_settings

logger = logging.getLogger(__name__)


class StreamingAnomalyDetector(BaseAnomalyDetector):
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
        super().__init__(num_trees, tree_size, shingle_size, threshold)

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
        # Extract features and analyze event using base class method
        (
            features,
            velocity_score,
            is_inhuman_speed,
            velocity_reason,
            suspicious_patterns,
        ) = self._extract_and_analyze_event(event, self.extractor)

        # Check if event was filtered
        if features is None:
            return None, [], None, 0.0, False, "Event filtered"

        # Calculate anomaly score using RRCF with base class method
        avg_codisp = self._process_rrcf_trees(
            self.forest, features, self.point_index
        )

        # Increment point index
        self.point_index += 1

        logger.debug(
            f"Event {event.id} - CoDisp: {avg_codisp:.2f}, "
            f"Patterns: {len(suspicious_patterns)}, Type: {event.type}, "
            f"Velocity: {velocity_score:.1f} events/min, Inhuman: {is_inhuman_speed}"
        )

        return (
            avg_codisp,
            suspicious_patterns,
            features,
            velocity_score,
            is_inhuman_speed,
            velocity_reason,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics.

        Returns:
            Dictionary with detector stats
        """
        total_points = sum(len(tree.leaves) for tree in self.forest.values())
        avg_tree_size = total_points / self.num_trees if self.num_trees > 0 else 0

        # Get base stats (includes adaptive threshold info)
        stats = super().get_stats()

        # Add detector-specific stats
        stats.update({
            "points_processed": self.point_index,
            "avg_tree_size": avg_tree_size,
            "total_actors_tracked": len(self.extractor.actor_event_counts),
            "total_repos_tracked": len(self.extractor.repo_event_counts),
        })

        return stats


class MultiForestAnomalyDetector(BaseAnomalyDetector):
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
        super().__init__(num_trees, tree_size, shingle_size, threshold)

        # Build event type to forest group mapping
        self.event_type_to_group: dict[str, str] = {}
        for (
            group_name,
            event_types,
        ) in service_settings.event_type_forest_groups.items():
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

        # Extract features and analyze event using base class method
        (
            features,
            velocity_score,
            is_inhuman_speed,
            velocity_reason,
            suspicious_patterns,
        ) = self._extract_and_analyze_event(event, self.extractor)

        # Check if event was filtered
        if features is None:
            return None, [], None, 0.0, False, "Event filtered"

        # Calculate anomaly score using RRCF with base class method
        avg_codisp = self._process_rrcf_trees(
            detector.forest, features, detector.point_index
        )

        # Increment point index for this detector
        detector.point_index += 1

        logger.debug(
            f"Event {event.id} - Forest: {forest_group}, CoDisp: {avg_codisp:.2f}, "
            f"Patterns: {len(suspicious_patterns)}, Type: {event.type}, "
            f"Velocity: {velocity_score:.1f} events/min, Inhuman: {is_inhuman_speed}"
        )

        return (
            avg_codisp,
            suspicious_patterns,
            features,
            velocity_score,
            is_inhuman_speed,
            velocity_reason,
        )

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

            # Get stats from each sub-detector (includes adaptive threshold)
            sub_stats = detector.get_stats()

            forest_stats[group_name] = {
                "points_processed": detector.point_index,
                "avg_tree_size": group_points / len(detector.forest)
                if detector.forest
                else 0,
                "adaptive_threshold": sub_stats.get("adaptive_threshold", {}),
            }

        # Get base stats (includes parent-level adaptive threshold config)
        stats = super().get_stats()

        # Add multi-forest specific stats
        stats.update({
            "num_forest_groups": len(self.detectors),
            "num_trees_per_group": self.num_trees,
            "total_trees": total_trees,
            "total_points": total_points,
            "avg_tree_size": total_points / total_trees if total_trees > 0 else 0,
            "total_actors_tracked": len(self.extractor.actor_event_counts),
            "total_repos_tracked": len(self.extractor.repo_event_counts),
            "forest_stats": forest_stats,
        })

        return stats


# Global detector instance (use multi-forest if enabled, otherwise single forest)
if service_settings.enable_multi_forest:
    detector = MultiForestAnomalyDetector()
    logger.info("Using MultiForestAnomalyDetector with event-type-specific forests")
else:
    detector = StreamingAnomalyDetector()
    logger.info("Using single StreamingAnomalyDetector (multi-forest disabled)")
