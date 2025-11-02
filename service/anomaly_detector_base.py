"""Base class for anomaly detectors with shared logic."""

import logging
from collections import deque
from typing import Any

import numpy as np
from newrrcf import RCTree

from github_client.feature_extractor import (
    GitHubFeatureExtractor,
    get_suspicious_patterns,
)
from github_client.models import Event
from github_client.utils import extract_timestamp
from service.config import service_settings

logger = logging.getLogger(__name__)


class BaseAnomalyDetector:
    """Base class with shared logic for anomaly detection.

    This class provides common functionality for RRCF-based anomaly detectors,
    including feature extraction, velocity analysis, and pattern detection.
    """

    def __init__(
        self,
        num_trees: int | None = None,
        tree_size: int | None = None,
        shingle_size: int | None = None,
        threshold: float | None = None,
    ):
        """Initialize base anomaly detector parameters.

        Args:
            num_trees: Number of trees in the forest
            tree_size: Maximum size of each tree
            shingle_size: Shingle size for streaming
            threshold: CoDisp score threshold for anomaly detection
        """
        self.num_trees = num_trees or service_settings.num_trees
        self.tree_size = tree_size or service_settings.tree_size
        self.shingle_size = shingle_size or service_settings.shingle_size
        self.threshold = threshold or service_settings.anomaly_threshold

        # Adaptive threshold tracking
        self.enable_adaptive = service_settings.enable_adaptive_threshold
        self.adaptive_percentile = service_settings.adaptive_percentile
        self.min_samples = service_settings.min_samples_for_adaptive
        self.recent_scores: deque[float] = deque(
            maxlen=service_settings.adaptive_window_size
        )
        self.current_adaptive_threshold: float | None = None

    def _extract_and_analyze_event(
        self, event: Event, extractor: GitHubFeatureExtractor
    ) -> tuple[np.ndarray | None, float, bool, str, list[str]]:
        """Extract features and analyze event for velocity and patterns.

        Args:
            event: GitHub Event to process
            extractor: Feature extractor instance

        Returns:
            Tuple of (features, velocity_score, is_inhuman_speed, velocity_reason, suspicious_patterns)
            - features: Extracted feature vector (None if filtered)
            - velocity_score: Events per minute velocity score
            - is_inhuman_speed: True if velocity exceeds threshold
            - velocity_reason: Human-readable velocity explanation
            - suspicious_patterns: List of detected patterns
        """
        # Extract features
        features = extractor.extract_features(event)

        # Check if event was filtered (e.g., bot)
        if features is None:
            logger.debug(f"Event {event.id} filtered (likely bot: {event.actor.login})")
            return None, 0.0, False, "Event filtered", []

        # Extract timestamp for velocity detection
        event_timestamp = extract_timestamp(event)

        # Get velocity-based anomaly score
        velocity_score, is_inhuman_speed, velocity_reason = (
            extractor.get_velocity_anomaly_score(event.actor.login, event_timestamp)
        )

        # Get suspicious patterns using rule-based detection
        suspicious_patterns = get_suspicious_patterns(event, extractor)

        return features, velocity_score, is_inhuman_speed, velocity_reason, suspicious_patterns

    def _process_rrcf_trees(
        self,
        forest: dict[int, RCTree],
        features: np.ndarray,
        point_index: int,
    ) -> float:
        """Process point through RRCF forest and return average CoDisp.

        Args:
            forest: Dictionary of RCTree instances
            features: Feature vector to insert
            point_index: Current point index

        Returns:
            Average CoDisp score across all trees
        """
        avg_codisp = 0.0

        # Insert point into all trees and compute CoDisp
        for tree_idx in forest:
            tree = forest[tree_idx]

            # Insert point
            tree.insert_point(features, index=point_index)

            # Compute CoDisp (collusive displacement)
            codisp = tree.codisp(point_index)
            avg_codisp += codisp

            # Maintain tree size by forgetting oldest points
            if len(tree.leaves) > self.tree_size:
                # Forget oldest point (FIFO)
                oldest_index = point_index - self.tree_size
                if oldest_index in tree.leaves:
                    tree.forget_point(oldest_index)

        # Average CoDisp across all trees
        avg_codisp /= self.num_trees

        return avg_codisp

    def _update_adaptive_threshold(self, score: float) -> None:
        """Update the adaptive threshold with a new score.

        Args:
            score: CoDisp score to add to rolling window
        """
        if not self.enable_adaptive:
            return

        # Add score to rolling window
        self.recent_scores.append(score)

        # Calculate adaptive threshold if we have enough samples
        if len(self.recent_scores) >= self.min_samples:
            self.current_adaptive_threshold = float(
                np.percentile(self.recent_scores, self.adaptive_percentile)
            )
            logger.debug(
                f"Adaptive threshold updated: {self.current_adaptive_threshold:.2f} "
                f"(based on {len(self.recent_scores)} scores, {self.adaptive_percentile}th percentile)"
            )

    def _get_effective_threshold(self) -> float:
        """Get the current effective threshold for anomaly detection.

        Returns:
            The adaptive threshold if available and enabled, otherwise the static threshold
        """
        if (
            self.enable_adaptive
            and self.current_adaptive_threshold is not None
            and len(self.recent_scores) >= self.min_samples
        ):
            return self.current_adaptive_threshold
        return self.threshold

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
        # Update adaptive threshold with this score
        self._update_adaptive_threshold(score)

        # Get the effective threshold (adaptive or static)
        effective_threshold = self._get_effective_threshold()

        # Score threshold is a hard requirement
        # Velocity detection is tracked for context but does not bypass the threshold
        return score >= effective_threshold

    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics.

        This method should be overridden by subclasses to provide
        detector-specific statistics.

        Returns:
            Dictionary with detector statistics
        """
        stats = {
            "num_trees": self.num_trees,
            "tree_size": self.tree_size,
            "static_threshold": self.threshold,
        }

        # Add adaptive threshold stats if enabled
        if self.enable_adaptive:
            stats["adaptive_threshold"] = {
                "enabled": True,
                "current_threshold": self.current_adaptive_threshold,
                "effective_threshold": self._get_effective_threshold(),
                "percentile_target": self.adaptive_percentile,
                "scores_tracked": len(self.recent_scores),
                "min_samples_warmup": self.min_samples,
                "is_warmed_up": len(self.recent_scores) >= self.min_samples,
            }
        else:
            stats["adaptive_threshold"] = {"enabled": False}

        return stats
