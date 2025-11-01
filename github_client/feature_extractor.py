"""Feature extraction for GitHub events using RRCF-optimized encoding.

This module extracts features from GitHub Event objects for anomaly detection.
All features use proper encoding for RRCF:
- Categorical features: one-hot hashing (no ordinal encoding)
- Text features: character n-gram hashing
- Counts: exponential decay (bounded, time-sensitive)
- Normalization: z-score using Welford's algorithm
"""

from collections import OrderedDict, defaultdict
from typing import Any

import numpy as np
import xxhash

from github_client.models import Event

# Memory limits for LRU eviction
MAX_ACTORS = 50_000
MAX_REPOS = 25_000
MAX_TEXT_BYTES = 2048  # 2KB text limit

# Known bot accounts to filter out
KNOWN_BOTS = {
    "github-actions[bot]",
    "dependabot[bot]",
    "coderabbitai[bot]",
    "socket-security[bot]",
    "renovate[bot]",
    "greenkeeper[bot]",
    "codecov[bot]",
    "codecov-io",
    "codecov-commenter",
    "imgbot[bot]",
    "snyk-bot",
    "sonarcloud[bot]",
    "renovate-bot",
    "pyup-bot",
    "mergify[bot]",
    "restyled-io[bot]",
    "vercel[bot]",
    "netlify[bot]",
    "github-advanced-security[bot]",
    "microsoft-github-policy-service[bot]",
    "allcontributors[bot]",
    "semantic-release-bot",
    "gitpod-io[bot]",
    "codesandbox[bot]",
    "pre-commit-ci[bot]",
    "stale[bot]",
    "fossabot",
    "coveralls",
    "lgtm-com[bot]",
    "dependabot-preview[bot]",
    "whitesource-bolt-for-github[bot]",
    "sourcery-ai[bot]",
    "deepsource-autofix[bot]",
    "sweep-ai[bot]",
    "linear[bot]",
    "gitguardian[bot]",
    "copilot[bot]",
    # Additional bots for moderate filtering
    "cloudflare-workers-and-pages[bot]",
    "release-please[bot]",
    "TheBoatyMcBotFace",  # Release-please style bot
    "Copilot",  # GitHub Copilot SWE agent
    "soc-se-bot",  # Educational institution bot
    "cursoragent[bot]",
    "gemini[bot]",
}


class GitHubFeatureExtractor:
    """Extract RRCF-optimized numerical features from GitHub events.

    This extractor uses:
    - One-hot feature hashing for categoricals (event type, actor, repo, org)
    - Character n-gram hashing for text (commits, titles, bodies, comments)
    - Exponential decay for counts (bounded, streaming-friendly)
    - Z-score normalization using Welford's online algorithm
    """

    def __init__(
        self,
        categorical_dims: dict[str, int] | None = None,
        text_dims: dict[str, int] | None = None,
        decay_halflife: float = 1000.0,
        normalize: bool = True,
        ngram_size: int = 4,
        normalization_frequency: int = 10,
        filter_bots: bool = True,
    ):
        """Initialize the feature extractor with RRCF-optimized settings.

        Args:
            categorical_dims: Hash dimensions for categoricals
                (event_type, actor, repo, org, action). Defaults to
                {event_type: 32, actor: 64, repo: 64, org: 32, action: 16}
            text_dims: Hash dimensions for text fields
                (commits, titles, bodies, comments). Defaults to
                {commits: 64, titles: 32, bodies: 32, comments: 32}
            decay_halflife: Number of events for counts to decay by 50%.
                Smaller = more sensitive to recent changes.
            normalize: Enable z-score normalization using running mean/std
            ngram_size: Character n-gram size for text hashing (default: 4)
            normalization_frequency: Update Welford stats every N events (default: 10)
            filter_bots: Filter out known bot accounts (default: True)
        """
        # Hash dimensions (reduced by 50% for performance)
        self.categorical_dims = categorical_dims or {
            "event_type": 32,
            "actor": 64,
            "repo": 64,
            "org": 32,
            "action": 16,
        }
        self.text_dims = text_dims or {
            "commits": 64,
            "titles": 32,
            "bodies": 32,
            "comments": 32,
        }

        # Exponential decay parameters
        self.decay_halflife = decay_halflife
        self.decay_alpha = 0.5 ** (1.0 / decay_halflife)  # Decay factor

        # Normalization
        self.normalize = normalize
        self.ngram_size = ngram_size
        self.normalization_frequency = normalization_frequency

        # Bot filtering
        self.filter_bots = filter_bots

        # LRU-bounded tracking with lazy decay (OrderedDict for LRU)
        self.actor_event_counts = OrderedDict()  # actor -> raw count
        self.actor_last_seen = {}  # actor -> event counter
        self.actor_repos = OrderedDict()  # actor -> set of repos
        self.actor_event_types = OrderedDict()  # actor -> list of types

        self.repo_event_counts = OrderedDict()  # repo -> raw count
        self.repo_last_seen = {}  # repo -> event counter
        self.repo_actors = OrderedDict()  # repo -> set of actors

        # Event counter for lazy decay
        self.total_events = 0

        # Welford's algorithm for running mean/std (for z-score normalization)
        self.feature_count = 0
        self.feature_mean = None
        self.feature_m2 = None  # Sum of squared differences from mean

    def _get_decayed_count(self, count: float, last_seen: int) -> float:
        """Compute decayed count using lazy decay.

        Args:
            count: Raw count stored at last_seen
            last_seen: Event counter when count was last updated

        Returns:
            Decayed count at current time
        """
        events_elapsed = self.total_events - last_seen
        return count * (self.decay_alpha ** events_elapsed)

    def _evict_lru_actors(self) -> None:
        """Evict oldest actor if limit exceeded."""
        if len(self.actor_event_counts) > MAX_ACTORS:
            # Remove oldest (first) item from OrderedDict
            oldest_actor = next(iter(self.actor_event_counts))
            del self.actor_event_counts[oldest_actor]
            self.actor_last_seen.pop(oldest_actor, None)
            self.actor_repos.pop(oldest_actor, None)
            self.actor_event_types.pop(oldest_actor, None)

    def _evict_lru_repos(self) -> None:
        """Evict oldest repo if limit exceeded."""
        if len(self.repo_event_counts) > MAX_REPOS:
            oldest_repo = next(iter(self.repo_event_counts))
            del self.repo_event_counts[oldest_repo]
            self.repo_last_seen.pop(oldest_repo, None)
            self.repo_actors.pop(oldest_repo, None)

    def is_bot(self, actor_login: str) -> bool:
        """Check if an actor is a known bot.

        Args:
            actor_login: Actor username to check

        Returns:
            True if actor is in KNOWN_BOTS set
        """
        return actor_login in KNOWN_BOTS

    def _hash_categorical(self, value: str, dim: int, seed: int = 0) -> np.ndarray:
        """Hash a categorical value into a fixed-dimension one-hot vector.

        Uses xxhash to map string to a single index in [0, dim).
        Creates a sparse one-hot vector with a single 1.0 at the hashed index.

        Args:
            value: Categorical string to hash
            dim: Dimension of output vector
            seed: Hash seed for different feature types

        Returns:
            One-hot numpy array of length dim
        """
        vec = np.zeros(dim, dtype=float)
        if value:
            idx = xxhash.xxh32(value.encode(), seed=seed).intdigest() % dim
            vec[idx] = 1.0
        return vec

    def _hash_text_ngrams(
        self, text: str, dim: int, n: int | None = None, seed: int = 0
    ) -> np.ndarray:
        """Hash text using character n-grams into a fixed-dimension vector.

        Creates character n-grams, hashes each to an index, and accumulates
        counts in a fixed-size vector (bag-of-hashed-ngrams).

        Args:
            text: Text string to hash
            dim: Dimension of output vector
            n: N-gram size (defaults to self.ngram_size)
            seed: Hash seed for different feature types

        Returns:
            Numpy array of n-gram hash counts (length dim)
        """
        if n is None:
            n = self.ngram_size

        vec = np.zeros(dim, dtype=float)
        if not text:
            return vec

        # Clip text to MAX_TEXT_BYTES for performance
        if len(text.encode('utf-8', errors='ignore')) > MAX_TEXT_BYTES:
            # Truncate at character boundary
            text = text.encode('utf-8', errors='ignore')[:MAX_TEXT_BYTES].decode('utf-8', errors='ignore')

        # Generate character n-grams
        text_clean = text.lower()
        for i in range(len(text_clean) - n + 1):
            ngram = text_clean[i : i + n]
            idx = xxhash.xxh32(ngram.encode(), seed=seed).intdigest() % dim
            vec[idx] += 1.0

        # Normalize by text length to make comparable
        if len(text_clean) >= n:
            vec /= max(len(text_clean) - n + 1, 1)

        return vec

    def _update_welford(self, features: np.ndarray) -> None:
        """Update running mean and variance using Welford's algorithm.

        Args:
            features: Feature vector to incorporate
        """
        self.feature_count += 1
        if self.feature_mean is None:
            # Initialize on first sample
            self.feature_mean = features.copy()
            self.feature_m2 = np.zeros_like(features)
        else:
            # Welford's online update
            delta = features - self.feature_mean
            self.feature_mean += delta / self.feature_count
            delta2 = features - self.feature_mean
            self.feature_m2 += delta * delta2

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to features.

        Args:
            features: Raw feature vector

        Returns:
            Z-scored feature vector (mean=0, std=1 per feature)
        """
        if self.feature_count < 2 or self.feature_mean is None:
            return features  # Not enough data yet

        variance = self.feature_m2 / self.feature_count
        std = np.sqrt(variance)
        # Avoid division by zero
        std = np.where(std < 1e-10, 1.0, std)

        return (features - self.feature_mean) / std

    def extract_features(self, event: Event) -> np.ndarray | None:
        """Extract RRCF-optimized feature vector from a GitHub event.

        Features (fixed length ~297):
        - 32: Event type (one-hot hash)
        - 64: Actor login (one-hot hash)
        - 64: Repo name (one-hot hash)
        - 32: Org login (one-hot hash, if present)
        - 3: Actor behavior (decayed count, unique repos, actor ID pattern)
        - 2: Repo activity (decayed count, unique actors)
        - 100: Event-specific features (hashed actions, text, etc.)
              Note: Reduced from 170 due to removal of non-existent PushEvent fields

        Args:
            event: GitHub Event object from models.py

        Returns:
            Numpy array of numerical features (z-scored if normalize=True),
            or None if event is from a bot and filter_bots=True
        """
        actor_login = event.actor.login

        # Filter bots if enabled
        if self.filter_bots and self.is_bot(actor_login):
            return None

        # Increment event counter for lazy decay
        self.total_events += 1

        # Preallocate feature array (32 + 64 + 3 + 64 + 2 + 32 + 100 = 297)
        repo_name = event.repo.name

        # Calculate feature dimensions
        feature_size = (
            self.categorical_dims["event_type"]  # 32
            + self.categorical_dims["actor"]  # 64
            + 3  # actor behavior
            + self.categorical_dims["repo"]  # 64
            + 2  # repo activity
            + self.categorical_dims["org"]  # 32
            + 100  # type-specific (reduced from 170 due to PushEvent field removal)
        )
        features = np.zeros(feature_size, dtype=float)
        idx = 0

        # === EVENT TYPE (32, one-hot hash) ===
        event_type_vec = self._hash_categorical(
            event.type, self.categorical_dims["event_type"], seed=1
        )
        features[idx : idx + len(event_type_vec)] = event_type_vec
        idx += len(event_type_vec)

        # === ACTOR FEATURES ===
        # One-hot hash (64)
        actor_vec = self._hash_categorical(
            actor_login, self.categorical_dims["actor"], seed=2
        )
        features[idx : idx + len(actor_vec)] = actor_vec
        idx += len(actor_vec)

        # Get decayed count (lazy decay)
        if actor_login in self.actor_event_counts:
            decayed_count = self._get_decayed_count(
                self.actor_event_counts[actor_login],
                self.actor_last_seen[actor_login],
            )
        else:
            decayed_count = 0.0

        # Update actor tracking with LRU eviction
        if actor_login in self.actor_event_counts:
            # Move to end (mark as recently used)
            self.actor_event_counts.move_to_end(actor_login)
        self.actor_event_counts[actor_login] = decayed_count + 1.0
        self.actor_last_seen[actor_login] = self.total_events

        # Update sets/lists
        if actor_login not in self.actor_repos:
            self.actor_repos[actor_login] = set()
        self.actor_repos[actor_login].add(repo_name)
        if actor_login not in self.actor_event_types:
            self.actor_event_types[actor_login] = []
        self.actor_event_types[actor_login].append(event.type)

        # Evict LRU if needed
        self._evict_lru_actors()

        # Actor behavioral features (3)
        features[idx] = decayed_count + 1.0  # Current decayed count
        features[idx + 1] = len(self.actor_repos[actor_login])
        features[idx + 2] = event.actor.id % 10000
        idx += 3

        # === REPOSITORY FEATURES ===
        # One-hot hash (64)
        repo_vec = self._hash_categorical(
            repo_name, self.categorical_dims["repo"], seed=3
        )
        features[idx : idx + len(repo_vec)] = repo_vec
        idx += len(repo_vec)

        # Get decayed count (lazy decay)
        if repo_name in self.repo_event_counts:
            repo_decayed_count = self._get_decayed_count(
                self.repo_event_counts[repo_name],
                self.repo_last_seen[repo_name],
            )
        else:
            repo_decayed_count = 0.0

        # Update repo tracking with LRU eviction
        if repo_name in self.repo_event_counts:
            self.repo_event_counts.move_to_end(repo_name)
        self.repo_event_counts[repo_name] = repo_decayed_count + 1.0
        self.repo_last_seen[repo_name] = self.total_events

        if repo_name not in self.repo_actors:
            self.repo_actors[repo_name] = set()
        self.repo_actors[repo_name].add(actor_login)

        # Evict LRU if needed
        self._evict_lru_repos()

        # Repo activity features (2)
        features[idx] = repo_decayed_count + 1.0
        features[idx + 1] = len(self.repo_actors[repo_name])
        idx += 2

        # === ORGANIZATION (32, one-hot hash) ===
        org_login = event.org.login if event.org else ""
        org_vec = self._hash_categorical(
            org_login, self.categorical_dims["org"], seed=4
        )
        features[idx : idx + len(org_vec)] = org_vec
        idx += len(org_vec)

        # === EVENT-SPECIFIC FEATURES (170) ===
        type_features = self._extract_type_specific_features(event)
        features[idx : idx + len(type_features)] = type_features

        # Batch normalization updates (every N events)
        if self.normalize and self.total_events % self.normalization_frequency == 0:
            self._update_welford(features)
            features = self._normalize_features(features)
        elif self.normalize and self.feature_mean is not None:
            # Apply normalization without updating stats
            features = self._normalize_features(features)

        return features

    def _extract_type_specific_features(self, event: Event) -> np.ndarray:
        """Extract features specific to each event type using proper encoding.

        Returns exactly 100 features (padded with 0s if event type has fewer).
        Reduced from 170 due to removal of non-existent PushEvent fields.
        Uses one-hot hashing for actions and text n-gram hashing for text fields.
        """
        p = event.payload
        features = np.zeros(100, dtype=float)  # Preallocate (was 170)
        idx = 0
        action_dim = self.categorical_dims["action"]  # 16

        if event.type == "PushEvent":
            # Action hash (always 0 for PushEvent, no action field)
            idx += action_dim  # Skip zeros

            ref = p.get("ref", "")

            # Scalar features (2) - only using real API fields
            # Note: GitHub Events API does NOT provide size, distinct_size, or commits
            features[idx] = (
                1.0 if ref.endswith("/main") or ref.endswith("/master") else 0.0
            )
            features[idx + 1] = ref.count("/")
            # Removed: size, distinct_size, commit count, avg message length,
            # empty push flag, distinct size comparison, commit message hashing
            # Total removed: 6 scalar + 64 text = 70 features

        elif event.type == "PullRequestEvent":
            action = p.get("action", "")
            action_vec = self._hash_categorical(action, action_dim, seed=5)
            features[idx : idx + len(action_vec)] = action_vec
            idx += action_dim

            pr = p.get("pull_request", {})

            # Scalar features (7)
            features[idx] = 1.0 if action == "opened" else 0.0
            features[idx + 1] = 1.0 if action == "closed" else 0.0
            features[idx + 2] = 1.0 if pr.get("merged", False) else 0.0
            features[idx + 3] = len(pr.get("assignees", []) or [])
            features[idx + 4] = len(pr.get("labels", []) or [])
            features[idx + 5] = len(pr.get("title", ""))
            features[idx + 6] = len(pr.get("body") or "")
            idx += 7

            # PR title n-gram hashing (32 features)
            title_vec = self._hash_text_ngrams(
                pr.get("title", ""), self.text_dims["titles"], seed=11
            )
            features[idx : idx + len(title_vec)] = title_vec
            idx += len(title_vec)

            # PR body n-gram hashing (32 features)
            body_vec = self._hash_text_ngrams(
                pr.get("body") or "", self.text_dims["bodies"], seed=12
            )
            features[idx : idx + len(body_vec)] = body_vec
            # Auto-padded to 170

        elif event.type == "IssuesEvent":
            action = p.get("action", "")
            action_vec = self._hash_categorical(action, action_dim, seed=5)
            features[idx : idx + len(action_vec)] = action_vec
            idx += action_dim

            issue = p.get("issue", {})

            # Scalar features (7)
            features[idx] = 1.0 if action == "opened" else 0.0
            features[idx + 1] = 1.0 if action == "closed" else 0.0
            features[idx + 2] = 1.0 if issue.get("state") == "open" else 0.0
            features[idx + 3] = len(issue.get("labels", []) or [])
            features[idx + 4] = len(issue.get("assignees", []) or [])
            features[idx + 5] = len(issue.get("title", ""))
            features[idx + 6] = len(issue.get("body") or "")
            idx += 7

            # Issue title n-gram hashing (32 features)
            title_vec = self._hash_text_ngrams(
                issue.get("title", ""), self.text_dims["titles"], seed=13
            )
            features[idx : idx + len(title_vec)] = title_vec
            idx += len(title_vec)

            # Issue body n-gram hashing (32 features)
            body_vec = self._hash_text_ngrams(
                issue.get("body") or "", self.text_dims["bodies"], seed=14
            )
            features[idx : idx + len(body_vec)] = body_vec

        elif event.type == "MemberEvent":
            # SECURITY CRITICAL - permission changes
            action = p.get("action", "")
            action_vec = self._hash_categorical(action, action_dim, seed=5)
            features[idx : idx + len(action_vec)] = action_vec
            idx += action_dim

            features[idx] = 1.0 if action == "added" else 0.0  # High risk!
            features[idx + 1] = 1.0 if action == "deleted" else 0.0
            features[idx + 2] = 1.0 if action == "edited" else 0.0

        elif event.type == "DeleteEvent":
            # DESTRUCTIVE - deletions
            idx += action_dim  # Skip zeros

            ref_type = p.get("ref_type", "")
            features[idx] = 1.0 if ref_type == "branch" else 0.0
            features[idx + 1] = 1.0 if ref_type == "tag" else 0.0

        elif event.type == "CreateEvent":
            idx += action_dim  # Skip zeros

            ref_type = p.get("ref_type", "")
            features[idx] = 1.0 if ref_type == "branch" else 0.0
            features[idx + 1] = 1.0 if ref_type == "tag" else 0.0
            features[idx + 2] = 1.0 if ref_type == "repository" else 0.0

        elif event.type == "ReleaseEvent":
            action = p.get("action", "")
            action_vec = self._hash_categorical(action, action_dim, seed=5)
            features[idx : idx + len(action_vec)] = action_vec
            idx += action_dim

            release = p.get("release", {})
            features[idx] = 1.0 if action == "published" else 0.0
            features[idx + 1] = 1.0 if release.get("draft", False) else 0.0
            features[idx + 2] = 1.0 if release.get("prerelease", False) else 0.0
            features[idx + 3] = len(release.get("body") or "")
            features[idx + 4] = len(release.get("name") or "")
            idx += 5

            # Release body n-gram hashing (32 features)
            body_vec = self._hash_text_ngrams(
                release.get("body") or "", self.text_dims["bodies"], seed=15
            )
            features[idx : idx + len(body_vec)] = body_vec

        elif event.type == "ForkEvent":
            idx += action_dim  # Skip zeros

            forkee = p.get("forkee", {})
            features[idx] = 1.0 if forkee.get("private", False) else 0.0
            features[idx + 1] = 1.0 if forkee.get("fork", False) else 0.0  # Fork of fork

        elif event.type == "IssueCommentEvent":
            action = p.get("action", "")
            action_vec = self._hash_categorical(action, action_dim, seed=5)
            features[idx : idx + len(action_vec)] = action_vec
            idx += action_dim

            comment = p.get("comment", {})
            features[idx] = len(comment.get("body", ""))
            features[idx + 1] = 1.0 if action == "created" else 0.0
            idx += 2

            # Comment body n-gram hashing (32 features)
            body_vec = self._hash_text_ngrams(
                comment.get("body", ""), self.text_dims["comments"], seed=16
            )
            features[idx : idx + len(body_vec)] = body_vec

        elif event.type == "CommitCommentEvent":
            action = p.get("action", "")
            action_vec = self._hash_categorical(action, action_dim, seed=5)
            features[idx : idx + len(action_vec)] = action_vec
            idx += action_dim

            comment = p.get("comment", {})
            features[idx] = len(comment.get("body", ""))
            features[idx + 1] = 1.0 if comment.get("path") else 0.0
            features[idx + 2] = 1.0 if comment.get("line") else 0.0
            idx += 3

            # Comment body n-gram hashing (32 features)
            body_vec = self._hash_text_ngrams(
                comment.get("body", ""), self.text_dims["comments"], seed=17
            )
            features[idx : idx + len(body_vec)] = body_vec

        elif event.type == "GollumEvent":
            # Wiki changes
            idx += action_dim  # Skip zeros

            pages = p.get("pages", [])
            features[idx] = len(pages)
            features[idx + 1] = (
                1.0 if any(pg.get("action") == "created" for pg in pages) else 0.0
            )

        elif event.type == "PublicEvent":
            # SECURITY RISK - repo made public
            idx += action_dim  # Skip zeros
            features[idx] = 1.0  # Flag this event type

        elif event.type == "WatchEvent":
            action = p.get("action", "")
            action_vec = self._hash_categorical(action, action_dim, seed=5)
            features[idx : idx + len(action_vec)] = action_vec
            idx += action_dim
            features[idx] = 1.0  # Star event

        elif event.type == "PullRequestReviewEvent":
            action = p.get("action", "")
            action_vec = self._hash_categorical(action, action_dim, seed=5)
            features[idx : idx + len(action_vec)] = action_vec
            idx += action_dim

            review = p.get("review", {})
            state = review.get("state", "")
            features[idx] = 1.0 if state == "approved" else 0.0
            features[idx + 1] = 1.0 if state == "changes_requested" else 0.0
            features[idx + 2] = len(review.get("body") or "")
            idx += 3

            # Review body n-gram hashing (32 features)
            body_vec = self._hash_text_ngrams(
                review.get("body") or "", self.text_dims["bodies"], seed=18
            )
            features[idx : idx + len(body_vec)] = body_vec

        # Unknown event types already have zeros (preallocated)
        return features

    def get_actor_stats(self, actor_login: str) -> dict[str, Any]:
        """Get statistics for a specific actor.

        Args:
            actor_login: Actor username

        Returns:
            Dictionary with actor statistics (counts are decayed)
        """
        from collections import Counter

        # Compute decayed count
        if actor_login in self.actor_event_counts:
            decayed_count = self._get_decayed_count(
                self.actor_event_counts[actor_login],
                self.actor_last_seen[actor_login],
            )
        else:
            decayed_count = 0.0

        return {
            "total_events": decayed_count,
            "unique_repos": len(self.actor_repos.get(actor_login, set())),
            "event_types": dict(Counter(self.actor_event_types.get(actor_login, []))),
        }

    def get_repo_stats(self, repo_name: str) -> dict[str, Any]:
        """Get statistics for a specific repository.

        Args:
            repo_name: Repository name

        Returns:
            Dictionary with repo statistics (counts are decayed)
        """
        # Compute decayed count
        if repo_name in self.repo_event_counts:
            decayed_count = self._get_decayed_count(
                self.repo_event_counts[repo_name],
                self.repo_last_seen[repo_name],
            )
        else:
            decayed_count = 0.0

        return {
            "total_events": decayed_count,
            "unique_actors": len(self.repo_actors.get(repo_name, set())),
        }

    def reset_stats(self) -> None:
        """Reset all tracking statistics."""
        self.actor_event_counts.clear()
        self.actor_last_seen.clear()
        self.actor_repos.clear()
        self.actor_event_types.clear()
        self.repo_event_counts.clear()
        self.repo_last_seen.clear()
        self.repo_actors.clear()
        self.total_events = 0
        self.feature_count = 0
        self.feature_mean = None
        self.feature_m2 = None


def get_suspicious_patterns(
    event: Event, extractor: GitHubFeatureExtractor
) -> list[str]:
    """Identify suspicious patterns in an event.

    Args:
        event: GitHub Event object
        extractor: Feature extractor with tracking state

    Returns:
        List of suspicious pattern descriptions
    """
    patterns = []
    actor_login = event.actor.login
    is_bot = actor_login in KNOWN_BOTS

    actor_stats = extractor.get_actor_stats(actor_login)

    # High velocity (skip for bots - they're expected to have high activity)
    if not is_bot and actor_stats["total_events"] > 50:
        patterns.append(
            f"High velocity: {actor_stats['total_events']:.1f} decayed events from {actor_login}"
        )

    # Repo hopping (skip for bots)
    if not is_bot and actor_stats["unique_repos"] > 20:
        patterns.append(
            f"Repo hopping: {actor_stats['unique_repos']} different repos"
        )

    # Destructive actions (ALWAYS flag - these should always alert)
    if event.type in ["DeleteEvent", "DestroyEvent"]:
        patterns.append(f"Destructive action: {event.type}")

    # Security-critical actions (removed from auto-patterns, will be caught by severity logic)
    # MemberEvent and PublicEvent no longer auto-generate patterns

    # Note: Large push and empty push detection removed because GitHub Events API
    # does not provide the 'size' field required for detection

    return patterns
