"""Episodic memory for cross-task learning."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from nexagen.constants import DEFAULT_MAX_EPISODES

# Same patterns used by BaseTool.execute() for error sanitization
_PATH_RE = re.compile(r'(/[a-zA-Z0-9._-]+){2,}')
_SECRET_RE = re.compile(r'[a-zA-Z0-9+/=_-]{32,}')


@dataclass
class Episode:
    """A single recorded episode from a completed task."""

    task: str
    outcome: str  # "success" | "failure" | "partial"
    tools_used: list[str]
    errors_encountered: list[str]
    reflections: list[str]
    timestamp: float = field(default_factory=time.time)


class EpisodicMemory:
    """Stores and retrieves past episodes for cross-task learning.

    Retrieval scoring: recency (0.3) + keyword_relevance (0.7).
    """

    def __init__(self, max_episodes: int = DEFAULT_MAX_EPISODES) -> None:
        self.max_episodes = max_episodes
        self._episodes: list[Episode] = []

    @staticmethod
    def _sanitize(text: str) -> str:
        """Strip file paths and potential secrets from text before storage."""
        text = _PATH_RE.sub('<path-redacted>', text)
        text = _SECRET_RE.sub('<redacted>', text)
        return text

    def record(self, episode: Episode) -> None:
        """Append an episode, evicting the oldest if over capacity.

        Sanitizes error and reflection strings to prevent sensitive data
        from leaking into future system prompts.
        """
        episode.errors_encountered = [
            self._sanitize(e) for e in episode.errors_encountered
        ]
        episode.reflections = [
            self._sanitize(r) for r in episode.reflections
        ]
        self._episodes.append(episode)
        if len(self._episodes) > self.max_episodes:
            self._episodes.pop(0)

    def retrieve(self, query: str, k: int = 3) -> list[Episode]:
        """Return the k most relevant episodes scored by recency and keyword overlap."""
        if not self._episodes:
            return []

        scored: list[tuple[float, Episode]] = []
        for ep in self._episodes:
            relevance = self._keyword_relevance(query, ep)
            recency = self._recency_score(ep)
            score = 0.7 * relevance + 0.3 * recency
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def format_for_context(self, episodes: list[Episode]) -> str:
        """Format episodes as readable text for system prompt injection."""
        if not episodes:
            return ""

        parts: list[str] = []
        for i, ep in enumerate(episodes, 1):
            lines = [
                f"Episode {i}:",
                f"  Task: {ep.task}",
                f"  Outcome: {ep.outcome}",
                f"  Tools used: {', '.join(ep.tools_used) if ep.tools_used else 'none'}",
            ]
            if ep.errors_encountered:
                lines.append(f"  Errors: {', '.join(ep.errors_encountered)}")
            if ep.reflections:
                lines.append(f"  Reflections: {'; '.join(ep.reflections)}")
            parts.append("\n".join(lines))

        result = "\n\n".join(parts)
        # Cap total output to prevent context window exhaustion
        if len(result) > 2000:
            return result[:2000] + "\n[...truncated]"
        return result

    def _keyword_relevance(self, query: str, episode: Episode) -> float:
        """Word overlap score between query and all episode text fields."""
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0

        episode_text = " ".join([
            episode.task,
            episode.outcome,
            " ".join(episode.tools_used),
            " ".join(episode.errors_encountered),
            " ".join(episode.reflections),
        ])
        episode_words = set(episode_text.lower().split())

        overlap = query_words & episode_words
        return len(overlap) / len(query_words)

    def _recency_score(self, episode: Episode) -> float:
        """Normalized 0-1 score based on newest/oldest timestamps."""
        if len(self._episodes) <= 1:
            return 1.0

        timestamps = [ep.timestamp for ep in self._episodes]
        oldest = min(timestamps)
        newest = max(timestamps)

        if newest == oldest:
            return 1.0

        return (episode.timestamp - oldest) / (newest - oldest)
