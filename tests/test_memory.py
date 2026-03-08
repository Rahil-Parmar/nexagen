"""Tests for episodic memory — cross-task learning."""

import time

import pytest

from nexagen.memory import Episode, EpisodicMemory


class TestEpisodeCreation:
    def test_episode_creation(self):
        ep = Episode(
            task="Parse CSV file",
            outcome="success",
            tools_used=["read_file", "write_file"],
            errors_encountered=[],
            reflections=["CSV parsing went smoothly"],
        )
        assert ep.task == "Parse CSV file"
        assert ep.outcome == "success"
        assert ep.tools_used == ["read_file", "write_file"]
        assert ep.errors_encountered == []
        assert ep.reflections == ["CSV parsing went smoothly"]
        assert isinstance(ep.timestamp, float)
        assert ep.timestamp <= time.time()

    def test_episode_custom_timestamp(self):
        ep = Episode(
            task="test",
            outcome="failure",
            tools_used=[],
            errors_encountered=["timeout"],
            reflections=[],
            timestamp=1000.0,
        )
        assert ep.timestamp == 1000.0


class TestRecordAndRetrieve:
    def test_record_and_retrieve(self):
        mem = EpisodicMemory(max_episodes=10)
        ep = Episode(
            task="Summarize document",
            outcome="success",
            tools_used=["read_file"],
            errors_encountered=[],
            reflections=["Used chunking strategy"],
        )
        mem.record(ep)
        results = mem.retrieve("summarize document", k=3)
        assert len(results) == 1
        assert results[0] is ep

    def test_retrieve_relevance_scoring(self):
        """CSV-related query should find CSV episodes over unrelated ones."""
        mem = EpisodicMemory(max_episodes=10)

        csv_episode = Episode(
            task="Parse CSV data file",
            outcome="success",
            tools_used=["read_file", "csv_parser"],
            errors_encountered=["CSV header mismatch"],
            reflections=["Handle CSV encoding carefully"],
            timestamp=time.time() - 10,
        )
        unrelated_episode = Episode(
            task="Send email notification",
            outcome="success",
            tools_used=["smtp_client"],
            errors_encountered=[],
            reflections=["Email delivered"],
            timestamp=time.time() - 5,
        )
        mem.record(csv_episode)
        mem.record(unrelated_episode)

        results = mem.retrieve("CSV file parsing errors", k=1)
        assert len(results) == 1
        assert results[0] is csv_episode

    def test_eviction_when_over_capacity(self):
        mem = EpisodicMemory(max_episodes=3)
        episodes = []
        for i in range(5):
            ep = Episode(
                task=f"Task {i}",
                outcome="success",
                tools_used=[],
                errors_encountered=[],
                reflections=[],
                timestamp=float(i),
            )
            mem.record(ep)
            episodes.append(ep)

        # Only 3 most recent should remain (tasks 2, 3, 4)
        assert len(mem._episodes) == 3
        assert episodes[0] not in mem._episodes
        assert episodes[1] not in mem._episodes
        assert episodes[4] in mem._episodes

    def test_retrieve_empty_memory(self):
        mem = EpisodicMemory()
        results = mem.retrieve("anything", k=5)
        assert results == []

    def test_retrieve_k_larger_than_stored(self):
        mem = EpisodicMemory()
        ep = Episode(
            task="Only task",
            outcome="partial",
            tools_used=["tool_a"],
            errors_encountered=["some error"],
            reflections=["learned something"],
        )
        mem.record(ep)
        results = mem.retrieve("task", k=10)
        assert len(results) == 1
        assert results[0] is ep


class TestFormatForContext:
    def test_format_for_context(self):
        mem = EpisodicMemory()
        ep = Episode(
            task="Debug API endpoint",
            outcome="failure",
            tools_used=["http_client", "debugger"],
            errors_encountered=["ConnectionTimeout", "InvalidJSON"],
            reflections=["Check timeout settings first"],
        )
        text = mem.format_for_context([ep])
        assert "Debug API endpoint" in text
        assert "failure" in text
        assert "ConnectionTimeout" in text or "InvalidJSON" in text

    def test_format_for_context_empty(self):
        mem = EpisodicMemory()
        text = mem.format_for_context([])
        assert text == ""


class TestSanitization:
    def test_record_strips_paths_from_errors(self):
        mem = EpisodicMemory()
        ep = Episode(
            task="test",
            outcome="failure",
            tools_used=[],
            errors_encountered=["FileNotFoundError: /Users/secret/project/config.env"],
            reflections=[],
        )
        mem.record(ep)
        stored = mem._episodes[0]
        assert "/Users/secret" not in stored.errors_encountered[0]
        assert "<path-redacted>" in stored.errors_encountered[0]

    def test_record_strips_secrets_from_errors(self):
        mem = EpisodicMemory()
        ep = Episode(
            task="test",
            outcome="failure",
            tools_used=[],
            errors_encountered=["API key: sk_live_abcdefghijklmnopqrstuvwxyz123456"],
            reflections=[],
        )
        mem.record(ep)
        stored = mem._episodes[0]
        assert "sk_live_abcdefghijklmnopqrstuvwxyz123456" not in stored.errors_encountered[0]
        assert "<redacted>" in stored.errors_encountered[0]

    def test_record_strips_paths_from_reflections(self):
        mem = EpisodicMemory()
        ep = Episode(
            task="test",
            outcome="failure",
            tools_used=[],
            errors_encountered=[],
            reflections=["Found issue in /home/user/app/secret/key.pem"],
        )
        mem.record(ep)
        stored = mem._episodes[0]
        assert "/home/user" not in stored.reflections[0]
