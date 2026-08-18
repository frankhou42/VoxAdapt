from __future__ import annotations

import difflib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .models import Channel, UserProfile


class ProfileStore:
    """SQLite-backed correction memory with deliberately inspectable state."""

    def __init__(self, path: str | Path = "voxadapt.db") -> None:
        self.path = str(path)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    suggested_text TEXT NOT NULL,
                    final_text TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS corrections_user_id_idx
                    ON corrections(user_id, id DESC);
                """
            )

    def add_correction(
        self,
        *,
        user_id: str,
        channel: Channel,
        raw_text: str,
        suggested_text: str,
        final_text: str,
        accepted: bool,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO corrections (
                    user_id, channel, raw_text, suggested_text, final_text, accepted
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, channel.value, raw_text, suggested_text, final_text, int(accepted)),
            )

    def get_profile(self, user_id: str, limit: int = 50) -> UserProfile:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT channel, raw_text, final_text
                FROM corrections
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        replacements: dict[str, str] = {}
        examples: list[tuple[str, str]] = []
        punctuation_by_channel: dict[str, list[bool]] = {}
        lowercase_by_channel: dict[str, list[bool]] = {}
        for row in reversed(rows):
            raw_text, final_text, channel = row["raw_text"], row["final_text"], row["channel"]
            examples.append((raw_text, final_text))
            replacements.update(_extract_replacements(raw_text, final_text))
            punctuation_by_channel.setdefault(channel, []).append(
                final_text.endswith((".", "!", "?"))
            )
            lowercase_by_channel.setdefault(channel, []).append(
                bool(final_text) and final_text[0].islower()
            )

        preferences = {
            channel: {
                "terminal_punctuation": sum(punctuation_by_channel[channel])
                >= len(punctuation_by_channel[channel]) / 2,
                "lowercase": sum(lowercase_by_channel[channel])
                >= len(lowercase_by_channel[channel]) / 2,
            }
            for channel in punctuation_by_channel
        }
        return UserProfile(
            user_id=user_id,
            replacements=replacements,
            examples=examples[-5:],
            channel_preferences=preferences,
        )

    def export_user(self, user_id: str) -> str:
        profile = self.get_profile(user_id)
        return json.dumps(
            {
                "user_id": profile.user_id,
                "replacements": profile.replacements,
                "examples": profile.examples,
                "channel_preferences": profile.channel_preferences,
            },
            indent=2,
        )

    def delete_user(self, user_id: str) -> int:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM corrections WHERE user_id = ?", (user_id,))
            return cursor.rowcount


def _extract_replacements(raw_text: str, final_text: str) -> dict[str, str]:
    raw_words = raw_text.split()
    final_words = final_text.split()
    matcher = difflib.SequenceMatcher(a=[word.lower() for word in raw_words], b=final_words)
    replacements: dict[str, str] = {}
    for operation, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if operation == "replace" and a_end - a_start <= 3 and b_end - b_start <= 3:
            source = " ".join(raw_words[a_start:a_end]).strip(".,!? ").lower()
            target = " ".join(final_words[b_start:b_end]).strip()
            if source and target and source != target.lower():
                replacements[source] = target
    return replacements


def normalized_similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(a=left.strip().lower(), b=right.strip().lower()).ratio()
