from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RewriteMode(StrEnum):
    VERBATIM = "verbatim"
    BALANCED = "balanced"
    POLISHED = "polished"


class Channel(StrEnum):
    CHAT = "chat"
    EMAIL = "email"
    NOTES = "notes"


class TranscriptRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    transcript: str = Field(min_length=1, max_length=20_000)
    channel: Channel = Channel.CHAT


class FeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    interaction_id: str = Field(min_length=1, max_length=128)
    raw_text: str = Field(min_length=1, max_length=20_000)
    suggested_text: str = Field(min_length=1, max_length=20_000)
    final_text: str = Field(min_length=1, max_length=20_000)
    accepted: bool
    channel: Channel = Channel.CHAT


class StageTiming(BaseModel):
    name: str
    milliseconds: float = Field(ge=0)


class TranscriptResponse(BaseModel):
    interaction_id: str
    raw_text: str
    polished_text: str
    rewrite_mode: RewriteMode
    timings: list[StageTiming]
    total_milliseconds: float = Field(ge=0)


@dataclass(slots=True)
class UserProfile:
    user_id: str
    replacements: dict[str, str] = field(default_factory=dict)
    examples: list[tuple[str, str]] = field(default_factory=list)
    channel_preferences: dict[str, dict[str, Any]] = field(default_factory=dict)
