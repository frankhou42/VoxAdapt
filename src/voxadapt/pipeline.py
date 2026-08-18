from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .asr import Transcriber
from .bandit import BanditChoice, LinUCBBandit
from .metrics import LatencyTracker
from .models import Channel, StageTiming, TranscriptResponse
from .personalization import ProfileStore, normalized_similarity
from .text import TextPolisher


@dataclass(slots=True)
class InteractionState:
    choice: BanditChoice


class VoicePipeline:
    def __init__(
        self,
        *,
        polisher: TextPolisher,
        profiles: ProfileStore,
        bandit: LinUCBBandit | None = None,
        metrics: LatencyTracker | None = None,
        transcriber: Transcriber | None = None,
    ) -> None:
        self.polisher = polisher
        self.profiles = profiles
        self.bandit = bandit or LinUCBBandit()
        self.metrics = metrics or LatencyTracker()
        self.transcriber = transcriber
        self._interactions: dict[str, InteractionState] = {}

    def process_text(
        self, *, user_id: str, transcript: str, channel: Channel
    ) -> TranscriptResponse:
        total_start = time.perf_counter_ns()

        profile_start = time.perf_counter_ns()
        profile = self.profiles.get_profile(user_id)
        profile_ms = _elapsed_ms(profile_start)

        choice_start = time.perf_counter_ns()
        choice = self.bandit.choose(transcript, channel)
        choice_ms = _elapsed_ms(choice_start)

        polish_start = time.perf_counter_ns()
        polished = self.polisher.polish(
            transcript,
            mode=choice.arm,
            channel=channel,
            profile=profile,
        )
        polish_ms = _elapsed_ms(polish_start)

        interaction_id = uuid.uuid4().hex
        self._interactions[interaction_id] = InteractionState(choice=choice)
        total_ms = _elapsed_ms(total_start)
        timings = [
            StageTiming(name="profile", milliseconds=profile_ms),
            StageTiming(name="bandit", milliseconds=choice_ms),
            StageTiming(name="polish", milliseconds=polish_ms),
        ]
        for timing in timings:
            self.metrics.observe(timing.name, timing.milliseconds)
        self.metrics.observe("total", total_ms)
        return TranscriptResponse(
            interaction_id=interaction_id,
            raw_text=transcript,
            polished_text=polished,
            rewrite_mode=choice.arm,
            timings=timings,
            total_milliseconds=total_ms,
        )

    def process_audio(
        self, *, user_id: str, audio_path: Path, channel: Channel
    ) -> TranscriptResponse:
        if self.transcriber is None:
            raise RuntimeError("No transcriber configured")
        start = time.perf_counter_ns()
        transcript = self.transcriber.transcribe(audio_path)
        asr_ms = _elapsed_ms(start)
        result = self.process_text(user_id=user_id, transcript=transcript, channel=channel)
        result.timings.insert(0, StageTiming(name="asr", milliseconds=asr_ms))
        result.total_milliseconds += asr_ms
        self.metrics.observe("asr", asr_ms)
        return result

    def record_feedback(
        self,
        *,
        interaction_id: str,
        user_id: str,
        raw_text: str,
        suggested_text: str,
        final_text: str,
        accepted: bool,
        channel: Channel,
    ) -> float:
        state = self._interactions.pop(interaction_id, None)
        if state is None:
            raise KeyError(interaction_id)
        similarity = normalized_similarity(suggested_text, final_text)
        reward = 1.0 if accepted else 0.8 * similarity
        self.bandit.update(state.choice.arm, state.choice.features, reward)
        self.profiles.add_correction(
            user_id=user_id,
            channel=channel,
            raw_text=raw_text,
            suggested_text=suggested_text,
            final_text=final_text,
            accepted=accepted,
        )
        return reward


def _elapsed_ms(start_ns: int) -> float:
    return round((time.perf_counter_ns() - start_ns) / 1_000_000, 3)
