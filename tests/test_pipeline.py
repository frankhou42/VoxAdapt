from pathlib import Path

import pytest

from voxadapt.asr import FilenameTranscriber
from voxadapt.bandit import LinUCBBandit
from voxadapt.models import Channel, RewriteMode
from voxadapt.personalization import ProfileStore
from voxadapt.pipeline import VoicePipeline
from voxadapt.text import RuleBasedPolisher


def make_pipeline(tmp_path, *, with_transcriber: bool = False) -> VoicePipeline:
    return VoicePipeline(
        polisher=RuleBasedPolisher(),
        profiles=ProfileStore(tmp_path / "profiles.db"),
        bandit=LinUCBBandit(alpha=0.1),
        transcriber=FilenameTranscriber() if with_transcriber else None,
    )


def test_pipeline_records_latency_and_feedback(tmp_path) -> None:
    pipeline = make_pipeline(tmp_path)
    result = pipeline.process_text(
        user_id="frank", transcript="um send this to wisper", channel=Channel.EMAIL
    )
    assert result.rewrite_mode is RewriteMode.BALANCED
    assert result.polished_text == "Send this to wisper."
    reward = pipeline.record_feedback(
        interaction_id=result.interaction_id,
        user_id="frank",
        raw_text=result.raw_text,
        suggested_text=result.polished_text,
        final_text="Send this to Wispr.",
        accepted=False,
        channel=Channel.EMAIL,
    )
    assert 0.0 < reward < 0.8
    personalized = pipeline.process_text(
        user_id="frank", transcript="send this to wisper", channel=Channel.EMAIL
    )
    assert personalized.polished_text == "Send this to Wispr."
    assert pipeline.metrics.snapshot()["total"]["count"] == 2


def test_audio_requires_configured_transcriber(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="No transcriber"):
        make_pipeline(tmp_path).process_audio(
            user_id="frank", audio_path=Path("test.wav"), channel=Channel.CHAT
        )


def test_audio_path_includes_asr_timing(tmp_path) -> None:
    result = make_pipeline(tmp_path, with_transcriber=True).process_audio(
        user_id="frank",
        audio_path=Path("um_send_the_report.wav"),
        channel=Channel.EMAIL,
    )
    assert result.raw_text == "um send the report"
    assert result.timings[0].name == "asr"
