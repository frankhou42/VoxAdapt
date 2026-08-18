from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...


class FasterWhisperTranscriber:
    """Lazy faster-whisper adapter supporting CPU INT8 and Apple Silicon CPU execution."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install VoxAdapt with the 'asr' extra") from exc
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path) -> str:
        segments, _ = self._model.transcribe(
            str(audio_path),
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class FilenameTranscriber:
    """Deterministic adapter for development: the filename stem is the transcript."""

    def transcribe(self, audio_path: Path) -> str:
        return audio_path.stem.replace("_", " ")
