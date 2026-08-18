from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .asr import FasterWhisperTranscriber
from .bandit import LinUCBBandit
from .metrics import LatencyTracker
from .models import Channel, FeedbackRequest, TranscriptRequest, TranscriptResponse
from .personalization import ProfileStore
from .pipeline import VoicePipeline
from .text import RuleBasedPolisher, TransformersPolisher


def create_pipeline() -> VoicePipeline:
    database_path = os.getenv("VOXADAPT_DB", "voxadapt.db")
    model_path = os.getenv("VOXADAPT_MODEL")
    polisher = TransformersPolisher(model_path) if model_path else RuleBasedPolisher()
    transcriber = None
    if os.getenv("VOXADAPT_ENABLE_ASR", "0") == "1":
        transcriber = FasterWhisperTranscriber(
            model_size=os.getenv("VOXADAPT_ASR_MODEL", "tiny.en"),
            compute_type=os.getenv("VOXADAPT_ASR_COMPUTE", "int8"),
        )
    return VoicePipeline(
        polisher=polisher,
        profiles=ProfileStore(database_path),
        bandit=LinUCBBandit(),
        metrics=LatencyTracker(),
        transcriber=transcriber,
    )


def create_app(pipeline: VoicePipeline | None = None) -> FastAPI:
    app = FastAPI(title="VoxAdapt API", version="0.1.0")
    app.state.pipeline = pipeline or create_pipeline()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[os.getenv("VOXADAPT_WEB_ORIGIN", "http://localhost:5173")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/polish", response_model=TranscriptResponse)
    def polish(request: TranscriptRequest) -> TranscriptResponse:
        return app.state.pipeline.process_text(
            user_id=request.user_id,
            transcript=request.transcript,
            channel=request.channel,
        )

    @app.post("/v1/transcribe", response_model=TranscriptResponse)
    async def transcribe(
        user_id: Annotated[str, Form()],
        audio: Annotated[UploadFile, File()],
        channel: Annotated[Channel, Form()] = Channel.CHAT,
    ) -> TranscriptResponse:
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
            temporary.write(await audio.read())
            temporary.flush()
            try:
                return app.state.pipeline.process_audio(
                    user_id=user_id,
                    audio_path=Path(temporary.name),
                    channel=channel,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/v1/feedback")
    def feedback(request: FeedbackRequest) -> dict[str, float | str]:
        try:
            reward = app.state.pipeline.record_feedback(**request.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown interaction") from exc
        return {"status": "learned", "reward": round(reward, 4)}

    @app.get("/v1/metrics")
    def metrics() -> dict[str, dict[str, float | int]]:
        return app.state.pipeline.metrics.snapshot()

    @app.get("/v1/users/{user_id}/profile")
    def profile(user_id: str) -> dict[str, object]:
        stored = app.state.pipeline.profiles.get_profile(user_id)
        return {
            "user_id": stored.user_id,
            "replacements": stored.replacements,
            "examples": stored.examples,
            "channel_preferences": stored.channel_preferences,
        }

    @app.delete("/v1/users/{user_id}")
    def delete_user(user_id: str) -> dict[str, int]:
        return {"deleted_corrections": app.state.pipeline.profiles.delete_user(user_id)}

    @app.websocket("/v1/stream/{user_id}")
    async def stream(websocket: WebSocket, user_id: str) -> None:
        """Accept byte chunks and transcribe once the client sends a JSON commit event."""
        await websocket.accept()
        channel = Channel.CHAT
        audio = bytearray()
        try:
            while True:
                message = await websocket.receive()
                if message.get("bytes") is not None:
                    audio.extend(message["bytes"])
                    continue
                event = message.get("text", "")
                if event.startswith("channel:"):
                    channel = Channel(event.split(":", 1)[1])
                elif event == "commit":
                    with tempfile.NamedTemporaryFile(suffix=".webm") as temporary:
                        temporary.write(audio)
                        temporary.flush()
                        result = app.state.pipeline.process_audio(
                            user_id=user_id,
                            audio_path=Path(temporary.name),
                            channel=channel,
                        )
                    await websocket.send_json(result.model_dump(mode="json"))
                    audio.clear()
                elif event == "reset":
                    audio.clear()
        except WebSocketDisconnect:
            return

    return app


app = create_app()
