from fastapi.testclient import TestClient

from voxadapt.api import create_app
from voxadapt.bandit import LinUCBBandit
from voxadapt.personalization import ProfileStore
from voxadapt.pipeline import VoicePipeline
from voxadapt.text import RuleBasedPolisher


def test_polish_feedback_and_privacy_endpoints(tmp_path) -> None:
    pipeline = VoicePipeline(
        polisher=RuleBasedPolisher(),
        profiles=ProfileStore(tmp_path / "profiles.db"),
        bandit=LinUCBBandit(alpha=0.1),
    )
    client = TestClient(create_app(pipeline))
    response = client.post(
        "/v1/polish",
        json={"user_id": "frank", "transcript": "um send the report", "channel": "email"},
    )
    assert response.status_code == 200
    interaction = response.json()
    assert interaction["polished_text"] == "Send the report."

    feedback = client.post(
        "/v1/feedback",
        json={
            "user_id": "frank",
            "interaction_id": interaction["interaction_id"],
            "raw_text": interaction["raw_text"],
            "suggested_text": interaction["polished_text"],
            "final_text": "Send the final report.",
            "accepted": False,
            "channel": "email",
        },
    )
    assert feedback.status_code == 200
    assert client.get("/v1/metrics").json()["total"]["count"] == 1
    assert client.delete("/v1/users/frank").json()["deleted_corrections"] == 1


def test_audio_endpoint_is_explicit_when_asr_is_disabled(tmp_path) -> None:
    pipeline = VoicePipeline(
        polisher=RuleBasedPolisher(), profiles=ProfileStore(tmp_path / "profiles.db")
    )
    client = TestClient(create_app(pipeline))
    response = client.post(
        "/v1/transcribe",
        data={"user_id": "frank", "channel": "chat"},
        files={"audio": ("sample.wav", b"not-a-real-wave", "audio/wav")},
    )
    assert response.status_code == 503
    assert "No transcriber" in response.json()["detail"]
