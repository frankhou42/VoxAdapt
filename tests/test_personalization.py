from voxadapt.models import Channel
from voxadapt.personalization import ProfileStore, normalized_similarity


def test_profile_learns_proper_noun_and_channel_style(tmp_path) -> None:
    store = ProfileStore(tmp_path / "profiles.db")
    store.add_correction(
        user_id="frank",
        channel=Channel.CHAT,
        raw_text="send this to wisper",
        suggested_text="Send this to wisper",
        final_text="send this to Wispr",
        accepted=False,
    )
    profile = store.get_profile("frank")
    assert profile.replacements["wisper"] == "Wispr"
    assert profile.channel_preferences["chat"]["lowercase"] is True
    assert profile.channel_preferences["chat"]["terminal_punctuation"] is False


def test_user_data_can_be_deleted(tmp_path) -> None:
    store = ProfileStore(tmp_path / "profiles.db")
    store.add_correction(
        user_id="frank",
        channel=Channel.EMAIL,
        raw_text="raw",
        suggested_text="Suggested.",
        final_text="Final.",
        accepted=False,
    )
    assert store.delete_user("frank") == 1
    assert store.get_profile("frank").examples == []


def test_normalized_similarity_ignores_case_and_edges() -> None:
    assert normalized_similarity(" Hello ", "hello") == 1.0
