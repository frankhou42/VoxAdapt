from voxadapt.models import Channel, RewriteMode, UserProfile
from voxadapt.text import RuleBasedPolisher, disfluency_ratio


def test_balanced_polisher_removes_fillers_repeats_and_formats_email() -> None:
    result = RuleBasedPolisher().polish(
        "um i i sent the api report",
        mode=RewriteMode.BALANCED,
        channel=Channel.EMAIL,
        profile=UserProfile(user_id="frank", replacements={"api": "API"}),
    )
    assert result == "I sent the API report."


def test_verbatim_mode_preserves_fillers() -> None:
    result = RuleBasedPolisher().polish(
        "um this works",
        mode=RewriteMode.VERBATIM,
        channel=Channel.CHAT,
        profile=UserProfile(user_id="frank"),
    )
    assert result == "Um this works"


def test_disfluency_ratio_is_bounded() -> None:
    assert disfluency_ratio("um um um") == 1.0
    assert disfluency_ratio("clean sentence") == 0.0
