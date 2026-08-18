from voxadapt.bandit import LinUCBBandit
from voxadapt.models import Channel, RewriteMode


def test_cold_start_prefers_balanced() -> None:
    bandit = LinUCBBandit(alpha=0.1)
    assert bandit.choose("um send the report", Channel.EMAIL).arm is RewriteMode.BALANCED


def test_feedback_can_shift_rewrite_strategy() -> None:
    bandit = LinUCBBandit(alpha=0.05)
    features = bandit.features("please preserve every word", Channel.NOTES).tolist()
    for _ in range(12):
        bandit.update(RewriteMode.VERBATIM, features, 1.0)
        bandit.update(RewriteMode.BALANCED, features, 0.0)
        bandit.update(RewriteMode.POLISHED, features, 0.0)
    assert bandit.choose("please preserve every word", Channel.NOTES).arm is RewriteMode.VERBATIM


def test_bandit_round_trip(tmp_path) -> None:
    path = tmp_path / "bandit.json"
    bandit = LinUCBBandit()
    choice = bandit.choose("hello", Channel.CHAT)
    bandit.update(choice.arm, choice.features, 0.8)
    bandit.save(path)
    loaded = LinUCBBandit.load(path)
    assert loaded.choose("hello", Channel.CHAT).arm == bandit.choose("hello", Channel.CHAT).arm
