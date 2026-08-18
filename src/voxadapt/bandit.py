from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import Channel, RewriteMode
from .text import disfluency_ratio


@dataclass(frozen=True, slots=True)
class BanditChoice:
    arm: RewriteMode
    features: list[float]


class LinUCBBandit:
    """Small contextual bandit for per-user rewrite-strength selection."""

    arms = tuple(RewriteMode)

    def __init__(self, alpha: float = 0.6, dimension: int = 6) -> None:
        self.alpha = alpha
        self.dimension = dimension
        self._a = {arm: np.eye(dimension) for arm in self.arms}
        self._b = {arm: np.zeros(dimension) for arm in self.arms}
        # A small cold-start prior makes conservative cleanup the default while
        # still allowing user feedback to move toward verbatim or polished output.
        self._b[RewriteMode.BALANCED][0] = 0.15

    def features(self, text: str, channel: Channel) -> np.ndarray:
        word_count = len(text.split())
        return np.asarray(
            [
                1.0,
                min(word_count / 80.0, 1.0),
                disfluency_ratio(text),
                float(channel is Channel.CHAT),
                float(channel is Channel.EMAIL),
                float(channel is Channel.NOTES),
            ],
            dtype=float,
        )

    def choose(self, text: str, channel: Channel) -> BanditChoice:
        x = self.features(text, channel)
        scores: dict[RewriteMode, float] = {}
        for arm in self.arms:
            inverse = np.linalg.inv(self._a[arm])
            theta = inverse @ self._b[arm]
            uncertainty = self.alpha * np.sqrt(x @ inverse @ x)
            scores[arm] = float(theta @ x + uncertainty)
        arm = max(self.arms, key=lambda candidate: (scores[candidate], -self.arms.index(candidate)))
        return BanditChoice(arm=arm, features=x.tolist())

    def update(self, arm: RewriteMode, features: list[float], reward: float) -> None:
        x = np.asarray(features, dtype=float)
        if x.shape != (self.dimension,):
            raise ValueError(f"expected {self.dimension} features, received {x.shape}")
        clipped_reward = float(np.clip(reward, 0.0, 1.0))
        self._a[arm] += np.outer(x, x)
        self._b[arm] += clipped_reward * x

    def save(self, path: Path) -> None:
        payload = {
            "alpha": self.alpha,
            "dimension": self.dimension,
            "arms": {
                arm.value: {"a": self._a[arm].tolist(), "b": self._b[arm].tolist()}
                for arm in self.arms
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LinUCBBandit:
        payload = json.loads(path.read_text(encoding="utf-8"))
        bandit = cls(alpha=payload["alpha"], dimension=payload["dimension"])
        for arm in bandit.arms:
            bandit._a[arm] = np.asarray(payload["arms"][arm.value]["a"], dtype=float)
            bandit._b[arm] = np.asarray(payload["arms"][arm.value]["b"], dtype=float)
        return bandit
