from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock

import numpy as np


class LatencyTracker:
    def __init__(self, capacity: int = 2_000) -> None:
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=capacity))
        self._lock = Lock()

    def observe(self, stage: str, milliseconds: float) -> None:
        with self._lock:
            self._samples[stage].append(milliseconds)

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            copied = {stage: list(values) for stage, values in self._samples.items()}
        return {
            stage: {
                "count": len(values),
                "p50_ms": round(float(np.percentile(values, 50)), 3),
                "p95_ms": round(float(np.percentile(values, 95)), 3),
                "mean_ms": round(float(np.mean(values)), 3),
            }
            for stage, values in copied.items()
            if values
        }
