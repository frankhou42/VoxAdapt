from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from .models import Channel, RewriteMode, UserProfile
from .personalization import normalized_similarity
from .text import RuleBasedPolisher, TextPolisher


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(prediction.lower().split())
    expected = Counter(reference.lower().split())
    overlap = sum((predicted & expected).values())
    if not predicted or not expected:
        return float(predicted == expected)
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def word_error_rate(prediction: str, reference: str) -> float:
    predicted = prediction.lower().split()
    expected = reference.lower().split()
    if not expected:
        return float(bool(predicted))
    previous = list(range(len(predicted) + 1))
    for expected_word in expected:
        current = [previous[0] + 1]
        for index, predicted_word in enumerate(predicted, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[index] + 1,
                    previous[index - 1] + int(expected_word != predicted_word),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def evaluate_jsonl(
    dataset_path: Path,
    *,
    polisher: TextPolisher | None = None,
    mode: RewriteMode = RewriteMode.BALANCED,
) -> dict[str, object]:
    model = polisher or RuleBasedPolisher()
    cases = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    similarities: list[float] = []
    token_scores: list[float] = []
    exact_matches = 0
    latencies: list[float] = []
    outputs: list[dict[str, str | float]] = []

    for case in cases:
        profile = UserProfile(
            user_id=case.get("user_id", "evaluation"),
            replacements=case.get("replacements", {}),
        )
        start = time.perf_counter_ns()
        prediction = model.polish(
            case["raw"],
            mode=mode,
            channel=Channel(case.get("channel", "chat")),
            profile=profile,
        )
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000
        similarity = normalized_similarity(prediction, case["reference"])
        f1 = token_f1(prediction, case["reference"])
        exact_matches += int(prediction == case["reference"])
        similarities.append(similarity)
        token_scores.append(f1)
        latencies.append(latency_ms)
        outputs.append(
            {
                "raw": case["raw"],
                "reference": case["reference"],
                "prediction": prediction,
                "similarity": round(similarity, 4),
            }
        )

    return {
        "dataset": str(dataset_path),
        "mode": mode.value,
        "cases": len(cases),
        "exact_match": round(exact_matches / len(cases), 4) if cases else 0.0,
        "mean_character_similarity": round(mean(similarities), 4) if similarities else 0.0,
        "mean_token_f1": round(mean(token_scores), 4) if token_scores else 0.0,
        "latency_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 4) if latencies else 0.0,
            "p95": round(float(np.percentile(latencies, 95)), 4) if latencies else 0.0,
            "mean": round(mean(latencies), 4) if latencies else 0.0,
        },
        "outputs": outputs,
    }
