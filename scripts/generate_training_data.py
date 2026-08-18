from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

FILLERS = ("um", "uh", "you know", "like")
SEED_TEXT = (
    "Please send the updated report before tomorrow morning.",
    "The inference benchmark should separate cold and warm starts.",
    "Can we move the design review to three in the afternoon?",
    "Add the latency results to the experiment summary.",
    "The model preserved the meaning of the original message.",
    "Run the regression suite before deploying the service.",
    "I finished reviewing the personalization experiment.",
    "The API returned a valid response for every test case.",
)


def make_spoken(text: str, rng: random.Random) -> str:
    output = text.rstrip(".!?")
    output = output[0].lower() + output[1:]
    words = output.split()
    if rng.random() < 0.75:
        words.insert(rng.randrange(0, min(4, len(words)) + 1), rng.choice(FILLERS))
    if len(words) > 3 and rng.random() < 0.5:
        index = rng.randrange(1, len(words) - 1)
        words.insert(index, words[index])
    return re.sub(r"\s+", " ", " ".join(words))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--examples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for _ in range(args.examples):
            clean = rng.choice(SEED_TEXT)
            destination.write(
                json.dumps({"input": make_spoken(clean, rng), "target": clean}) + "\n"
            )


if __name__ == "__main__":
    main()
