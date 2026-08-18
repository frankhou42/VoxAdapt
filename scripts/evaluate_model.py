from __future__ import annotations

import argparse
import json
from pathlib import Path

from voxadapt.evaluation import evaluate_jsonl
from voxadapt.text import TransformersPolisher


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base or LoRA VoxAdapt post-editor")
    parser.add_argument("model")
    parser.add_argument("--dataset", type=Path, default=Path("data/sample_eval.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_jsonl(args.dataset, polisher=TransformersPolisher(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
