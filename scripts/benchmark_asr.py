from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

import av
import numpy as np

from voxadapt.asr import FasterWhisperTranscriber
from voxadapt.evaluation import word_error_rate


def audio_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        return sum(frame.samples / frame.sample_rate for frame in container.decode(audio=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local faster-whisper inference")
    parser.add_argument("--manifest", type=Path, default=Path("data/audio_manifest.jsonl"))
    parser.add_argument("--audio-dir", type=Path, default=Path("artifacts/audio"))
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--compute-type", default="int8", choices=["int8", "float32"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    model_start = time.perf_counter()
    transcriber = FasterWhisperTranscriber(args.model, compute_type=args.compute_type)
    model_load_seconds = time.perf_counter() - model_start
    warmup_path = Path(cases[0].get("path", args.audio_dir / f"{cases[0]['id']}.aiff"))
    transcriber.transcribe(warmup_path)

    runs = []
    for run_index in range(args.runs):
        latencies = []
        real_time_factors = []
        error_rates = []
        predictions = []
        for case in cases:
            path = Path(case.get("path", args.audio_dir / f"{case['id']}.aiff"))
            duration = audio_duration(path)
            start = time.perf_counter()
            prediction = transcriber.transcribe(path)
            latency = time.perf_counter() - start
            latencies.append(latency * 1_000)
            real_time_factors.append(latency / duration)
            error_rates.append(word_error_rate(prediction, case["text"]))
            predictions.append(
                {"id": case["id"], "reference": case["text"], "prediction": prediction}
            )
        runs.append(
            {
                "run": run_index + 1,
                "mean_wer": mean(error_rates),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "mean_rtf": mean(real_time_factors),
                "predictions": predictions,
            }
        )

    report = {
        "model": args.model,
        "compute_type": args.compute_type,
        "utterances": len(cases),
        "measured_runs": args.runs,
        "model_load_seconds": round(model_load_seconds, 4),
        "median_mean_wer": round(float(np.median([run["mean_wer"] for run in runs])), 4),
        "median_p50_ms": round(float(np.median([run["p50_ms"] for run in runs])), 2),
        "median_p95_ms": round(float(np.median([run["p95_ms"] for run in runs])), 2),
        "median_mean_rtf": round(float(np.median([run["mean_rtf"] for run in runs])), 4),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
