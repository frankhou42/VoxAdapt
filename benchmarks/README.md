# Reproducible benchmark results

## Local ASR compute comparison

Hardware: 16 GB Apple M1 Pro MacBook Pro, CPU inference. Model: faster-whisper `tiny.en`.
Dataset: first 50 utterances from the public LibriSpeech ASR dummy validation subset. Each mode used
one warm-up utterance followed by five complete measured passes (250 inference calls per mode).

| Compute | Median p50 | Median p95 | Mean real-time factor | Mean WER |
| --- | ---: | ---: | ---: | ---: |
| float32 | 200.91 ms | 366.11 ms | 0.0418 | 29.92% |
| INT8 | 201.84 ms | 378.28 ms | 0.0431 | 29.13% |

Both configurations stayed below 400 ms p95 on this fixture. INT8 was **not** faster on this
hardware/model pair: its mean real-time factor was 3.1% higher and p95 was 3.3% higher. It produced
a 0.79 percentage-point lower mean WER. This negative result is retained because choosing a compute
type from hardware-specific measurements is more credible than assuming quantization always wins.

Run the experiment:

```bash
uv sync --extra asr --extra dev
uv run python scripts/prepare_librispeech_fixture.py --examples 50
uv run python scripts/benchmark_asr.py \
  --manifest artifacts/librispeech/manifest.jsonl --compute-type float32 --runs 5 \
  --output artifacts/asr-float32.json
uv run python scripts/benchmark_asr.py \
  --manifest artifacts/librispeech/manifest.jsonl --compute-type int8 --runs 5 \
  --output artifacts/asr-int8.json
```

The raw reports preserve every per-run prediction under `artifacts/` and are intentionally ignored
by Git because they are generated evidence rather than source.

## LoRA post-editor experiment

Hardware: 16 GB Apple M1 Pro MacBook Pro, CPU inference. Base model:
`google/flan-t5-small`; LoRA rank 8 on attention query/value projections. Training used 1,000
synthetically corrupted examples for two epochs with seed 7. The 12-case held-out fixture contains
different clean sentences from the training generator.

| Metric | Result |
| --- | ---: |
| Exact match | 41.67% |
| Mean character similarity | 95.67% |
| Mean token F1 | 88.07% |
| Post-edit p50 | 345.16 ms |
| Post-edit p95 | 878.37 ms |
| Final validation loss | 0.0959 |

This is a mechanics experiment, not evidence of production text quality. The adapter handled
capitalization and repeated words well but sometimes retained fillers and once deleted the phrase
"and warm start." The deterministic rule editor remains the serving default because conservative
content preservation matters more than the adapter's stylistic fluency on this small synthetic set.

Reproduce the run from the generated training data:

```bash
uv run python scripts/generate_training_data.py --examples 1000
uv sync --extra ml
uv run python training/train_lora.py --dataset data/train.jsonl --epochs 2 \
  --output artifacts/flan-t5-voxadapt-lora
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/evaluate_model.py \
  artifacts/flan-t5-voxadapt-lora --dataset data/sample_eval.jsonl \
  --output artifacts/lora-eval.json
```
