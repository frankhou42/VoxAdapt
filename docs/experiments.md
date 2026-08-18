# Experiment methodology

## Questions

1. Does learned post-editing improve held-out text quality over deterministic cleanup?
2. How much does correction-based personalization improve proper-noun and formatting accuracy?
3. What latency/quality tradeoff results from INT8 versus float32 ASR?
4. Does the contextual bandit reduce user edit distance over sequential feedback?

## Required comparisons

| Experiment | Baseline | Treatment | Primary metrics |
| --- | --- | --- | --- |
| Post-editing | rules | LoRA model | similarity, token F1, exact match, p95 |
| Personalization | empty profile | 5 prior corrections | proper-noun accuracy, edit similarity |
| ASR compute | float32 | INT8 | WER, real-time factor, p50/p95 |
| Online policy | fixed balanced | LinUCB | reward, final edit distance |

## Reporting rules

- Freeze the held-out JSONL before tuning.
- Record hardware, model checksum, dataset size, seed, and package lockfile.
- Warm each model once; run five measured passes and report the median aggregate.
- Report cold start separately.
- Keep all failed and low-quality cases in the output artifact.
- Put only measured, reproducible numbers on the resume.

## Current conclusions

- On the M1 Pro CPU fixture, both `tiny.en` compute modes stayed below 400 ms p95, but INT8 was
  slightly slower than float32. Compute type must be selected from device measurements.
- The small LoRA adapter reached 88.07% token F1 on 12 held-out synthetic cases but exhibited a
  content-deletion error. It remains an optional research backend; the conservative rules are the
  default serving path.
- The small post-editor fixture validates the evaluation loop, not generalization. A larger,
  human-reviewed spoken-editing corpus is required before making a quality claim.
