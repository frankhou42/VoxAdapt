import json

from voxadapt.evaluation import evaluate_jsonl, token_f1, word_error_rate


def test_token_f1() -> None:
    assert token_f1("send report", "send the report") == 0.8
    assert token_f1("", "") == 1.0


def test_word_error_rate() -> None:
    assert word_error_rate("send report", "send the report") == 1 / 3
    assert word_error_rate("", "") == 0.0


def test_evaluation_produces_quality_and_latency_report(tmp_path) -> None:
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "raw": "um send the api report",
                "reference": "Send the API report.",
                "channel": "email",
                "replacements": {"api": "API"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = evaluate_jsonl(dataset)
    assert report["cases"] == 1
    assert report["exact_match"] == 1.0
    assert report["mean_token_f1"] == 1.0
    assert report["latency_ms"]["p95"] >= 0.0
