from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as parquet
from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small real-audio LibriSpeech fixture")
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("artifacts/librispeech"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parquet_path = hf_hub_download(
        repo_id="hf-internal-testing/librispeech_asr_dummy",
        repo_type="dataset",
        filename="clean/validation-00000-of-00001.parquet",
    )
    table = parquet.read_table(parquet_path)
    manifest = []
    for index, row in enumerate(table.slice(0, args.examples).to_pylist()):
        audio = row["audio"]
        extension = Path(audio["path"]).suffix or ".flac"
        destination = args.output / f"{index:03d}{extension}"
        destination.write_bytes(audio["bytes"])
        manifest.append(
            {
                "id": f"librispeech-{index:03d}",
                "path": str(destination),
                "text": row["text"],
            }
        )
    manifest_path = args.output / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(case) for case in manifest) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
