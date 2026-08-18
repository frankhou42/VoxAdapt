from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible macOS TTS audio fixture")
    parser.add_argument("--manifest", type=Path, default=Path("data/audio_manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/audio"))
    args = parser.parse_args()
    if platform.system() != "Darwin":
        raise SystemExit("This fixture generator uses the macOS 'say' command")
    args.output.mkdir(parents=True, exist_ok=True)
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        destination = args.output / f"{case['id']}.aiff"
        subprocess.run(
            [
                "say",
                "-v",
                case["voice"],
                "-o",
                str(destination),
                case["text"],
            ],
            check=True,
        )
        print(destination)


if __name__ == "__main__":
    main()
