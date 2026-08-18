from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from .evaluation import evaluate_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(prog="voxadapt")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    evaluate = commands.add_parser("evaluate", help="evaluate a JSONL corpus")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "serve":
        uvicorn.run("voxadapt.api:app", host=args.host, port=args.port, reload=False)
    elif args.command == "evaluate":
        report = evaluate_jsonl(args.dataset)
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)


if __name__ == "__main__":
    main()
