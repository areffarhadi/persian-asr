#!/usr/bin/env python3
"""Dispatch inference to the Whisper or Qwen engine.

Prefer the family scripts when you want the backend flags:

    python scripts/infer_whisper.py --model Taraneh-WLarge --manifest manifests/test.csv --output_dir results/Taraneh-WLarge
    python scripts/infer_qwen.py --model Taraneh-Q1.7B --manifest manifests/test.csv --output_dir results/Taraneh-Q1.7B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taraneh.inputs import collect_inputs, write_results  # noqa: E402
from taraneh.models import resolve_model  # noqa: E402
from taraneh.qwen_engine import transcribe as transcribe_qwen  # noqa: E402
from taraneh.whisper_engine import transcribe as transcribe_whisper  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Taraneh Persian ASR inference")
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", type=str)
    p.add_argument("--audio", type=str)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--backend", type=str, default="auto")
    p.add_argument("--language", type=str, default="")
    p.add_argument("--compute_type", type=str, default="float16")
    p.add_argument("--compatible", action="store_true")
    args = p.parse_args()

    spec = resolve_model(args.model)
    rows = collect_inputs(args.manifest, args.audio)
    print(f"Model: {spec['name']} ({spec['family']}) <- {spec['checkpoint']}")
    print(f"Utterances: {len(rows)}")
    if spec["family"] == "qwen":
        hyps = transcribe_qwen(
            rows,
            spec["checkpoint"],
            args.batch_size or 32,
            args.device,
            args.language or None,
            backend=args.backend,
        )
    else:
        hyps = transcribe_whisper(
            rows,
            spec["checkpoint"],
            args.batch_size or 16,
            args.device,
            backend=args.backend,
            compute_type=args.compute_type,
            compatible=args.compatible,
        )
    write_results(rows, hyps, args.output_dir)
    if len(rows) == 1 and not args.manifest:
        print(hyps[0])


if __name__ == "__main__":
    main()
