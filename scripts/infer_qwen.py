#!/usr/bin/env python3
"""Fast inference for Taraneh-Q0.6B and Taraneh-Q1.7B.

    python scripts/infer_qwen.py \
        --model Taraneh-Q1.7B \
        --manifest manifests/test.csv \
        --output_dir results/Taraneh-Q1.7B \
        --batch_size 32

Install the fast backend once: pip install "qwen-asr[vllm]"
Leave --language empty to keep automatic language identification.
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
from taraneh.qwen_engine import transcribe  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Fast Taraneh Qwen3-ASR inference")
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", type=str)
    p.add_argument("--audio", type=str)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--batch_size", type=int, default=32, help="Utterances loaded together. vLLM decodes the chunk in one call.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--backend", choices=("auto", "vllm", "transformers"), default="auto")
    p.add_argument(
        "--language",
        type=str,
        default="",
        help="Empty keeps automatic language id. Persian forces Persian.",
    )
    args = p.parse_args()

    spec = resolve_model(args.model)
    if spec["family"] != "qwen":
        raise SystemExit(f"{spec['name']} is not a Qwen model. Use scripts/infer_whisper.py")
    rows = collect_inputs(args.manifest, args.audio)
    print(f"Model: {spec['name']} <- {spec['checkpoint']}")
    print(f"Utterances: {len(rows)}")
    hyps = transcribe(
        rows,
        spec["checkpoint"],
        args.batch_size,
        args.device,
        args.language or None,
        backend=args.backend,
    )
    write_results(rows, hyps, args.output_dir)
    if len(rows) == 1 and not args.manifest:
        print(hyps[0])


if __name__ == "__main__":
    main()
