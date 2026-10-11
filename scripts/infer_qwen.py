#!/usr/bin/env python3
"""vLLM inference for Taraneh-Q0.6B and Taraneh-Q1.7B.

Offline, one batch:

    python scripts/infer_qwen.py \
        --model Taraneh-Q1.7B \
        --manifest manifests/test.csv \
        --output_dir results/Taraneh-Q1.7B \
        --mode offline \
        --batch_size 32

Streaming, 2-second chunks:

    python scripts/infer_qwen.py \
        --model Taraneh-Q1.7B \
        --manifest manifests/test.csv \
        --output_dir results/Taraneh-Q1.7B-stream \
        --mode streaming \
        --chunk_size_sec 2

Requires Python >= 3.10 and vLLM 0.14, which is what `qwen-asr[vllm]` installs.
Leave --language empty to keep automatic language identification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taraneh.forced_align import add_task_argument, complete  # noqa: E402
from taraneh.inputs import collect_inputs  # noqa: E402
from taraneh.models import resolve_model  # noqa: E402
from taraneh.qwen_engine import transcribe  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Fast Taraneh Qwen3-ASR inference")
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", type=str)
    p.add_argument("--audio", type=str)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--batch_size", type=int, default=32, help="Offline only. Utterances loaded together.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--backend", choices=("auto", "vllm", "transformers"), default="vllm")
    p.add_argument("--mode", choices=("offline", "streaming"), default="offline")
    p.add_argument(
        "--chunk_size_sec",
        type=float,
        default=2.0,
        help="Streaming only. Seconds of audio per update. 2.0 is the Qwen API default; the demo uses 1.0.",
    )
    p.add_argument(
        "--language",
        type=str,
        default="",
        help="Empty keeps automatic language id. Persian forces Persian.",
    )
    add_task_argument(p)
    args = p.parse_args()

    spec = resolve_model(args.model)
    if spec["family"] != "qwen":
        raise SystemExit(f"{spec['name']} is not a Qwen model. Use scripts/infer_whisper.py")
    rows = collect_inputs(args.manifest, args.audio)
    print(f"Model: {spec['name']} <- {spec['checkpoint']}")
    print(f"Utterances: {len(rows)}")
    if args.mode == "streaming" and args.backend == "transformers":
        raise SystemExit("Streaming requires --backend vllm.")
    print(f"Mode: {args.mode}")
    hyps = transcribe(
        rows,
        spec["checkpoint"],
        args.batch_size,
        args.device,
        args.language or None,
        backend="vllm" if args.mode == "streaming" else args.backend,
        mode=args.mode,
        chunk_size_sec=args.chunk_size_sec,
    )
    complete(spec, rows, hyps, args.output_dir, args.task, args.device)
    if len(rows) == 1 and not args.manifest:
        print(hyps[0])


if __name__ == "__main__":
    main()
