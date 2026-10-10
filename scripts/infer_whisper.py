#!/usr/bin/env python3
"""Fast inference for Taraneh-WBase, Taraneh-WMedium, and Taraneh-WLarge.

    python scripts/infer_whisper.py \
        --model Taraneh-WLarge \
        --manifest manifests/test.csv \
        --output_dir results/Taraneh-WLarge \
        --batch_size 16

Install the fast backend once: pip install ctranslate2
The first run converts the checkpoint to ~/.cache/taraneh/ct2/.
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
from taraneh.whisper_engine import transcribe  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Fast Taraneh Whisper inference")
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", type=str)
    p.add_argument("--audio", type=str)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--backend",
        choices=("auto", "ct2", "transformers"),
        default="auto",
        help="auto uses CTranslate2 when installed",
    )
    p.add_argument(
        "--compute_type",
        default="float16",
        help="CTranslate2 compute type. int8_float16 is faster and slightly less exact.",
    )
    p.add_argument(
        "--compatible",
        action="store_true",
        help="Use the slower Transformers decoder, including temperature fallback.",
    )
    args = p.parse_args()

    spec = resolve_model(args.model)
    if spec["family"] != "whisper":
        raise SystemExit(f"{spec['name']} is not a Whisper model. Use scripts/infer_qwen.py")
    rows = collect_inputs(args.manifest, args.audio)
    print(f"Model: {spec['name']} <- {spec['checkpoint']}")
    print(f"Utterances: {len(rows)}")
    hyps = transcribe(
        rows,
        spec["checkpoint"],
        args.batch_size,
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
