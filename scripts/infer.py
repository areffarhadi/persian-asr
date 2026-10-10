#!/usr/bin/env python3
"""Transcribe Persian audio with a Taraneh model.

Examples
--------
One file::

    python scripts/infer.py --model Taraneh-WLarge --audio utt.wav --output out.txt

A CSV manifest (path,text) — this is the usual evaluation path::

    python scripts/infer.py \\
        --model Taraneh-WLarge \\
        --manifest manifests/test.csv \\
        --output_dir results/Taraneh-WLarge

A local checkpoint instead of the Hugging Face id::

    python scripts/infer.py --model /path/to/checkpoint --manifest manifests/test.csv --output_dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taraneh.audio import TARGET_SR, load_audio  # noqa: E402
from taraneh.models import resolve_model  # noqa: E402

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    path_col = cols.get("path") or cols.get("audio")
    if not path_col:
        raise SystemExit(f"{path} needs a path (or audio) column. Got {list(df.columns)}")
    text_col = cols.get("text") or cols.get("sentence") or cols.get("transcript")
    out = pd.DataFrame({"path": df[path_col].astype(str)})
    out["text"] = df[text_col].fillna("").astype(str) if text_col else ""
    return out


def collect_inputs(args) -> pd.DataFrame:
    if args.manifest:
        return read_manifest(Path(args.manifest))
    if args.audio:
        p = Path(args.audio)
        if p.is_dir():
            files = sorted(x for x in p.rglob("*") if x.suffix.lower() in AUDIO_EXTS)
            if not files:
                raise SystemExit(f"No audio files under {p}")
            return pd.DataFrame({"path": [str(x) for x in files], "text": ""})
        return pd.DataFrame({"path": [str(p)], "text": ""})
    raise SystemExit("Pass --manifest or --audio")


def transcribe_whisper(rows: pd.DataFrame, checkpoint: str, batch_size: int, device: str) -> list[str]:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(checkpoint, language="persian", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint)
    model.generation_config.language = "persian"
    model.generation_config.task = "transcribe"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    hyps: list[str] = []
    paths = rows["path"].tolist()
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        waves = []
        for path in batch:
            try:
                waves.append(load_audio(path))
            except Exception as exc:  # noqa: BLE001
                print(f"Error reading {path}: {exc}")
                waves.append(None)
        good = [(i, w) for i, w in enumerate(waves) if w is not None]
        texts = ["ERROR"] * len(batch)
        if good:
            feats = processor.feature_extractor(
                [w for _, w in good],
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            ).input_features.to(device)
            with torch.no_grad():
                ids = model.generate(feats, max_new_tokens=225)
            decoded = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
            for (i, _), text in zip(good, decoded):
                texts[i] = text.strip()
        hyps.extend(texts)
        print(f"{min(start + batch_size, len(paths))}/{len(paths)}")
    return hyps


def transcribe_qwen(rows: pd.DataFrame, checkpoint: str, batch_size: int, device: str, language: str | None) -> list[str]:
    import torch
    from qwen_asr import Qwen3ASRModel

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    model = Qwen3ASRModel.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=device,
        max_inference_batch_size=batch_size,
        max_new_tokens=256,
    )
    hyps: list[str] = []
    paths = rows["path"].tolist()
    lang = language or None
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        try:
            results = model.transcribe(audio=batch if len(batch) > 1 else batch[0], language=lang)
            if len(batch) == 1:
                results = list(results)
            hyps.extend((r.text or "").strip() for r in results)
        except Exception as exc:  # noqa: BLE001
            print(f"Batch failed ({exc}); falling back to single files")
            for path in batch:
                try:
                    one = model.transcribe(audio=path, language=lang)
                    hyps.append((one[0].text or "").strip())
                except Exception as one_exc:  # noqa: BLE001
                    print(f"Error processing {path}: {one_exc}")
                    hyps.append("ERROR")
        print(f"{min(start + batch_size, len(paths))}/{len(paths)}")
    return hyps


def main():
    p = argparse.ArgumentParser(description="Taraneh Persian ASR inference")
    p.add_argument(
        "--model",
        required=True,
        help="Taraneh-WBase | Taraneh-WMedium | Taraneh-WLarge | Taraneh-Q0.6B | Taraneh-Q1.7B, "
        "a local checkpoint, or a Hugging Face repo id",
    )
    p.add_argument("--manifest", type=str, help="CSV with columns path,text")
    p.add_argument("--audio", type=str, help="One audio file or a directory of audio files")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--language",
        type=str,
        default="",
        help="Qwen only. Leave empty for automatic language id (used for the reported numbers). "
        "Pass Persian to force Persian.",
    )
    args = p.parse_args()

    spec = resolve_model(args.model)
    rows = collect_inputs(args)
    print(f"Model: {spec['name']} ({spec['family']}) <- {spec['checkpoint']}")
    print(f"Utterances: {len(rows)}")

    if spec["family"] == "qwen":
        hyps = transcribe_qwen(rows, spec["checkpoint"], args.batch_size, args.device, args.language)
    else:
        hyps = transcribe_whisper(rows, spec["checkpoint"], args.batch_size, args.device)

    out = pd.DataFrame(
        {
            "path": rows["path"],
            "original_text": rows["text"],
            "generated_text": hyps,
        }
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "inference_results.csv"
    out.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    if len(rows) == 1 and not args.manifest:
        print(hyps[0])


if __name__ == "__main__":
    main()
