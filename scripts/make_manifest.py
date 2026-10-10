#!/usr/bin/env python3
"""Build the CSV manifest used by inference and Whisper training, and the JSONL used by Qwen training.

CSV columns (required):
    path,text

path is a filesystem path to a WAV (16 kHz mono is what the released models were trained on).
text is the transcript. Do not add the Qwen language prefix here; csv_to_jsonl does that.

JSONL line (Qwen fine-tuning only):
    {"audio": "/abs/utt.wav", "text": "language fa<asr_text>متن"}

The released Taraneh-Q checkpoints were trained with the prefix ``language fa``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def write_csv(rows: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "text"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def write_jsonl(rows: list[tuple[str, str]], path: Path, language: str) -> None:
    prefix = f"language {language}<asr_text>"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for audio, text in rows:
            f.write(json.dumps({"audio": audio, "text": prefix + text}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} lines to {path}")


def from_csv(csv_path: Path, base_path: str | None) -> list[tuple[str, str]]:
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = {c.lower(): c for c in (reader.fieldnames or [])}
        path_key = fields.get("path") or fields.get("audio")
        text_key = fields.get("text") or fields.get("sentence") or fields.get("transcript")
        if not path_key or not text_key:
            raise SystemExit(f"{csv_path} needs path and text columns. Got {reader.fieldnames}")
        for row in reader:
            rel = (row[path_key] or "").strip()
            text = (row[text_key] or "").strip()
            if not rel or not text:
                continue
            if base_path:
                rel = os.path.normpath(os.path.join(base_path, rel.lstrip("./")))
            rows.append((rel, text))
    return rows


def from_audio_dir(audio_dir: Path) -> list[tuple[str, str]]:
    rows = []
    files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    for audio in files:
        sidecar = audio.with_suffix(".txt")
        if not sidecar.exists():
            raise SystemExit(f"Missing transcript {sidecar}. Put one UTF-8 line next to each audio file.")
        text = sidecar.read_text(encoding="utf-8").strip()
        if text:
            rows.append((str(audio.resolve()), text))
    return rows


def main():
    p = argparse.ArgumentParser(description="Build a Taraneh CSV manifest and optional Qwen JSONL")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=str, help="Existing CSV with path,text")
    src.add_argument("--audio_dir", type=str, help="Directory of audio files, each with a sibling .txt transcript")
    p.add_argument("--base_path", type=str, default="", help="Prefix joined to relative paths in --csv")
    p.add_argument("--output", type=str, required=True, help="Output CSV path")
    p.add_argument("--jsonl", type=str, default="", help="Also write a Qwen fine-tuning JSONL")
    p.add_argument(
        "--language",
        type=str,
        default="fa",
        help="Language token inside the Qwen prefix (default fa, matching the released Taraneh-Q training data)",
    )
    args = p.parse_args()

    if args.audio_dir:
        rows = from_audio_dir(Path(args.audio_dir))
    else:
        rows = from_csv(Path(args.csv), args.base_path or None)
    if not rows:
        raise SystemExit("No utterances written")
    write_csv(rows, Path(args.output))
    if args.jsonl:
        write_jsonl(rows, Path(args.jsonl), args.language)


if __name__ == "__main__":
    main()
