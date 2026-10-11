"""Paper-recipe forced alignment on top of an ASR transcript.

The four models below are the ones with a reported FarsDat alignment number.
Taraneh-WBase stays ASR-only. Heads are the dev-pool selections that the paper
applied to held-out speech: two Whisper cross-attention heads, three Qwen
self-attention heads. Whisper uses the Hugging Face attention probabilities
with no second softmax. Both families use a width-3 median filter, column
L2-normalization, dynamic time warping, and a -35 dB silence trim on the
first and last word.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HEADS_PATH = Path(__file__).resolve().parents[1] / "alignment" / "paper_heads.json"


def paper_heads() -> dict:
    return json.loads(HEADS_PATH.read_text(encoding="utf-8"))


def fa_model_name(spec: dict) -> str | None:
    blob = f"{spec.get('name', '')} {spec.get('checkpoint', '')}".lower()
    for name in paper_heads():
        if name.lower() in blob:
            return name
    return None


def add_task_argument(parser) -> None:
    parser.add_argument(
        "--task",
        choices=("asr", "asr+fa"),
        default="asr",
        help="asr writes the transcript. asr+fa also writes word times for the four alignment models.",
    )


def complete(spec: dict, rows: pd.DataFrame, hyps: list[str], output_dir: str, task: str, device: str) -> None:
    from taraneh.inputs import write_results

    write_results(rows, hyps, output_dir)
    if task == "asr":
        return
    name = fa_model_name(spec)
    if name is None:
        raise SystemExit(
            "asr+fa is available for Taraneh-WMedium, Taraneh-WLarge, Taraneh-Q0.6B, and Taraneh-Q1.7B. "
            "Taraneh-WBase is ASR only."
        )
    entry = paper_heads()[name]
    heads = [tuple(h) for h in entry["heads"]]
    print(f"Forced alignment: {name}")
    print(f"Heads (layer, head): {heads}")
    records = align_transcripts(spec["checkpoint"], entry["family"], list(rows["path"]), hyps, heads, device)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "alignments.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {path}")
    if len(records) == 1:
        for word in records[0].get("words", []):
            print(f"{word['start']:.2f}\t{word['end']:.2f}\t{word['word']}")


def align_transcripts(checkpoint, family, paths, texts, heads, device) -> list[dict]:
    if family == "qwen":
        return _align_qwen(checkpoint, paths, texts, heads, device)
    return _align_whisper(checkpoint, paths, texts, heads, device)


def _device(device: str) -> str | None:
    if device == "auto":
        return None
    return device


def _align_whisper(checkpoint, paths, texts, heads, device) -> list[dict]:
    from alignment.whisper_forced_align import MAX_AUDIO_SAMPLES, WhisperForcedAligner, load_audio

    aligner = WhisperForcedAligner(checkpoint, language="Persian", device=_device(device))
    records = []
    for path, text in zip(paths, texts):
        try:
            audio, _ = load_audio(path)
            if len(audio) > MAX_AUDIO_SAMPLES:
                audio = audio[:MAX_AUDIO_SAMPLES]
            spans = aligner.align(
                audio,
                text,
                aligned_unit_type="char",
                medfilt_width=3,
                resoftmax=False,
                heads=heads,
                sil_trim=True,
                energy_thr_db=-35.0,
            )
            words = [
                {"word": s.word, "start": round(float(s.start), 3), "end": round(float(s.end), 3)}
                for s in spans
            ]
            records.append({"path": path, "text": text, "words": words})
        except Exception as exc:  # noqa: BLE001
            records.append({"path": path, "text": text, "words": [], "error": str(exc)})
    return records


def _align_qwen(checkpoint, paths, texts, heads, device) -> list[dict]:
    import torch

    from alignment.qwen_attn_align import QwenAttnAligner
    from alignment.whisper_forced_align import load_audio

    dev = device
    if dev == "auto":
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if str(dev).startswith("cuda") and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        dtype = torch.bfloat16
    elif str(dev).startswith("cuda"):
        dtype = torch.float16
    aligner = QwenAttnAligner(checkpoint, device=dev, dtype=dtype, language="Persian")
    records = []
    for path, text in zip(paths, texts):
        try:
            audio, _ = load_audio(path)
            words, _scores = aligner.align(
                audio,
                text,
                heads=heads,
                medfilt=3,
                sil_trim=True,
                refine=False,
                energy_thr_db=-35.0,
            )
            records.append({"path": path, "text": text, "words": words})
        except Exception as exc:  # noqa: BLE001
            records.append({"path": path, "text": text, "words": [], "error": str(exc)})
    return records
