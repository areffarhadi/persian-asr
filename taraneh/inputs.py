"""Manifest and audio-file inputs shared by the inference scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def collect_inputs(manifest: str | None, audio: str | None) -> pd.DataFrame:
    if manifest:
        return read_manifest(Path(manifest))
    if audio:
        p = Path(audio)
        if p.is_dir():
            files = sorted(x for x in p.rglob("*") if x.suffix.lower() in AUDIO_EXTS)
            if not files:
                raise SystemExit(f"No audio files under {p}")
            return pd.DataFrame({"path": [str(x) for x in files], "text": ""})
        return pd.DataFrame({"path": [str(p)], "text": ""})
    raise SystemExit("Pass --manifest or --audio")


def write_results(rows: pd.DataFrame, hyps: list[str], output_dir: str) -> Path:
    out = pd.DataFrame(
        {
            "path": rows["path"],
            "original_text": rows["text"],
            "generated_text": hyps,
        }
    )
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "inference_results.csv"
    out.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    return csv_path
