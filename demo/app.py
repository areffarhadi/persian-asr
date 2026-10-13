"""Public demo for the five Taraneh models.

One model is loaded at a time. Clips are limited to 30 seconds so a free CPU
Space can finish a request. Word times use the same heads as the paper.
"""

from __future__ import annotations

import gc
import json
import threading
from pathlib import Path

import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent
HEADS = json.loads((ROOT / "alignment" / "paper_heads.json").read_text(encoding="utf-8"))
MAX_SECONDS = 30
SAMPLE_RATE = 16000

MODELS = [
    "Taraneh-WBase",
    "Taraneh-WMedium",
    "Taraneh-WLarge",
    "Taraneh-Q0.6B",
    "Taraneh-Q1.7B",
]

_lock = threading.Lock()
_asr = {"key": None, "model": None, "processor": None}


def _repo(name: str) -> str:
    return f"areffarhadi/{name}"


def _family(name: str) -> str:
    return "qwen" if name.startswith("Taraneh-Q") else "whisper"


def _empty_cache() -> None:
    _asr["key"] = None
    _asr["model"] = None
    _asr["processor"] = None
    gc.collect()


def _load_wave(path: str) -> tuple[np.ndarray, str]:
    import soundfile as sf

    audio, sr = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != SAMPLE_RATE:
        n_out = int(round(len(audio) * SAMPLE_RATE / sr))
        old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new = np.linspace(0.0, 1.0, num=max(n_out, 1), endpoint=False)
        audio = np.interp(new, old, audio).astype(np.float32)
    note = ""
    limit = SAMPLE_RATE * MAX_SECONDS
    if len(audio) > limit:
        audio = audio[:limit]
        note = f"The clip was longer than {MAX_SECONDS} seconds, so only the first {MAX_SECONDS} seconds were used."
    if audio.size == 0:
        raise ValueError("The recording is empty.")
    return audio, note


def _whisper_asr(name: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    key = f"whisper:{name}"
    if _asr["key"] != key:
        _empty_cache()
        repo = _repo(name)
        processor = WhisperProcessor.from_pretrained(repo, language="persian", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(
            repo, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        model.to("cpu")
        model.eval()
        _asr.update(key=key, model=model, processor=processor)
    return _asr["model"], _asr["processor"]


def _qwen_asr(name: str):
    import torch
    from qwen_asr import Qwen3ASRModel

    key = f"qwen:{name}"
    if _asr["key"] != key:
        _empty_cache()
        model = Qwen3ASRModel.from_pretrained(
            _repo(name),
            dtype=torch.float32,
            device_map="cpu",
            max_new_tokens=225,
            attn_implementation="sdpa",
        )
        _asr.update(key=key, model=model, processor=None)
    return _asr["model"]


def _transcribe(name: str, audio: np.ndarray) -> str:
    if _family(name) == "whisper":
        import torch

        model, processor = _whisper_asr(name)
        feats = processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        with torch.inference_mode():
            ids = model.generate(
                feats,
                language="persian",
                task="transcribe",
                num_beams=1,
                do_sample=False,
                max_new_tokens=225,
                return_timestamps=False,
            )
        return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()

    model = _qwen_asr(name)
    result = model.transcribe(audio=(audio, SAMPLE_RATE), language="Persian")
    return (result[0].text or "").strip()


def _align(name: str, audio: np.ndarray, text: str) -> list[list]:
    import torch

    _empty_cache()
    heads = [tuple(pair) for pair in HEADS[name]["heads"]]
    if _family(name) == "whisper":
        from alignment.whisper_forced_align import WhisperForcedAligner

        aligner = WhisperForcedAligner(_repo(name), language="Persian", device="cpu", dtype=torch.float32)
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
        rows = [[span.word, round(float(span.start), 2), round(float(span.end), 2)] for span in spans]
        del aligner
    else:
        from alignment.qwen_attn_align import QwenAttnAligner

        aligner = QwenAttnAligner(_repo(name), device="cpu", dtype=torch.float32, language="Persian")
        words, _scores = aligner.align(
            audio,
            text,
            heads=heads,
            medfilt=3,
            sil_trim=True,
            refine=False,
            energy_thr_db=-35.0,
        )
        rows = [[item["word"], item["start"], item["end"]] for item in words]
        del aligner
    _empty_cache()
    return rows


def recognize(audio_path: str | None, name: str, task: str):
    if not audio_path:
        yield "Record or upload a clip first.", []
        return
    if task.startswith("Transcript and") and name not in HEADS:
        task = "Transcript only"
        base_note = "Taraneh-WBase returns a transcript. Word times are available for the other four models."
    else:
        base_note = ""
    try:
        with _lock:
            yield "Reading the audio.", []
            audio, trim_note = _load_wave(audio_path)
            yield f"Loading {name}. The first run downloads the weights.", []
            text = _transcribe(name, audio)
            rows: list[list] = []
            if task.startswith("Transcript and") and text:
                yield "Aligning words. This loads the model a second time and takes longer.", []
                rows = _align(name, audio, text)
    except Exception as exc:  # noqa: BLE001
        yield f"This run failed: {exc}", []
        return
    notes = [part for part in (base_note, trim_note) if part]
    if notes:
        text = text + "\n\n" + " ".join(notes)
    yield text or "(empty transcript)", rows


def build() -> gr.Blocks:
    with gr.Blocks(title="Taraneh-ASR") as demo:
        gr.Markdown(
            """
# Taraneh-ASR
**درود بر پارسی‌زبانان،**  
**و بر آن خاکِ گرامی.**

Persian speech recognition for a clip of up to 30 seconds.
This demo runs on a free CPU and loads one model at a time.
[Models](https://huggingface.co/collections/areffarhadi/taraneh-asr-persian-automatic-speech-recognition)
· [Paper and code](https://github.com/areffarhadi/persian-asr)
"""
        )
        with gr.Row():
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
            with gr.Column():
                model = gr.Dropdown(MODELS, value="Taraneh-WBase", label="Model")
                task = gr.Radio(
                    ["Transcript only", "Transcript and word times"],
                    value="Transcript only",
                    label="Output",
                )
                button = gr.Button("Transcribe", variant="primary")
        transcript = gr.Textbox(label="Transcript", lines=4)
        words = gr.Dataframe(
            headers=["word", "start", "end"],
            datatype=["str", "number", "number"],
            label="Word times (seconds)",
            row_count=(0, "dynamic"),
        )
        button.click(recognize, [audio, model, task], [transcript, words])
    return demo


if __name__ == "__main__":
    build().queue(max_size=8).launch()
