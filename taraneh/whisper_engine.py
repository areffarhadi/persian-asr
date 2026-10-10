"""Fast Whisper inference.

Preferred path: convert the Hugging Face checkpoint to CTranslate2 once, then
decode many utterances per GPU batch with beam size 1 and no timestamp tokens.
If CTranslate2 is not installed, fall back to a float16 Transformers generate
that skips Whisper's temperature-fallback retries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from taraneh.audio import TARGET_SR, load_audio

_PROMPT_TOKENS = ("<|startoftranscript|>", "<|fa|>", "<|transcribe|>", "<|notimestamps|>")


def _device(requested: str) -> tuple[str, int]:
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        index = int(requested.split(":")[1]) if ":" in requested else 0
        return "cuda", index
    return "cpu", 0


def _ct2_dir(checkpoint: str) -> Path | None:
    folder = Path(checkpoint)
    if folder.is_dir() and (folder / "model.bin").exists():
        return folder
    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError:
        return None
    safe = checkpoint.replace("/", "--")
    cache = Path.home() / ".cache" / "taraneh" / "ct2" / safe
    if (cache / "model.bin").exists():
        return cache
    print(f"Converting {checkpoint} to CTranslate2 float16 (one time) -> {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        converter = TransformersConverter(
            checkpoint,
            copy_files=[
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "vocab.json",
                "special_tokens_map.json",
            ],
            load_as_float16=True,
        )
        converter.convert(str(cache), quantization="float16", force=False)
    except Exception as exc:  # noqa: BLE001
        print(f"CTranslate2 conversion failed ({exc})")
        return None
    return cache


def _load_batch(paths: list[str]) -> tuple[list[int], list[np.ndarray]]:
    keep: list[int] = []
    waves: list[np.ndarray] = []
    for i, path in enumerate(paths):
        try:
            waves.append(load_audio(path))
            keep.append(i)
        except Exception as exc:  # noqa: BLE001
            print(f"Error reading {path}: {exc}")
    return keep, waves


def _transcribe_ct2(
    paths: list[str],
    checkpoint: str,
    ct2_path: Path,
    batch_size: int,
    device: str,
    device_index: int,
    compute_type: str,
) -> list[str]:
    import ctranslate2
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(checkpoint, language="persian", task="transcribe")
    prompt = processor.tokenizer.convert_tokens_to_ids(list(_PROMPT_TOKENS))
    model = ctranslate2.models.Whisper(
        str(ct2_path),
        device=device,
        device_index=device_index,
        compute_type=compute_type if device == "cuda" else "float32",
        intra_threads=1,
    )
    hyps = ["ERROR"] * len(paths)
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        keep, waves = _load_batch(batch)
        if waves:
            feats = processor.feature_extractor(
                waves, sampling_rate=TARGET_SR, return_tensors="np"
            ).input_features
            features = [np.ascontiguousarray(row) for row in feats]
            results = model.generate(
                features,
                [prompt] * len(features),
                beam_size=1,
                num_hypotheses=1,
                max_length=225,
                return_scores=False,
            )
            ids = [result.sequences_ids[0] for result in results]
            texts = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
            for index, text in zip(keep, texts):
                hyps[start + index] = text.strip()
        print(f"{min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return hyps


def _transcribe_hf(
    paths: list[str],
    checkpoint: str,
    batch_size: int,
    device: str,
    device_index: int,
    compatible: bool,
) -> list[str]:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch_device = f"cuda:{device_index}" if device == "cuda" else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(checkpoint, language="persian", task="transcribe")
    load_kwargs = {"low_cpu_mem_usage": True, "torch_dtype": dtype}
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            checkpoint, attn_implementation="sdpa", **load_kwargs
        )
    except TypeError:
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint, **load_kwargs)
    model.to(torch_device)
    model.eval()
    model.generation_config.language = "persian"
    model.generation_config.task = "transcribe"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    hyps = ["ERROR"] * len(paths)
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        keep, waves = _load_batch(batch)
        if waves:
            feats = processor.feature_extractor(
                waves, sampling_rate=TARGET_SR, return_tensors="pt"
            ).input_features
            feats = feats.to(device=torch_device, dtype=dtype)
            gen_kwargs = dict(
                language="persian",
                task="transcribe",
                max_new_tokens=225,
                return_timestamps=False,
            )
            if not compatible:
                gen_kwargs.update(num_beams=1, do_sample=False, temperature=0.0)
            with torch.inference_mode():
                try:
                    ids = model.generate(feats, **gen_kwargs)
                except TypeError:
                    gen_kwargs.pop("temperature", None)
                    ids = model.generate(feats, **gen_kwargs)
            texts = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
            for index, text in zip(keep, texts):
                hyps[start + index] = text.strip()
        print(f"{min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return hyps


def transcribe(
    rows: pd.DataFrame,
    checkpoint: str,
    batch_size: int,
    device: str,
    backend: str = "auto",
    compute_type: str = "float16",
    compatible: bool = False,
) -> list[str]:
    dev, index = _device(device)
    paths = rows["path"].tolist()
    use_ct2 = backend in {"auto", "ct2"} and not compatible
    ct2_path = _ct2_dir(checkpoint) if use_ct2 else None
    if ct2_path is not None:
        print(f"Whisper backend: ctranslate2 ({compute_type}, batch {batch_size})")
        return _transcribe_ct2(paths, checkpoint, ct2_path, batch_size, dev, index, compute_type)
    if backend == "ct2":
        raise SystemExit("CTranslate2 is not installed. pip install ctranslate2")
    print(f"Whisper backend: transformers ({'compatible' if compatible else 'fast'}, batch {batch_size})")
    if backend == "auto":
        print("Install ctranslate2 for the faster Whisper path.")
    return _transcribe_hf(paths, checkpoint, batch_size, dev, index, compatible)
