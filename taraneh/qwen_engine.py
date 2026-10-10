"""Fast Qwen3-ASR inference.

Preferred path: vLLM, which continuous-batches the utterances. The Transformers
path uses FlashAttention or SDPA and bfloat16 when the GPU supports it.
Automatic language identification is unchanged when --language is omitted.
"""

from __future__ import annotations

import os

import pandas as pd


def _torch_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return requested


def _dtype_name(device: str) -> str:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return "bfloat16"
    if device.startswith("cuda"):
        return "float16"
    return "float32"


def _load(checkpoint: str, backend: str, device: str, batch_size: int):
    import torch
    from qwen_asr import Qwen3ASRModel

    if backend in {"auto", "vllm"} and device.startswith("cuda"):
        if ":" in device:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", device.split(":", 1)[1])
        try:
            model = Qwen3ASRModel.LLM(
                checkpoint,
                max_inference_batch_size=-1,
                max_new_tokens=256,
                dtype=_dtype_name(device),
                gpu_memory_utilization=0.9,
            )
            print("Qwen backend: vllm")
            return model
        except Exception as exc:  # noqa: BLE001
            if backend == "vllm":
                raise SystemExit(f"vLLM failed to load: {exc}") from exc
            print(f"vLLM unavailable ({exc}). Using transformers.")

    dtype = getattr(torch, _dtype_name(device))
    kwargs = dict(
        dtype=dtype,
        device_map=device if device.startswith("cuda") else "cpu",
        max_inference_batch_size=batch_size,
        max_new_tokens=256,
    )
    model = None
    for attn in ("flash_attention_2", "sdpa", None):
        try:
            extra = {"attn_implementation": attn} if attn else {}
            model = Qwen3ASRModel.from_pretrained(checkpoint, **kwargs, **extra)
            print(f"Qwen backend: transformers ({attn or 'default'}, batch {batch_size})")
            break
        except Exception as exc:  # noqa: BLE001
            if attn is None:
                raise
            print(f"Attention {attn} unavailable ({exc})")
    return model


def _prepare_device(device: str) -> str:
    if device.startswith("cuda:") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        device = "cuda:0"
    return _torch_device(device)


def transcribe_streaming(
    rows: pd.DataFrame,
    checkpoint: str,
    device: str,
    language: str | None,
    chunk_size_sec: float = 2.0,
) -> list[str]:
    """vLLM streaming. One utterance at a time, 16 kHz chunks, final text only."""
    from taraneh.audio import TARGET_SR, load_audio

    device = _prepare_device(device)
    model = _load(checkpoint, "vllm", device, batch_size=1)
    if getattr(model, "backend", None) != "vllm":
        raise SystemExit("Qwen streaming requires the vLLM backend.")
    lang = language or None
    hyps: list[str] = []
    paths = rows["path"].tolist()
    for index, path in enumerate(paths, start=1):
        try:
            wav = load_audio(path, TARGET_SR)
            state = model.init_streaming_state(language=lang, chunk_size_sec=chunk_size_sec)
            step = state.chunk_size_samples
            for start in range(0, wav.shape[0], step):
                state = model.streaming_transcribe(wav[start : start + step], state)
            state = model.finish_streaming_transcribe(state)
            hyps.append((state.text or "").strip())
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {path}: {exc}")
            hyps.append("ERROR")
        print(f"{index}/{len(paths)}", flush=True)
    return hyps


def transcribe(
    rows: pd.DataFrame,
    checkpoint: str,
    batch_size: int,
    device: str,
    language: str | None,
    backend: str = "auto",
    mode: str = "offline",
    chunk_size_sec: float = 2.0,
) -> list[str]:
    if mode == "streaming":
        return transcribe_streaming(rows, checkpoint, device, language, chunk_size_sec)
    device = _prepare_device(device)
    model = _load(checkpoint, backend, device, batch_size)
    paths = rows["path"].tolist()
    lang = language or None
    hyps: list[str] = []
    # Outer chunks bound host RAM. vLLM still receives the whole chunk at once.
    step = batch_size if batch_size > 0 else len(paths)
    for start in range(0, len(paths), step):
        batch = paths[start : start + step]
        try:
            results = model.transcribe(audio=batch if len(batch) > 1 else batch[0], language=lang)
            hyps.extend((item.text or "").strip() for item in results)
        except Exception as exc:  # noqa: BLE001
            print(f"Batch failed ({exc}); retrying file by file")
            for path in batch:
                try:
                    one = model.transcribe(audio=path, language=lang)
                    hyps.append((one[0].text or "").strip())
                except Exception as one_exc:  # noqa: BLE001
                    print(f"Error processing {path}: {one_exc}")
                    hyps.append("ERROR")
        print(f"{min(start + step, len(paths))}/{len(paths)}", flush=True)
    return hyps
