"""Load audio as mono float32 at 16 kHz."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    data, sr = sf.read(str(path), always_2d=False)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    if sr != target_sr:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return data
