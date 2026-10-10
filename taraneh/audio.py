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
        data = _resample(data, sr, target_sr)
    return data


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    try:
        import soxr

        return soxr.resample(data, orig_sr, target_sr).astype(np.float32)
    except Exception:
        pass
    try:
        from math import gcd

        from scipy.signal import resample_poly

        factor = gcd(orig_sr, target_sr)
        return resample_poly(data, target_sr // factor, orig_sr // factor).astype(np.float32)
    except Exception:
        import librosa

        return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
