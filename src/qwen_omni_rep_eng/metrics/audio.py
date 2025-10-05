from __future__ import annotations
import numpy as np
import librosa

def f0_stats(y: np.ndarray, sr: int = 24000, fmin=50, fmax=400) -> dict:
    """Return crude prosody stats (mean/var F0 and RMS energy)."""
    if y.ndim > 1:
        y = np.mean(y, axis=-1)
    f0 = librosa.yin(y.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr)  # [T]
    f0 = f0[np.isfinite(f0)]
    f0_mean = float(np.nanmean(f0)) if f0.size else 0.0
    f0_std  = float(np.nanstd(f0)) if f0.size else 0.0
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)  # [1, F]
    return {"f0_mean": f0_mean, "f0_std": f0_std, "rms_mean": float(np.mean(rms))}

def delta_f0_mean(y_base: np.ndarray, y_steer: np.ndarray, sr: int = 24000) -> float:
    """Mean F0 shift (steered - base). Positive means 'brighter' on average."""
    b = f0_stats(y_base, sr=sr)["f0_mean"]
    s = f0_stats(y_steer, sr=sr)["f0_mean"]
    return float(s - b)
