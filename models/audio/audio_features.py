"""
models/audio/audio_features.py
================================
198-dim feature extraction for the MoodSyncAI audio emotion model.

Feature layout (identical to notebook 5 — 5_audio_emotion_recognition.ipynb):
    MFCC              N_MFCC = 40  rows
    MFCC delta        N_MFCC = 40  rows
    MFCC delta-delta  N_MFCC = 40  rows
    Mel-spectrogram   N_MELS = 64  rows
    Chroma STFT            12  rows
    Zero-crossing rate      1  row
    RMS energy              1  row
                       ──────
                         198  ← FEATURE_DIM

Output dtype : float32
Output shape : (FEATURE_DIM, T)  where T = frames determined by HOP_LENGTH
"""

import numpy as np
import librosa

from moodsyncai.config import (
    SAMPLE_RATE,
    MAX_DURATION,
    N_MFCC,
    N_MELS,
    HOP_LENGTH,
    N_FFT,
    FEATURE_DIM,
)


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract the 198-dim feature matrix from an in-memory audio array.

    Parameters
    ----------
    audio : np.ndarray
        Raw audio samples.  Mono or stereo, any float / int dtype.
    sr : int
        Native sample rate of *audio*.  Resampled to ``SAMPLE_RATE``
        (16 kHz) automatically if different.

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(198, T)``.

    Raises
    ------
    AssertionError
        If the concatenated feature dimension ≠ FEATURE_DIM.
        This guards against accidental hyper-parameter drift between
        config.py and notebook 5.
    """
    # ── resample & force mono ─────────────────────────────────────────────────
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # ── trim / zero-pad to MAX_DURATION ───────────────────────────────────────
    max_samples = int(SAMPLE_RATE * MAX_DURATION)
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)))
    else:
        audio = audio[:max_samples]

    # ── feature extraction ────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT,
    )                                                        # (40, T)

    delta  = librosa.feature.delta(mfcc)                    # (40, T)
    delta2 = librosa.feature.delta(mfcc, order=2)           # (40, T)

    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE,
            n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT,
        ),
        ref=np.max,
    )                                                        # (64, T)

    chroma = librosa.feature.chroma_stft(
        y=audio, sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH, n_fft=N_FFT,
    )                                                        # (12, T)

    zcr = librosa.feature.zero_crossing_rate(
        audio, hop_length=HOP_LENGTH,
    )                                                        #  (1, T)

    rms = librosa.feature.rms(
        y=audio, hop_length=HOP_LENGTH,
    )                                                        #  (1, T)

    # ── concatenate → (198, T) ────────────────────────────────────────────────
    feat = np.concatenate(
        [mfcc, delta, delta2, mel, chroma, zcr, rms], axis=0
    )

    assert feat.shape[0] == FEATURE_DIM, (
        f"Feature dim mismatch: expected {FEATURE_DIM}, got {feat.shape[0]}. "
        "Check N_MFCC / N_MELS / HOP_LENGTH / N_FFT in config.py."
    )
    return feat.astype(np.float32)
