import numpy as np
import librosa

# ---- Fixed pipeline constants (do not change without retraining) ----
SAMPLE_RATE = 16000          # Speech Commands is natively 16kHz
CLIP_DURATION_S = 1.0        # fixed window length
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_S)
N_MFCC = 13                  # classic MFCC count, expand to 20-40 if needed
N_MELS = 64                  # mel bands for spectrogram/CNN path
N_FFT = 512
HOP_LENGTH = 160             # 10ms hop at 16kHz
TOP_DB = 25                  # silence-trim threshold


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio, resample to SAMPLE_RATE, mono."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def trim_silence(y: np.ndarray, top_db: int = TOP_DB) -> np.ndarray:
    """Trim leading/trailing silence. Leaves internal silence intact."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed if len(trimmed) > 0 else y


def normalize_amplitude(y: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1]. Guards against silent/zero clips."""
    peak = np.max(np.abs(y))
    if peak < 1e-6:
        return y
    return y / peak


def fix_length(y: np.ndarray, target_samples: int = CLIP_SAMPLES) -> np.ndarray:
    """
    Pad with zeros or center-crop to a fixed length.
    Deliberate choice: center-crop (not left-crop) so a word that starts
    slightly late isn't chopped off; zero-pad symmetrically for short clips.
    """
    n = len(y)
    if n == target_samples:
        return y
    if n < target_samples:
        pad_total = target_samples - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(y, (pad_left, pad_right), mode="constant")
    # n > target_samples: center crop
    start = (n - target_samples) // 2
    return y[start:start + target_samples]


def preprocess_audio(path: str) -> np.ndarray:
    """Full raw-audio pipeline: load -> trim -> normalize -> fixed length."""
    y = load_audio(path)
    y = trim_silence(y)
    y = normalize_amplitude(y)
    y = fix_length(y)
    return y


def extract_mfcc(y: np.ndarray, include_deltas: bool = False) -> np.ndarray:
    """
    MFCCs approximate human auditory perception: the mel scale compresses
    frequency the way the cochlea does (roughly linear below ~1kHz, log
    above), and the DCT step decorrelates the mel-band energies so each
    coefficient carries mostly independent information — this is why MFCCs
    work well with simple classifiers where raw spectrograms wouldn't.

    Returns shape (n_mfcc [* 3 if deltas], n_frames).
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    if not include_deltas:
        return mfcc
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2])


def extract_mfcc_stats(y: np.ndarray, include_deltas: bool = False) -> np.ndarray:
    """
    Collapse MFCC time-series to per-coefficient mean + std — the feature
    vector for the classical ML baseline (CatBoost/RF/SVM). Fixed-length
    regardless of clip length, which is what tree/kernel models need.
    """
    mfcc = extract_mfcc(y, include_deltas=include_deltas)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def extract_mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """
    Log-mel spectrogram as a 2D 'image' for the CNN path.
    Returns shape (N_MELS, n_frames), log-scaled (dB).
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def featurize_for_classical(path: str) -> np.ndarray:
    """One-call path: audio file -> classical ML feature vector."""
    y = preprocess_audio(path)
    return extract_mfcc_stats(y, include_deltas=True)


def featurize_for_cnn(path: str) -> np.ndarray:
    """One-call path: audio file -> CNN input tensor (N_MELS, n_frames, 1)."""
    y = preprocess_audio(path)
    mel = extract_mel_spectrogram(y)
    return mel[..., np.newaxis]


if __name__ == "__main__":
    # Smoke test on synthetic audio (no dataset needed) — verifies the
    # pipeline runs end to end and reports output shapes.
    import soundfile as sf
    import tempfile, os

    t = np.linspace(0, 0.6, int(SAMPLE_RATE * 0.6))
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz test tone, 0.6s
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, tone, SAMPLE_RATE)
        path = f.name

    try:
        y = preprocess_audio(path)
        print(f"preprocessed audio shape: {y.shape} (expect {CLIP_SAMPLES},)")

        stats_vec = featurize_for_classical(path)
        print(f"classical feature vector shape: {stats_vec.shape} "
              f"(expect ({N_MFCC * 3 * 2},) with deltas)")

        cnn_input = featurize_for_cnn(path)
        print(f"CNN input shape: {cnn_input.shape}")
        print("data_prep.py smoke test passed.")
    finally:
        os.remove(path)
