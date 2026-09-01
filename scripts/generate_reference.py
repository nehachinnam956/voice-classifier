"""
generate_reference.py — Regenerate ref_data.json, the librosa reference
used by verify_feature_parity.mjs to numerically check the JS mel
extraction against the real Python pipeline. Re-run this (and the JS
verify script) any time data_prep.py's preprocessing constants change.

Usage: python scripts/generate_reference.py
"""
import sys, os, json
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_prep import preprocess_audio, extract_mel_spectrogram, SAMPLE_RATE

np.random.seed(0)
t = np.linspace(0, 1.0, SAMPLE_RATE)
sig = (0.4 * np.sin(2 * np.pi * 220 * t)
       + 0.2 * np.sin(2 * np.pi * 440 * t)
       + 0.1 * np.random.randn(len(t))).astype(np.float32)

tmp_wav = "/tmp/ref_test.wav"
sf.write(tmp_wav, sig, SAMPLE_RATE)

y = preprocess_audio(tmp_wav)
mel = extract_mel_spectrogram(y)

out_path = os.path.join(os.path.dirname(__file__), "ref_data.json")
with open(out_path, "w") as f:
    json.dump({"samples": y.tolist(), "mel": mel.tolist()}, f)

print(f"Reference written to {out_path}, mel shape {mel.shape}")
