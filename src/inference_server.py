"""
inference_server.py — Local inference bridge for the mobile app.

Runs on your laptop, loads the trained CNN, and exposes one endpoint that
accepts a recorded WAV clip and returns a prediction + confidence. This
exists specifically to avoid the mobile-native-build path (EAS/Gradle):
the phone just sends audio over your local WiFi and gets JSON back — no
TFLite native module, no custom dev client, works in plain Expo Go.

This is architecturally identical to what a real deployed backend would
look like, minus auth/scaling — a legitimate design choice for a demo,
not a shortcut around "real" inference.

Usage:
    python src/inference_server.py --model models/cnn/cnn_model.keras --port 8000

Find your laptop's LAN IP (needed by the phone to reach this server):
    Windows: ipconfig  -> look for "IPv4 Address" under your WiFi adapter
Then in mobile/App.js, set SERVER_URL to http://<that IP>:8000
"""

import argparse
import io
import os
import subprocess
import numpy as np
import soundfile as sf
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import imageio_ffmpeg

# We call ffmpeg directly via subprocess rather than through pydub.
# pydub's AudioSegment.from_file() unconditionally calls a separate
# "ffprobe" binary (regardless of the format= parameter) to read stream
# metadata before decoding — imageio_ffmpeg does NOT bundle ffprobe, only
# ffmpeg, so that call fails with FileNotFoundError on any machine that
# doesn't have ffprobe installed separately. Piping bytes straight into
# ffmpeg and reading raw PCM back out avoids needing ffprobe at all.
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from data_prep import (
    SAMPLE_RATE, trim_silence, normalize_amplitude, fix_length,
    extract_mel_spectrogram,
)

app = FastAPI(title="Voice Classifier Inference Server")

MODEL = None
CLASSES = None


def load_model_and_classes(model_path: str):
    global MODEL, CLASSES
    MODEL = tf.keras.models.load_model(model_path)
    classes_path = os.path.join(os.path.dirname(model_path), "classes.txt")
    with open(classes_path) as f:
        CLASSES = [line.strip() for line in f if line.strip()]
    print(f"Loaded model from {model_path}")
    print(f"Classes ({len(CLASSES)}): {CLASSES}")


def decode_audio_with_ffmpeg(raw_bytes: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Decode any audio format ffmpeg understands (AAC/M4A, WAV, etc.) straight
    to mono float32 PCM at target_sr, entirely via subprocess — no pydub,
    no ffprobe dependency. Bytes go in on stdin, raw PCM comes out on
    stdout, and we know the format because we told ffmpeg to produce it
    (s16le), so no metadata-probing step is needed.
    """
    cmd = [
        FFMPEG_PATH, "-i", "pipe:0",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(target_sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=raw_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='ignore')[-500:]}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASSES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a recorded audio clip in whatever format the phone actually
    produced (Android: AAC/M4A despite any ".wav" filename; iOS: usually
    real WAV or CAF). Decodes via direct ffmpeg subprocess, then runs the
    exact same preprocessing as training.
    """
    raw_bytes = await file.read()

    try:
        y = decode_audio_with_ffmpeg(raw_bytes)
    except Exception as e:
        import traceback
        print("=" * 60)
        print("DECODE FAILURE — full traceback:")
        traceback.print_exc()
        print("=" * 60)
        return JSONResponse(status_code=400, content={"error": f"Could not decode audio: {e}"})

    # exact same pipeline as training (data_prep.preprocess_audio, inlined
    # here since preprocess_audio expects a file path, not an array)
    y = trim_silence(y)
    y = normalize_amplitude(y)
    y = fix_length(y)

    mel = extract_mel_spectrogram(y)
    x = mel[np.newaxis, ..., np.newaxis].astype("float32")  # (1, N_MELS, n_frames, 1)

    probs = MODEL.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))

    return {
        "label": CLASSES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {CLASSES[i]: float(p) for i, p in enumerate(probs)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/cnn/cnn_model.keras")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    load_model_and_classes(args.model)
    print(f"\nServer starting on http://{args.host}:{args.port}")
    print("On your phone/app, use your laptop's LAN IP, not 'localhost'.")
    uvicorn.run(app, host=args.host, port=args.port)