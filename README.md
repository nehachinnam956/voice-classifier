# Voice Detection & Classification System

I built this for a task that asked for an AI voice classification pipeline end to end: record audio, extract features, train a model, evaluate it properly, and get it running on a phone. I picked keyword spotting on Google's Speech Commands dataset (10 words: yes/no/up/down/left/right/on/off/stop/go, plus unknown and silence) because it's clean, well-labeled, and small enough to iterate on quickly.

**Live demo:** https://drive.google.com/file/d/1MtuRTOSwUlYyubd6mktfAvER7fQ8Py10/view?usp=drive_link

## What's actually in here

I trained two models on purpose, not one, so I'd have something to compare:

- A CatBoost baseline on hand-computed MFCC statistics (mean/std per coefficient, plus deltas)
- A small CNN on log-mel spectrograms

The CNN wins by a wide margin — about 90-91% test accuracy versus CatBoost's 69%. Full breakdown, including where each model gets confused, is in `docs/technical_writeup.md` and `model_evaluation_results.md`. The short version: MFCC summary stats throw away the time structure of a word, so CatBoost can't reliably separate "unknown" (a grab-bag of random words) from anything else. The CNN sees the actual shape of the sound over time and does much better on exactly that class.

## Tech stack

**Data & preprocessing**
Google Speech Commands v0.02, librosa (resampling, silence trimming, MFCC/mel-spectrogram extraction), soundfile, numpy/scipy.

**Models**
CatBoost (MFCC-statistics baseline), TensorFlow/Keras (CNN on log-mel spectrograms), scikit-learn (splitting, label encoding, metrics).

**Evaluation**
matplotlib for confusion matrix plots.

**Model export**
TensorFlow Lite — the on-device export path (built and verified, not currently wired into the running app — see below).

**Inference server**
FastAPI + Uvicorn, with ffmpeg (via `imageio-ffmpeg`, invoked directly through `subprocess`) decoding whatever audio format the phone sends.

**Mobile app**
React Native, Expo (SDK 54), expo-av for recording.

**Tooling**
Google Colab (T4 GPU) for training, Node.js for the feature-parity verification script (a hand-written FFT + mel filterbank in plain JS, no DSP library), Git/GitHub.

## Pipeline

```
Mic (1s clip)
  -> resample to 16kHz, trim silence, normalize, pad/crop to fixed length   (src/data_prep.py)
  -> MFCC stats (CatBoost) or log-mel spectrogram (CNN)
  -> trained model -> predicted class + confidence
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# needs real internet access — I ran this in Colab
python src/download_data.py --out_dir data

python src/train_classical.py --manifest data/manifest.csv --out models/classical
python src/train_cnn.py --manifest data/manifest.csv --out models/cnn

python src/evaluate.py --model_dir models/classical
python src/evaluate.py --model_dir models/cnn
```

## The mobile app — and why it's client-server, not on-device

Originally I planned to export the CNN to TFLite and run inference directly on the phone. I got that far — `src/export_tflite.py` works and produces a 37KB model, and `mobile/featureExtraction.js` is a from-scratch JS port of the mel-spectrogram extraction that I numerically verified against the actual Python output (max difference 0.39dB across the whole spectrogram, checked with `scripts/verify_feature_parity.mjs`).

Where it fell apart was the native Android build. `react-native-fast-tflite` needs a custom Expo dev client, which means building through EAS — and I hit a wall there: Gradle build failures on top of an actual EAS infrastructure outage on Expo's end that day (confirmed on their status page, not just a guess). After two failed builds I decided not to keep burning time on build infrastructure and switched to a simpler architecture: the phone records audio and sends it to a small FastAPI server running on my laptop, which runs the real trained model and sends back a prediction. Same model, same preprocessing, just running server-side instead of on-device.

It works. The demo video above is that exact setup — phone and laptop on the same WiFi, live predictions with confidence scores.

To run it yourself:

```bash
# on your laptop
pip install -r requirements.txt
python src/inference_server.py --model models/cnn/cnn_model.keras --port 8000

# find your laptop's LAN IP (ipconfig on Windows), set it as the server
# URL in the app, then:
cd mobile
npm install
npx expo start
```

Scan the QR code with Expo Go. Phone and laptop need to be on the same WiFi.

The on-device TFLite path is still fully there in the code (`mobile/featureExtraction.js`, `src/export_tflite.py`) — I just didn't get it wired up end to end given the build issues. If EAS cooperates next time, that's the natural next step.

## What I'd flag as genuinely hard, not just "future work"

- **The `unknown` class.** Both models struggle with it relative to the actual keywords, and that's structural — it's a grab-bag category by design, not a bug to fix.
- **Feature parity between Python and JS.** I got this working and verified for the on-device path, but it was easily the most failure-prone part of the whole project — a silent mismatch there would tank accuracy with no error thrown anywhere.
- **Windows + ffmpeg without ffprobe.** The inference server originally used `pydub`, which calls a separate `ffprobe` binary regardless of what format you tell it — and `imageio-ffmpeg` only bundles `ffmpeg`, not `ffprobe`. Took a while to track down since the error message just said "file not found" with no indication of which file. Fixed by calling `ffmpeg` directly via subprocess instead of going through pydub at all.

## Repo layout

```
src/               preprocessing, training, eval, TFLite export, inference server
mobile/            Expo app (client-server inference) + the on-device JS pipeline
scripts/           feature-parity verification (Python reference vs JS)
data/              manifest.csv + raw audio (gitignored — regenerate with download_data.py)
models/            trained models + confusion matrices (gitignored — regenerate by training)
docs/              technical write-up
```

## Results, briefly

| Model | Accuracy | Macro F1 |
|---|---|---|
| CatBoost (MFCC stats) | 68.7% | 0.68 |
| CNN (log-mel), run 1 | 90.0% | 0.896 |
| CNN (log-mel), run 2 | 91.0% | 0.908 |

Full metrics, per-class breakdown, and confusion matrices are in `model_evaluation_results.md`.
