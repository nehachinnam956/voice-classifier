# Voice Detection & Classification System

Records a spoken keyword, classifies it, and shows the prediction with a
confidence score — trained pipeline in Python, inference on-device in a
React Native / Expo mobile app.

**Task:** 10-keyword classification (yes/no/up/down/left/right/on/off/stop/go)
plus `unknown` and `silence`, on Google Speech Commands v0.02. Two models are
trained and compared: a classical CatBoost baseline on MFCC statistics, and
a small CNN on log-mel spectrograms.

## Architecture

```
Voice Input (mic, 1s clip)
      |
      v
Preprocessing (src/data_prep.py)
  resample 16kHz -> trim silence -> peak-normalize -> fix to 1.0s window
      |
      +---------------------------+
      |                           |
      v                           v
MFCC stats (13 coef,        Log-mel spectrogram
mean+std, +deltas)          (64 mels x ~101 frames)
      |                           |
      v                           v
CatBoost                    CNN (3x Conv2D + GAP + Dense)
(src/train_classical.py)    (src/train_cnn.py)
      |                           |
      v                           v
   evaluate.py              export_tflite.py -> model.tflite
                                   |
                                   v
                          mobile/ (Expo app)
                          record -> extract features (JS, must mirror
                          data_prep.py) -> run TFLite -> label + confidence
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Fetch data + build manifest (needs open internet — run in Colab if your
# environment restricts network access)
python src/download_data.py --out_dir data

# Train both models
python src/train_classical.py --manifest data/manifest.csv --out models/classical
python src/train_cnn.py --manifest data/manifest.csv --out models/cnn

# Evaluate (confusion matrix + top confusions)
python src/evaluate.py --model_dir models/classical
python src/evaluate.py --model_dir models/cnn

# Export the CNN for mobile
python src/export_tflite.py --model models/cnn/cnn_model.keras --out models/cnn/model.tflite
cp models/cnn/model.tflite mobile/assets/model.tflite
```

## Mobile app (client-server inference)

**Architecture note:** the original plan ran TFLite inference on-device,
which needs a custom native build (EAS Build). That path was blocked by
Gradle build failures compounded by an EAS infrastructure outage. This repo
uses client-server inference instead: your laptop runs the trained model
behind a small local API, the phone just records audio and uploads it. This
runs in plain Expo Go with zero native modules. State this plainly in the
write-up as a deliberate tradeoff, not a hidden shortcut — an on-device
TFLite path remains a documented "next step," and `mobile/featureExtraction.js`
(a verified, working JS port of the mel-spectrogram pipeline) is kept in the
repo for exactly that future path.

1. Start the inference server on your laptop:
   ```bash
   python src/inference_server.py --model models/cnn/cnn_model.keras --port 8000
   ```
2. Find your laptop's LAN IP (Windows: `ipconfig`, look for "IPv4 Address"
   under your WiFi adapter — something like `192.168.1.42`).
3. In the app (or by editing `DEFAULT_SERVER_URL` in `mobile/App.js`), set
   the server URL to `http://<that IP>:8000`.
4. Run the app:
   ```bash
   cd mobile
   npm install
   npx expo start
   ```
   Scan the QR code with Expo Go. Phone and laptop must be on the same WiFi.

## On-device path (not required for the demo, documented for completeness)

`mobile/featureExtraction.js` is a working, numerically-verified JS port of
`data_prep.py`'s mel-spectrogram extraction (see `scripts/verify_feature_parity.mjs`
— matched a librosa reference within 0.39dB max / 0.013dB mean difference).
Wiring it to on-device TFLite inference would need a custom Expo dev client
(`eas build --profile development --platform android`) — this is the
integration point that was blocked by build infrastructure issues during
this project's timeline, not by any gap in the feature-extraction code itself.

## Repo layout

```
src/               preprocessing, training, eval, export scripts
mobile/            Expo app
data/              manifest.csv + raw audio (gitignored, fetched by download_data.py)
models/            trained model artifacts + confusion matrices (gitignored)
docs/              technical write-up
notebooks/         exploratory notebooks (optional)
```

## Status / next steps

- [x] Preprocessing pipeline (Phase 2) — implemented, smoke-tested
- [x] Classical baseline + CNN training scripts (Phase 3–4)
- [x] Evaluation + TFLite export (Phase 4–5)
- [x] Mobile app shell with recording + inference wiring (Phase 6)
- [ ] Run `download_data.py` and actually train both models — needs open
      internet, do this in Colab
- [ ] Port `featureExtraction.js` from stub to working JS, verified against
      the Python reference
- [ ] Fill in `docs/technical_writeup.md` with real metrics once trained
- [ ] Record demo video
