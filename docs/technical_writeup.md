# Technical Write-up: Voice Detection & Classification System

## 1. Dataset

- Google Speech Commands v0.02, 10 keyword classes + `unknown` + `silence`.
- [ ] Fill in: total clips, per-class counts, train/val/test split sizes.

## 2. Preprocessing

- Resample to 16kHz mono, trim leading/trailing silence (`top_db=25`),
  peak-normalize to [-1, 1], fix to a 1.0s window (center-crop / symmetric
  zero-pad).
- **Why fixed-length windows matter:** classical ML needs a fixed feature
  vector regardless of speech length, and the CNN needs a fixed tensor
  shape for batching — padding/truncation strategy directly affects how
  much of a longer word gets cut off or how much silence a short word
  carries.

## 3. Feature extraction

- **MFCC (classical path):** 13 coefficients + first/second deltas, mean
  and std pooled over time -> 78-dim vector per clip.
- **Why MFCC approximates human hearing:** the mel scale compresses
  frequency roughly the way the cochlea does — linear below ~1kHz, log
  above — and the DCT decorrelates mel-band energies so each coefficient
  is largely independent, which is why a handful of MFCCs work well with
  simple classifiers where raw spectrograms wouldn't.
- **Log-mel spectrogram (CNN path):** 64 mel bands x ~101 time frames,
  treated as a single-channel image.

## 4. Models

| Model | Features | Params | Notes |
|---|---|---|---|
| CatBoost | MFCC stats (78-dim) | [ ] iterations | fast, interpretable, CPU-only |
| CNN | log-mel (64x101x1) | [ ] fill in | 3x Conv2D + GAP + Dense, trained with early stopping |

## 5. Results

- [ ] Accuracy, macro F1, per-class precision/recall for both models
- [ ] Confusion matrices (see `models/*/confusion_matrix.png`)
- [ ] Comparison: does the CNN's accuracy gain justify its extra
      inference cost / model size vs. CatBoost, for this deployment target?

## 6. Failure modes

- [ ] Fill in from `evaluate.py`'s top-confusions output — e.g. phonetically
  similar pairs (`no`/`go`, `on`/`off`) are the expected hard cases; note
  which actually showed up and by how much.

## 7. Mobile integration

- Model exported to TFLite (`export_tflite.py`), quantized for size.
- On-device pipeline mirrors training preprocessing exactly (same sample
  rate, window length, mel parameters) — this is the step most likely to
  silently break accuracy if it drifts from the training-time code.
- [ ] Fill in: model size before/after quantization, on-device inference
  latency, any accuracy drop observed after quantization.

## 8. Challenges encountered

- [ ] Fill in as you hit them — e.g. feature-extraction parity between
  Python and JS, class imbalance in `unknown`, quantization accuracy trade-offs.
