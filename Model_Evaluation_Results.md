# Model Evaluation Results — Voice Classification System

**Task:** 12-class audio classification (10 keywords + unknown + silence)
**Dataset:** Google Speech Commands v0.02 — 46,254 clips
**Split:** 70% train (32,377) / 15% validation (6,938) / 15% test (6,939), stratified by class

Two models were trained and compared, as a deliberate baseline-vs-improved-model
design: a classical ML baseline on hand-engineered MFCC features, and a CNN
on log-mel spectrograms.

---

## 1. Summary comparison

| Model | Features | Test Accuracy | Macro F1 |
|---|---|---|---|
| CatBoost | MFCC stats (78-dim: 13 coef × mean/std × 3 for delta/delta-delta) | 68.68% | 0.682 |
| CNN (Run 1) | Log-mel spectrogram (64 × 101 × 1) | 90.03% | 0.896 |
| CNN (Run 2) | Log-mel spectrogram (64 × 101 × 1) | **91.00%** | **0.9075** |

The CNN outperforms the classical baseline by roughly **21-22 points of accuracy**,
consistent across two independent training runs (90.0% and 91.0%). The
gap is explained by what each feature representation preserves: MFCC summary
statistics collapse all temporal structure into a mean and standard
deviation per coefficient, discarding the shape of how a word unfolds over
time. The CNN instead sees the full time-frequency pattern, which matters
most for classes that need to be distinguished from many different sounds
at once (see `unknown`, below).

---

## 2. CatBoost baseline — full results

**Training:** 500 iterations, depth 6, learning rate 0.05, best test accuracy at iteration 491.

```
              precision    recall  f1-score   support

        down       0.57      0.65      0.61       588
          go       0.62      0.58      0.60       582
        left       0.68      0.68      0.68       570
          no       0.61      0.66      0.64       591
         off       0.69      0.71      0.70       562
          on       0.65      0.66      0.66       577
       right       0.71      0.72      0.72       567
     silence       0.99      1.00      1.00       578
        stop       0.72      0.70      0.71       581
     unknown       0.48      0.31      0.37       578
          up       0.66      0.74      0.70       558
         yes       0.79      0.83      0.81       607

    accuracy                           0.69      6939
   macro avg       0.68      0.69      0.68      6939
weighted avg       0.68      0.69      0.68      6939
```

**Top confusions (true → predicted, count):**

| True | Predicted | Count |
|---|---|---|
| unknown | down | 73 |
| go | no | 58 |
| unknown | no | 57 |
| stop | up | 57 |
| unknown | right | 54 |
| unknown | on | 51 |
| go | down | 50 |
| up | off | 48 |
| on | off | 47 |
| off | on | 46 |

**Failure mode:** `unknown` is the clear weak point (F1 = 0.37, far below every
other class). This is expected rather than a modeling failure — `unknown`
bundles dozens of unrelated words into one label, so its MFCC mean/std
statistics are highly variable and structurally overlap with every other
class's statistics. `on`/`off` also show a near-symmetric mutual confusion
(46 and 47 respectively), consistent with the two words' short duration and
similar vowel sound.

---

## 3. CNN — full results

**Architecture:** 3× Conv2D (16→32→64 filters) + GlobalAveragePooling2D +
Dropout(0.3) + Dense(64) + Dense(12, softmax). 28,236 parameters (110 KB).
Trained with early stopping (patience 8) and LR reduction on plateau;
converged around epoch 44–50 at val_accuracy ≈ 0.90–0.91.

**Run-to-run stability:** the CNN was trained twice independently (same
architecture, same data split, different random weight initialization).
Results were consistent across both runs, indicating the model's
performance is stable rather than a lucky single result:

| Run | Test Accuracy | Macro F1 |
|---|---|---|
| Run 1 | 90.03% | 0.8962 |
| Run 2 | 91.00% | 0.9075 |

Full classification report below is from Run 2 (the more recent run).

```
              precision    recall  f1-score   support

        down       0.92      0.87      0.89       588
          go       0.87      0.85      0.86       582
        left       0.95      0.91      0.93       570
          no       0.88      0.95      0.91       591
         off       0.91      0.89      0.90       562
          on       0.92      0.94      0.93       577
       right       0.94      0.92      0.93       567
     silence       0.99      1.00      1.00       578
        stop       0.92      0.95      0.93       581
     unknown       0.78      0.73      0.75       578
          up       0.86      0.92      0.89       558
         yes       0.96      0.96      0.96       607

    accuracy                           0.91      6939
   macro avg       0.91      0.91      0.91      6939
weighted avg       0.91      0.91      0.91      6939
```

For reference, Run 1's full classification report:

```
              precision    recall  f1-score   support

        down       0.82      0.93      0.87       588
          go       0.84      0.84      0.84       582
        left       0.94      0.89      0.91       570
          no       0.92      0.87      0.90       591
         off       0.93      0.86      0.89       562
          on       0.89      0.94      0.91       577
       right       0.95      0.89      0.92       567
     silence       1.00      1.00      1.00       578
        stop       0.89      0.94      0.92       581
     unknown       0.77      0.72      0.74       578
          up       0.87      0.91      0.89       558
         yes       0.96      0.96      0.96       607

    accuracy                           0.90      6939
   macro avg       0.90      0.90      0.90      6939
weighted avg       0.90      0.90      0.90      6939
```

**Top confusions (Run 1, true → predicted, count):**

| True | Predicted | Count |
|---|---|---|
| go | down | 44 |
| unknown | down | 36 |
| no | go | 36 |
| off | up | 35 |
| right | unknown | 33 |
| unknown | go | 27 |
| unknown | on | 25 |
| no | down | 22 |
| off | on | 20 |
| unknown | stop | 19 |

**Failure mode:** `unknown`'s F1 nearly doubles versus the baseline (0.37 →
0.74) — the clearest evidence that temporal structure, not just spectral
energy, is what separates a heterogeneous "everything else" category from
real keywords. The confusions that persist (`go`↔`down`, `no`↔`go`, `on`↔`off`)
are phonetically genuine hard cases — short duration, overlapping vowel
sounds — that survive the architecture change because they reflect real
acoustic similarity, not a modeling gap. One notable shift: `right → unknown`
(33 occurrences) is a new top confusion for the CNN — the model is more
conservative about committing to a specific keyword when uncertain, which is
arguably the safer failure mode for a deployed product (hedging into
"unknown" rather than confidently naming the wrong word).

---

## 4. `silence` — perfect on both models

Both models score F1 = 1.00 on `silence`. This is expected rather than a
sign of overfitting: background noise is acoustically nothing like speech in
either MFCC or mel-spectrogram feature space, so it's the easiest class in
the dataset by a wide margin.

---

## 5. Attachments

The following files (generated during training, in `models/classical/` and
`models/cnn/`) should be attached alongside this document:
- `confusion_matrix.png` (CatBoost)
- `confusion_matrix.png` (CNN)
- `cnn_model.keras` / `catboost_model.cbm` (trained model weights)
- `model.tflite` (32.9 KB export — not used in the final client-server
  mobile architecture, but included as the on-device inference path that
  was implemented and verified, documented in the technical write-up)
