"""
evaluate.py — Shared evaluation reporting for both models.

Loads a saved confusion_matrix.npy + classes.txt (written by train_classical.py
or train_cnn.py) and produces a plot + a plain-text failure-mode summary,
i.e. which class pairs get confused most, for the technical write-up.

Usage:
    python src/evaluate.py --model_dir models/classical
    python src/evaluate.py --model_dir models/cnn
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_eval_artifacts(model_dir):
    cm = np.load(os.path.join(model_dir, "confusion_matrix.npy"))
    with open(os.path.join(model_dir, "classes.txt")) as f:
        classes = [line.strip() for line in f if line.strip()]
    return cm, classes


def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix plot to {out_path}")


def top_confusions(cm, classes, top_k=10):
    """Report the most-confused class pairs, excluding the diagonal."""
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], classes[i], classes[j]))
    pairs.sort(reverse=True)
    print(f"\nTop {top_k} confusions (true -> predicted, count):")
    for count, true_cls, pred_cls in pairs[:top_k]:
        print(f"  {true_cls:>10} -> {pred_cls:<10} : {count}")
    return pairs[:top_k]


def main(args):
    cm, classes = load_eval_artifacts(args.model_dir)
    plot_path = os.path.join(args.model_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes, plot_path)
    top_confusions(cm, classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    main(parser.parse_args())
