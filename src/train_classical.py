"""
train_classical.py — Baseline: MFCC summary stats -> CatBoost.

This is the fast, interpretable baseline. Run after download_data.py has
produced data/manifest.csv.

Usage:
    python src/train_classical.py --manifest data/manifest.csv --out models/classical
"""

import argparse
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from catboost import CatBoostClassifier

from data_prep import featurize_for_classical


def load_manifest(path):
    filepaths, labels = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepaths.append(row["filepath"])
            labels.append(row["label"])
    return filepaths, labels


def build_feature_matrix(filepaths, cache_path=None):
    """Extract MFCC-stat features for every file. Cache to .npy since this
    is the slow step and you'll want to rerun training many times."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return np.load(cache_path)

    feats = []
    for i, fp in enumerate(filepaths):
        if i % 500 == 0:
            print(f"  featurizing {i}/{len(filepaths)}")
        feats.append(featurize_for_classical(fp))
    X = np.stack(feats)

    if cache_path:
        np.save(cache_path, X)
    return X


def main(args):
    os.makedirs(args.out, exist_ok=True)

    filepaths, labels = load_manifest(args.manifest)
    print(f"Loaded manifest: {len(filepaths)} files, "
          f"{len(set(labels))} classes: {sorted(set(labels))}")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    cache = os.path.join(args.out, "X_features.npy")
    X = build_feature_matrix(filepaths, cache_path=cache)

    # stratified split: train/val/test = 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    print(f"Split: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=50,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    y_pred = model.predict(X_test).flatten()
    print("\n=== Test set results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    np.save(os.path.join(args.out, "confusion_matrix.npy"), cm)
    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("\n".join(le.classes_))

    model.save_model(os.path.join(args.out, "catboost_model.cbm"))
    print(f"\nModel + eval artifacts saved to {args.out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out", default="models/classical")
    main(parser.parse_args())
