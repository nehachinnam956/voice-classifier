import argparse
import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from data_prep import featurize_for_cnn, N_MELS


def load_manifest(path):
    filepaths, labels = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepaths.append(row["filepath"])
            labels.append(row["label"])
    return filepaths, labels


def build_feature_tensor(filepaths, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return np.load(cache_path)

    feats = []
    for i, fp in enumerate(filepaths):
        if i % 500 == 0:
            print(f"  featurizing {i}/{len(filepaths)}")
        feats.append(featurize_for_cnn(fp))
    X = np.stack(feats).astype("float32")

    if cache_path:
        np.save(cache_path, X)
    return X


def build_model(input_shape, n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Rescaling(1.0 / 80.0, offset=1.0),  # rough dB->[~-1,1]
        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(args):
    os.makedirs(args.out, exist_ok=True)

    filepaths, labels = load_manifest(args.manifest)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)
    print(f"{len(filepaths)} files, {n_classes} classes: {list(le.classes_)}")

    cache = os.path.join(args.out, "X_features.npy")
    X = build_feature_tensor(filepaths, cache_path=cache)
    print(f"Feature tensor shape: {X.shape}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    print(f"Split: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    model = build_model(X.shape[1:], n_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\n=== Test set results ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    np.save(os.path.join(args.out, "confusion_matrix.npy"), cm)
    with open(os.path.join(args.out, "classes.txt"), "w") as f:
        f.write("\n".join(le.classes_))

    model.save(os.path.join(args.out, "cnn_model.keras"))
    print(f"\nModel + eval artifacts saved to {args.out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--out", default="models/cnn")
    main(parser.parse_args())
