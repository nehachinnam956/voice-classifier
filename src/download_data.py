import argparse
import os
import tarfile
import urllib.request
import csv
import random

SPEECH_COMMANDS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "data/speech_commands_v0.02.tar.gz"
)

DEFAULT_CORE_CLASSES = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"
]


def download_and_extract(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    archive_path = os.path.join(out_dir, "speech_commands_v0.02.tar.gz")
    extract_dir = os.path.join(out_dir, "raw")

    if not os.path.exists(archive_path):
        print(f"Downloading {SPEECH_COMMANDS_URL} ...")
        urllib.request.urlretrieve(SPEECH_COMMANDS_URL, archive_path)
    else:
        print("Archive already downloaded, skipping.")

    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        print("Extracting ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)
    else:
        print("Already extracted, skipping.")

    return extract_dir


def build_manifest(extract_dir: str, classes: list, out_dir: str,
                    unknown_ratio: float = 0.1, silence_ratio: float = 0.1,
                    seed: int = 42):
    """
    Build a CSV manifest [filepath,label] with the chosen keyword classes
    plus 'unknown' (sampled from non-target words) and 'silence'
    (background noise clips), which matters for a realistic demo — a model
    that's never seen non-keyword speech will confidently mislabel it.
    """
    random.seed(seed)
    rows = []
    all_dirs = [d for d in os.listdir(extract_dir)
                if os.path.isdir(os.path.join(extract_dir, d)) and not d.startswith("_")]

    for cls in classes:
        cls_dir = os.path.join(extract_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"WARNING: class dir missing: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            if fname.endswith(".wav"):
                rows.append((os.path.join(cls_dir, fname), cls))

    n_target = len(rows)

    # unknown: sample from words NOT in our target classes
    other_dirs = [d for d in all_dirs if d not in classes and d != "_background_noise_"]
    unknown_files = []
    for d in other_dirs:
        d_path = os.path.join(extract_dir, d)
        unknown_files += [os.path.join(d_path, f) for f in os.listdir(d_path) if f.endswith(".wav")]
    random.shuffle(unknown_files)
    n_unknown = int(n_target * unknown_ratio)
    for f in unknown_files[:n_unknown]:
        rows.append((f, "unknown"))

    # silence: background noise dir, if present
    bg_dir = os.path.join(extract_dir, "_background_noise_")
    if os.path.isdir(bg_dir):
        bg_files = [os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith(".wav")]
        n_silence = int(n_target * silence_ratio)
        for i in range(n_silence):
            f = bg_files[i % len(bg_files)] if bg_files else None
            if f:
                rows.append((f, "silence"))

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        writer.writerows(rows)

    print(f"Manifest written: {manifest_path} ({len(rows)} rows, "
          f"{len(classes)} keyword classes + unknown + silence)")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CORE_CLASSES)
    args = parser.parse_args()

    extract_dir = download_and_extract(args.out_dir)
    build_manifest(extract_dir, args.classes, args.out_dir)
