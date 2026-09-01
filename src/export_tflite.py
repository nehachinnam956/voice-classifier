import argparse
import os
import numpy as np
import tensorflow as tf


def representative_dataset_gen(X_sample):
    """Feeds real feature examples for int8 calibration. Pass a handful
    (100-200) of real training feature tensors here for best accuracy."""
    for i in range(min(200, len(X_sample))):
        yield [X_sample[i:i + 1].astype(np.float32)]


def convert(model_path, out_path, quantize, calib_features_path):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if calib_features_path and os.path.exists(calib_features_path):
            X_sample = np.load(calib_features_path)
            converter.representative_dataset = lambda: representative_dataset_gen(X_sample)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            print("Using full int8 quantization with calibration data.")
        else:
            print("No calibration features found — using dynamic-range "
                  "quantization instead (still shrinks the model, less "
                  "aggressive than full int8).")

    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/cnn/cnn_model.keras")
    parser.add_argument("--out", default="models/cnn/model.tflite")
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--calib_features", default="models/cnn/X_features.npy",
                         help="Path to cached training feature tensor for int8 calibration")
    args = parser.parse_args()
    convert(args.model, args.out, args.quantize, args.calib_features)
