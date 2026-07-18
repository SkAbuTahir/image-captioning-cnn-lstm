"""
One-time conversion script: run locally before deploying.
Converts caption_model.keras and ResNet50 to TFLite.

Usage:
    python scripts/convert_to_tflite.py

Requires: tensorflow, pillow, numpy
Source artifacts expected at: saved_model/caption_model.keras
Output: models/caption_model.tflite, models/resnet50_encoder.tflite
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from PIL import Image

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SOURCE_MODEL = os.path.join(os.path.dirname(__file__), "..", "saved_model", "caption_model.keras")
os.makedirs(MODELS_DIR, exist_ok=True)


def convert_caption_model():
    print("Loading caption_model.keras ...")
    model = tf.keras.models.load_model(SOURCE_MODEL)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # The caption model uses LSTM + Embedding — these are standard ops and
    # should convert cleanly. If you see a "Select TF Ops" error, uncomment:
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,
    #     tf.lite.OpsSet.SELECT_TF_OPS,
    # ]
    # NOTE: enabling SELECT_TF_OPS adds ~30-50 MB to the tflite-runtime wheel
    # and may push the Vercel function over the 250 MB limit. If that happens,
    # see Plan B in the README.

    print("Converting caption model ...")
    tflite_model = converter.convert()

    out_path = os.path.join(MODELS_DIR, "caption_model.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  caption_model.tflite  →  {size_mb:.1f} MB  ({out_path})")
    return model, size_mb


def convert_resnet50():
    print("Loading ResNet50 (imagenet weights) ...")
    resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg",
                      input_shape=(224, 224, 3))
    resnet.trainable = False

    converter = tf.lite.TFLiteConverter.from_keras_model(resnet)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("Converting ResNet50 ...")
    tflite_model = converter.convert()

    out_path = os.path.join(MODELS_DIR, "resnet50_encoder.tflite")
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  resnet50_encoder.tflite  →  {size_mb:.1f} MB  ({out_path})")
    return resnet, size_mb


def sanity_check(keras_caption_model, keras_resnet):
    """Run a dummy forward pass through both Keras and TFLite and compare."""
    import pickle, json

    config_path = os.path.join(os.path.dirname(__file__), "..", "saved_model", "config.json")
    tok_path = os.path.join(os.path.dirname(__file__), "..", "saved_model", "tokenizer.pkl")

    with open(config_path) as f:
        cfg = json.load(f)
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)

    max_length = cfg["max_length"]
    idx2word = {v: k for k, v in tokenizer.word_index.items()}

    # --- dummy image (random noise) ---
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_arr = preprocess_input(dummy_img.astype(np.float32)[np.newaxis])

    # Keras ResNet50 feature
    feat_keras = keras_resnet.predict(img_arr, verbose=0)  # (1, 2048)

    # TFLite ResNet50 feature
    interp_r = tf.lite.Interpreter(
        model_path=os.path.join(MODELS_DIR, "resnet50_encoder.tflite"))
    interp_r.allocate_tensors()
    inp = interp_r.get_input_details()[0]
    out = interp_r.get_output_details()[0]
    interp_r.set_tensor(inp["index"], img_arr)
    interp_r.invoke()
    feat_tflite = interp_r.get_tensor(out["index"])  # (1, 2048)

    feat_diff = np.max(np.abs(feat_keras - feat_tflite))
    print(f"\nSanity check — ResNet50 max feature diff (Keras vs TFLite): {feat_diff:.6f}")
    if feat_diff > 0.1:
        print("  WARNING: large diff — check conversion flags.")

    # Greedy decode with TFLite caption model
    interp_c = tf.lite.Interpreter(
        model_path=os.path.join(MODELS_DIR, "caption_model.tflite"))
    interp_c.allocate_tensors()
    c_inputs = interp_c.get_input_details()
    c_outputs = interp_c.get_output_details()

    # Determine input order: image feature vs sequence
    # The model has two inputs; identify by shape
    for d in c_inputs:
        print(f"  caption model input: name={d['name']}  shape={d['shape']}  dtype={d['dtype']}")

    seq = [tokenizer.word_index.get("startseq", 1)]
    caption_words = []
    for _ in range(max_length):
        seq_pad = np.zeros((1, max_length), dtype=np.int32)
        seq_pad[0, :len(seq)] = seq

        # Set inputs — match by shape: (1,2048) = image, (1,max_length) = seq
        for d in c_inputs:
            if d["shape"][-1] == 2048:
                interp_c.set_tensor(d["index"], feat_tflite.astype(np.float32))
            else:
                interp_c.set_tensor(d["index"], seq_pad)

        interp_c.invoke()
        probs = interp_c.get_tensor(c_outputs[0]["index"])[0]
        next_id = int(np.argmax(probs))
        word = idx2word.get(next_id, "")
        if word == "endseq" or word == "":
            break
        caption_words.append(word)
        seq.append(next_id)

    print(f"  TFLite caption (dummy image): {' '.join(caption_words)}")
    print("\nSanity check complete.")


def main():
    keras_caption, cap_mb = convert_caption_model()
    keras_resnet, res_mb = convert_resnet50()

    total_mb = cap_mb + res_mb
    print(f"\n{'='*50}")
    print(f"Total .tflite size: {total_mb:.1f} MB")
    print(f"  caption_model.tflite:    {cap_mb:.1f} MB")
    print(f"  resnet50_encoder.tflite: {res_mb:.1f} MB")
    print()

    # Vercel limit guidance
    # Vercel uncompressed function limit: ~250 MB (including all deps)
    # tflite-runtime wheel: ~5-10 MB; numpy: ~20 MB; pillow: ~5 MB
    # Rough overhead: ~35 MB deps + model files
    overhead_mb = 35
    estimated_total = total_mb + overhead_mb
    print(f"Estimated deployed function size: ~{estimated_total:.0f} MB "
          f"(models + ~{overhead_mb} MB deps)")
    if estimated_total > 220:
        print("  ⚠️  WARNING: Approaching Vercel's 250 MB limit.")
        print("  Consider Plan B (separate backend) if deployment fails.")
    else:
        print("  ✅ Should fit within Vercel's 250 MB limit.")

    print()
    sanity_check(keras_caption, tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)))


if __name__ == "__main__":
    main()
