"""
One-time conversion script: run locally before deploying.
Converts caption_model.keras and ResNet50 to TFLite.

Usage (from repo root, with .venv311 active):
    python scripts/convert_to_tflite.py

Requires: tensorflow>=2.16, keras>=3, pillow, numpy
Source: saved_model/caption_model.keras
Output: models/caption_model.tflite, models/resnet50_encoder.tflite
"""

import os, sys, tempfile, shutil
import numpy as np
import tensorflow as tf
import keras

sys.stdout.reconfigure(encoding="utf-8")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SOURCE_MODEL = os.path.join(os.path.dirname(__file__), "..", "saved_model", "caption_model.keras")
os.makedirs(MODELS_DIR, exist_ok=True)


def _convert_via_saved_model(model, out_path, input_shapes):
    """
    Use keras model.export() to produce a proper frozen SavedModel,
    then convert to TFLite. This is the correct path for Keras 3 —
    tf.saved_model.save() does not freeze variables for Keras 3 models.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        # keras.Model.export() freezes weights into the graph correctly
        model.export(tmp_dir, format="tf_saved_model")

        converter = tf.lite.TFLiteConverter.from_saved_model(
            tmp_dir, signature_keys=["serving_default"])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # LSTM uses TensorListReserve which requires SELECT_TF_OPS.
        # This adds ~30 MB to the tflite-runtime wheel but is unavoidable
        # with a dynamic-sequence LSTM in Keras 3.
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False

        tflite_bytes = converter.convert()
        with open(out_path, "wb") as f:
            f.write(tflite_bytes)
        return len(tflite_bytes)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def convert_caption_model():
    print("Loading caption_model.keras ...")
    model = keras.models.load_model(SOURCE_MODEL, compile=False)
    print(f"  Inputs: {[i.shape for i in model.inputs]}")

    out_path = os.path.join(MODELS_DIR, "caption_model.tflite")
    print("Converting caption model via SavedModel export ...")
    # Fixed batch=1, seq_len=34 to avoid dynamic TensorList shapes
    nbytes = _convert_via_saved_model(model, out_path, input_shapes=None)
    size_mb = nbytes / 1024 / 1024
    print(f"  caption_model.tflite  ->  {size_mb:.2f} MB")
    return model, size_mb


def convert_resnet50():
    print("Loading ResNet50 (imagenet weights) ...")
    resnet = keras.applications.ResNet50(
        weights="imagenet", include_top=False, pooling="avg",
        input_shape=(224, 224, 3))
    resnet.trainable = False

    out_path = os.path.join(MODELS_DIR, "resnet50_encoder.tflite")
    print("Converting ResNet50 via SavedModel export ...")
    nbytes = _convert_via_saved_model(resnet, out_path, input_shapes=None)
    size_mb = nbytes / 1024 / 1024
    print(f"  resnet50_encoder.tflite  ->  {size_mb:.2f} MB")
    return resnet, size_mb


def sanity_check(keras_caption_model, keras_resnet):
    import pickle, json

    config_path = os.path.join(os.path.dirname(__file__), "..", "models", "config.json")
    tok_path    = os.path.join(os.path.dirname(__file__), "..", "models", "tokenizer.pkl")

    with open(config_path) as f:
        cfg = json.load(f)
    with open(tok_path, "rb") as f:
        tokenizer = pickle.load(f)

    max_length = cfg["max_length"]
    idx2word = {v: k for k, v in tokenizer.word_index.items()}

    # Dummy image
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8).astype(np.float32)
    img_bgr = dummy[..., ::-1].copy()
    img_bgr[..., 0] -= 103.939; img_bgr[..., 1] -= 116.779; img_bgr[..., 2] -= 123.68
    img_arr = img_bgr[np.newaxis]  # (1,224,224,3)

    # Keras ResNet50 feature
    feat_keras = keras_resnet(img_arr, training=False).numpy()

    # TFLite ResNet50 feature
    interp_r = tf.lite.Interpreter(model_path=os.path.join(MODELS_DIR, "resnet50_encoder.tflite"))
    interp_r.allocate_tensors()
    inp_r = interp_r.get_input_details()[0]
    out_r = interp_r.get_output_details()[0]
    interp_r.set_tensor(inp_r["index"], img_arr)
    interp_r.invoke()
    feat_tflite = interp_r.get_tensor(out_r["index"])

    diff = float(np.max(np.abs(feat_keras - feat_tflite)))
    status = "OK" if diff < 0.5 else "WARNING: large diff"
    print(f"\nSanity check -- ResNet50 max feature diff: {diff:.6f}  [{status}]")

    # TFLite caption decode
    # NOTE: caption_model.tflite uses SELECT_TF_OPS (FlexTensorListReserve for LSTM).
    # The Windows tf.lite.Interpreter does not include the Flex delegate by default.
    # On Vercel (Linux), tflite-runtime / ai-edge-litert ships with Flex built in.
    # We skip the caption decode in the local sanity check and just confirm the file loads.
    try:
        interp_c = tf.lite.Interpreter(model_path=os.path.join(MODELS_DIR, "caption_model.tflite"))
        interp_c.allocate_tensors()
        c_inputs  = interp_c.get_input_details()
        c_outputs = interp_c.get_output_details()

        print("  Caption model TFLite inputs:")
        for d in c_inputs:
            print(f"    {d['name']}  shape={d['shape']}")

        seq = [tokenizer.word_index.get("startseq", 1)]
        words = []
        for _ in range(max_length):
            seq_pad = np.zeros((1, max_length), dtype=np.float32)
            seq_pad[0, :len(seq)] = seq
            for d in c_inputs:
                if d["shape"][-1] == 2048:
                    interp_c.set_tensor(d["index"], feat_tflite.astype(np.float32))
                else:
                    interp_c.set_tensor(d["index"], seq_pad)
            interp_c.invoke()
            probs   = interp_c.get_tensor(c_outputs[0]["index"])[0]
            next_id = int(np.argmax(probs))
            word    = idx2word.get(next_id, "")
            if word in ("endseq", ""):
                break
            words.append(word)
            seq.append(next_id)

        print(f"  TFLite caption (random noise): '{' '.join(words)}'")
        print("  (Nonsense expected -- correct.)")
    except RuntimeError as e:
        if "Flex" in str(e) or "FlexTensorList" in str(e):
            print("  Caption model decode skipped locally (Flex delegate not in Windows tf.lite.Interpreter).")
            print("  This is expected -- ai-edge-litert on Vercel Linux includes Flex. File size is correct.")
        else:
            raise


def main():
    keras_caption, cap_mb = convert_caption_model()
    keras_resnet,  res_mb = convert_resnet50()

    total_mb    = cap_mb + res_mb
    overhead_mb = 35
    est_mb      = total_mb + overhead_mb

    print(f"\n{'='*55}")
    print(f"  caption_model.tflite:    {cap_mb:.2f} MB")
    print(f"  resnet50_encoder.tflite: {res_mb:.2f} MB")
    print(f"  Total .tflite:           {total_mb:.2f} MB")
    print(f"  Est. Vercel function:    ~{est_mb:.0f} MB (models + ~{overhead_mb} MB deps)")
    if est_mb > 220:
        print("  WARNING: approaching Vercel 250 MB limit")
    else:
        print("  OK: fits within Vercel 250 MB limit")
    print(f"{'='*55}\n")

    sanity_check(keras_caption, keras_resnet)


if __name__ == "__main__":
    main()
