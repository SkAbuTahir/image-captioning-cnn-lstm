"""
Shared inference helpers imported by api/predict.py.
Loaded once per cold start via module-level globals.
"""

import os
import json
import pickle
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy-loaded globals (populated on first call to load_models())
# ---------------------------------------------------------------------------
_resnet_interp = None
_caption_interp = None
_tokenizer = None
_config = None
_idx2word = None
_c_inputs = None
_c_outputs = None

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_models():
    global _resnet_interp, _caption_interp, _tokenizer, _config
    global _idx2word, _c_inputs, _c_outputs

    if _resnet_interp is not None:
        return  # already loaded

    try:
        from ai_edge_litert.interpreter import Interpreter
        from ai_edge_litert.interpreter import OpResolverType
        _resnet_interp = Interpreter(
            model_path=os.path.join(MODELS_DIR, "resnet50_encoder.tflite"))
        _caption_interp = Interpreter(
            model_path=os.path.join(MODELS_DIR, "caption_model.tflite"),
            experimental_op_resolver_type=OpResolverType.AUTO)
    except Exception:
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
            flex = load_delegate("libflexdelegate.so")
            _resnet_interp = Interpreter(
                model_path=os.path.join(MODELS_DIR, "resnet50_encoder.tflite"))
            _caption_interp = Interpreter(
                model_path=os.path.join(MODELS_DIR, "caption_model.tflite"),
                experimental_delegates=[flex])
        except Exception:
            import tensorflow as tf
            _resnet_interp = tf.lite.Interpreter(
                model_path=os.path.join(MODELS_DIR, "resnet50_encoder.tflite"))
            _caption_interp = tf.lite.Interpreter(
                model_path=os.path.join(MODELS_DIR, "caption_model.tflite"))

    _resnet_interp.allocate_tensors()
    _caption_interp.allocate_tensors()
    _c_inputs = _caption_interp.get_input_details()
    _c_outputs = _caption_interp.get_output_details()

    with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb") as f:
        _tokenizer = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "config.json")) as f:
        _config = json.load(f)

    _idx2word = {v: k for k, v in _tokenizer.word_index.items()}


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize to 224×224, apply ResNet50 preprocess_input, return (1,224,224,3)."""
    img = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    # ResNet50 preprocess_input: BGR channel swap + mean subtraction
    arr = arr[..., ::-1]  # RGB -> BGR
    arr[..., 0] -= 103.939
    arr[..., 1] -= 116.779
    arr[..., 2] -= 123.68
    return arr[np.newaxis]  # (1, 224, 224, 3)


def extract_features(image_bytes: bytes) -> np.ndarray:
    """Run ResNet50 TFLite interpreter → (1, 2048) feature vector."""
    load_models()
    img_arr = preprocess_image(image_bytes)
    inp = _resnet_interp.get_input_details()[0]
    out = _resnet_interp.get_output_details()[0]
    _resnet_interp.set_tensor(inp["index"], img_arr)
    _resnet_interp.invoke()
    return _resnet_interp.get_tensor(out["index"])  # (1, 2048)


def greedy_decode(features: np.ndarray) -> str:
    """Greedy decode using caption model TFLite interpreter."""
    load_models()
    max_length = _config["max_length"]
    start_id = _tokenizer.word_index.get("startseq", 1)
    seq = [start_id]
    words = []

    for _ in range(max_length):
        seq_pad = np.zeros((1, max_length), dtype=np.int32)
        seq_pad[0, : len(seq)] = seq

        for d in _c_inputs:
            if d["shape"][-1] == 2048:
                _caption_interp.set_tensor(d["index"], features.astype(np.float32))
            else:
                _caption_interp.set_tensor(d["index"], seq_pad)

        _caption_interp.invoke()
        probs = _caption_interp.get_tensor(_c_outputs[0]["index"])[0]
        next_id = int(np.argmax(probs))
        word = _idx2word.get(next_id, "")
        if word in ("endseq", ""):
            break
        words.append(word)
        seq.append(next_id)

    return " ".join(words)
