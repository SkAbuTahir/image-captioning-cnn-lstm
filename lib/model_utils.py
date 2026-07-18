"""
Shared inference helpers imported by api/predict.py.
Loaded once per cold start via module-level globals.
"""

import os
import io
import json
import pickle
import numpy as np
from PIL import Image

_resnet_model = None
_caption_model = None
_tokenizer = None
_config = None
_idx2word = None

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
KERAS_MODEL = os.path.join(os.path.dirname(__file__), "..", "saved_model", "caption_model.keras")


def load_models():
    global _resnet_model, _caption_model, _tokenizer, _config, _idx2word

    if _resnet_model is not None:
        return

    import tensorflow as tf
    import keras

    _resnet_model = keras.applications.ResNet50(
        weights="imagenet", include_top=False, pooling="avg",
        input_shape=(224, 224, 3))
    _resnet_model.trainable = False

    _caption_model = keras.models.load_model(KERAS_MODEL, compile=False)

    with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb") as f:
        _tokenizer = pickle.load(f)

    with open(os.path.join(MODELS_DIR, "config.json")) as f:
        _config = json.load(f)

    _idx2word = {v: k for k, v in _tokenizer.word_index.items()}


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = arr[..., ::-1]  # RGB -> BGR
    arr[..., 0] -= 103.939
    arr[..., 1] -= 116.779
    arr[..., 2] -= 123.68
    return arr[np.newaxis]  # (1, 224, 224, 3)


def extract_features(image_bytes: bytes) -> np.ndarray:
    load_models()
    img_arr = preprocess_image(image_bytes)
    return _resnet_model(img_arr, training=False).numpy()  # (1, 2048)


def greedy_decode(features: np.ndarray) -> str:
    load_models()
    max_length = _config["max_length"]
    start_id = _tokenizer.word_index.get("startseq", 1)
    seq = [start_id]
    words = []

    for _ in range(max_length):
        seq_pad = np.zeros((1, max_length), dtype=np.int32)
        seq_pad[0, :len(seq)] = seq
        probs = _caption_model.predict(
            [features.astype(np.float32), seq_pad.astype(np.float32)],
            verbose=0)[0]
        next_id = int(np.argmax(probs))
        word = _idx2word.get(next_id, "")
        if word in ("endseq", ""):
            break
        words.append(word)
        seq.append(next_id)

    return " ".join(words)
