"""
Vercel Python Serverless Function — POST /api/predict
Accepts multipart/form-data with field "image", returns {"caption": "..."}.
"""

import sys
import os
import json
import cgi
import io

# Make lib/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.model_utils import extract_features, greedy_decode


def handler(request, response):
    if request.method != "POST":
        response.status_code = 405
        return response.json({"error": "Method not allowed"})

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        response.status_code = 400
        return response.json({"error": "Expected multipart/form-data"})

    try:
        # Parse multipart body
        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": str(len(request.body)),
        }
        form = cgi.FieldStorage(
            fp=io.BytesIO(request.body),
            environ=environ,
            keep_blank_values=True,
        )

        if "image" not in form:
            response.status_code = 400
            return response.json({"error": "Missing 'image' field"})

        image_bytes = form["image"].file.read()
        if not image_bytes:
            response.status_code = 400
            return response.json({"error": "Empty image"})

        features = extract_features(image_bytes)
        caption = greedy_decode(features)

        return response.json({"caption": caption})

    except Exception as exc:
        response.status_code = 500
        return response.json({"error": str(exc)})
