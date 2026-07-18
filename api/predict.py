"""
Vercel Python Serverless Function — POST /api/predict
Accepts multipart/form-data with field "image", returns {"caption": "..."}.
"""

import sys
import os
import json
import cgi
import io
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.model_utils import extract_features, greedy_decode


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_type = self.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            self._json(400, {"error": "Expected multipart/form-data"})
            return

        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)

        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": content_type,
            "CONTENT_LENGTH": str(length),
        }
        form = cgi.FieldStorage(
            fp=io.BytesIO(body),
            environ=environ,
            keep_blank_values=True,
        )

        if "image" not in form:
            self._json(400, {"error": "Missing 'image' field"})
            return

        image_bytes = form["image"].file.read()
        if not image_bytes:
            self._json(400, {"error": "Empty image"})
            return

        try:
            features = extract_features(image_bytes)
            caption = greedy_decode(features)
            self._json(200, {"caption": caption})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def do_GET(self):
        self._json(405, {"error": "Method not allowed"})

    def _json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
