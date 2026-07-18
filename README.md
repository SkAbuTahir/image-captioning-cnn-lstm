# Image Captioning — ResNet50 + LSTM

Automatic image captioning in a single Vercel deployment: Next.js frontend + Python serverless inference, no separate backend required.

**Architecture:** Pretrained ResNet50 (frozen, `include_top=False`, `pooling='avg'`) extracts a 2048-d feature vector. An LSTM decoder (Embedding → Dropout → LSTM(256) → merged with image features via `add` → Dense(256, relu) → Dense(vocab_size, softmax)) generates captions word-by-word using greedy decoding.

**[LIVE DEMO LINK]** ← replace after deploying to Vercel

---

## Why TFLite?

Full TensorFlow is 400–600 MB uncompressed. Vercel serverless functions have a hard ~250 MB limit. Converting both models to TFLite and using `tflite-runtime` (~5–10 MB) instead of full TensorFlow cuts the dependency footprint by an order of magnitude, making it possible to run the entire app — frontend and inference — as a single Vercel project with no separate always-on backend.

---

## Dataset

[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — 8,000 images, 5 reference captions each. Not included in this repo (CC0-1.0 licence; download from Kaggle).

---

## Results

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.5617 |
| BLEU-2 | 0.3434 |

Scores are from the original Keras model evaluated on the Flickr8k test split. TFLite conversion with `DEFAULT` optimizations (dynamic-range quantization) may introduce minor numerical differences; in practice the captions are identical or near-identical for most images.

---

## Sample Predictions

**Image 1** — A child playing outdoors:
> "a little girl in a pink dress is running through a field"

**Image 2** — A dog near water:
> "a brown dog is running through the water"

*(From the internship report — actual outputs may vary slightly with TFLite.)*

---

## Local Development

### Prerequisites
- Node.js 18+
- Python 3.9 or 3.11 (matching Vercel's runtime)
- Vercel CLI: `npm i -g vercel`

### Step 1 — Convert models (one-time, run locally)

```bash
pip install tensorflow pillow numpy
python scripts/convert_to_tflite.py
```

This reads `saved_model/caption_model.keras`, converts both models to TFLite, saves them to `models/`, and prints the file sizes. **Run this before anything else** — the API won't work without the `.tflite` files.

### Step 2 — Install and run

```bash
npm install
vercel dev        # runs Next.js + Python /api routes together locally
```

> `vercel dev` is required (not `npm run dev`) to test the Python serverless functions locally. Plain `npm run dev` only starts the Next.js frontend.

---

## Deployment (Vercel)

1. Push this repo to a public GitHub repo named `image-captioning-cnn-lstm`.
2. Import the repo in [vercel.com/new](https://vercel.com/new).
3. Vercel auto-detects Next.js + `/api` Python functions — no extra config needed beyond `vercel.json`.
4. After deploy, verify:
   - `GET /api/health` → `{"status": "ok"}`
   - `POST /api/predict` with a test image → `{"caption": "..."}`
5. Update `[LIVE DEMO LINK]` above with the live URL.

**Python runtime:** Vercel currently supports Python 3.9 and 3.12 for serverless functions. `tflite-runtime` has wheels for 3.9 and 3.11 on Linux x86_64. If Vercel selects 3.12 and `tflite-runtime` fails to install, switch to `ai-edge-litert>=1.0.1` in `requirements.txt` (it's the successor package with 3.12 support).

**Plan B:** If the deployed function exceeds 250 MB or consistently times out on cold starts (>60s), the fallback is to host the Python inference on a small always-on service (Render or Railway free tier) and have the Next.js frontend call it via an environment variable `INFERENCE_API_URL`. The frontend code requires only a one-line change in `app/page.tsx` (`fetch(process.env.NEXT_PUBLIC_INFERENCE_API_URL + "/predict", ...)`).

---

## Project Structure

```
├── api/
│   ├── predict.py          # POST /api/predict — image → caption
│   └── health.py           # GET  /api/health
├── app/                    # Next.js App Router
│   ├── page.tsx
│   ├── layout.tsx
│   ├── globals.css
│   └── components/
│       ├── ImageUploader.tsx
│       ├── CaptionResult.tsx
│       └── LimitationsNote.tsx
├── lib/
│   └── model_utils.py      # TFLite inference helpers
├── models/                 # .tflite files + tokenizer.pkl + config.json
├── scripts/
│   └── convert_to_tflite.py
├── notebooks/
│   └── image_captioning_CNN_LSTM.ipynb
├── requirements.txt        # Python deps for Vercel /api
├── vercel.json
└── package.json
```

---

## Future Work

- **Attention mechanism** — Bahdanau or spatial attention over ResNet feature maps would improve fine-grained detail and reduce generic phrasing.
- **MS COCO scaling** — Training on MS COCO (~330k images, 5 captions each) would substantially improve generalisation and BLEU scores.
- **Beam search** — Replace greedy decoding with beam search (k=3–5) for more fluent captions.

---

## Credits

NIT Jamshedpur, Department of Computer Science & Engineering.
Internship project under the supervision of **Dr. Mayukh Sarkar**.

MIT Licence — see [LICENSE](LICENSE).
