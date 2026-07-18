"use client";

import { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import CaptionResult from "./components/CaptionResult";
import LimitationsNote from "./components/LimitationsNote";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [caption, setCaption] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function generateCaption() {
    if (!file) return;
    setLoading(true);
    setCaption(null);
    setError(null);

    try {
      const form = new FormData();
      form.append("image", file);

      const res = await fetch("/api/predict", { method: "POST", body: form });
      const data = await res.json();

      if (!res.ok) {
        setError(data.error ?? `Server error (${res.status})`);
      } else {
        setCaption(data.caption);
      }
    } catch {
      setError("Request failed — check your connection or try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Image Captioning</h1>
      <p className="subtitle">
        ResNet50 encoder + LSTM decoder · Flickr8k · BLEU-1 0.56
      </p>

      <div className="card">
        <ImageUploader onFileSelected={setFile} disabled={loading} />
        <div className="btn-row">
          <button
            className="btn btn-primary"
            onClick={generateCaption}
            disabled={!file || loading}
          >
            {loading ? "Generating…" : "Generate Caption"}
          </button>
        </div>
      </div>

      <CaptionResult caption={caption} error={error} loading={loading} />
      <LimitationsNote />
    </main>
  );
}
