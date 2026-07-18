export default function LimitationsNote() {
  return (
    <div className="card">
      <details className="limitations">
        <summary>ℹ️ Model limitations (click to expand)</summary>
        <div className="limitations-body">
          <p>This model was trained on Flickr8k (8,000 images) and has known limitations:</p>
          <ul>
            <li>
              <strong>Gender misclassification</strong> — the model may assign incorrect
              gender pronouns when people are partially visible or in ambiguous poses.
            </li>
            <li>
              <strong>Generic phrasing</strong> — captions tend toward common Flickr8k
              patterns (e.g. "a dog is running in a field") and may not capture
              scene-specific detail.
            </li>
            <li>
              <strong>Fine-grained detail</strong> — breed, colour, and object-count
              accuracy is limited by the small training set and the absence of an
              attention mechanism.
            </li>
            <li>
              <strong>Out-of-distribution images</strong> — indoor scenes, abstract art,
              or images very different from Flickr8k content may produce poor captions.
            </li>
          </ul>
          <p style={{ marginTop: "0.5rem" }}>
            BLEU-1: 0.5617 · BLEU-2: 0.3434 (original Keras model, Flickr8k test set).
            TFLite conversion may introduce minor numerical differences.
          </p>
        </div>
      </details>
    </div>
  );
}
