interface Props {
  caption: string | null;
  error: string | null;
  loading: boolean;
}

export default function CaptionResult({ caption, error, loading }: Props) {
  if (loading) {
    return (
      <div className="card">
        <p className="loading-msg">
          ⏳ Generating caption — waking up the model on first request after
          inactivity can take up to a minute. Please wait…
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="error-box">⚠️ {error}</div>
      </div>
    );
  }

  if (caption) {
    return (
      <div className="card">
        <div className="caption-box">"{caption}"</div>
      </div>
    );
  }

  return null;
}
