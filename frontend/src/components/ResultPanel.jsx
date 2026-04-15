import React, { useEffect, useState } from 'react';

import Spinner from './ui/Spinner';

function ConfidenceBar({ label, value, variant }) {
  const percent = Math.max(0, Math.min(100, Number(value || 0) * 100));
  return (
    <div className={`confidence-row confidence-row--${variant}`}>
      <div className="confidence-label">
        <span>{label}</span>
        <span className="confidence-label__pct">{percent.toFixed(1)}%</span>
      </div>
      <div className="confidence-track">
        <div className="confidence-fill" style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

export default function ResultPanel({ result, isLoading }) {
  const [showResult, setShowResult] = useState(false);

  useEffect(() => {
    if (!result || isLoading) {
      setShowResult(false);
      return;
    }
    const t = requestAnimationFrame(() => setShowResult(true));
    return () => cancelAnimationFrame(t);
  }, [result, isLoading]);

  const verdict = result?.predicted_class || '';
  const isReal = verdict === 'Real';
  const isUncertain = verdict === 'Uncertain';

  return (
    <section className="panel result-panel glass-panel" aria-labelledby="result-heading" aria-live="polite">
      <div className="panel-header">
        <h2 id="result-heading">Classification</h2>
      </div>

      {isLoading ? (
        <div className="result-loading glass-card-subtle">
          <Spinner label="Running inference" />
          <p className="result-loading__text">Running inference…</p>
        </div>
      ) : null}

      {!isLoading && !result ? (
        <div className="result-empty glass-card-subtle">
          <span className="result-empty__icon" aria-hidden="true">
            ◇
          </span>
          <p className="result-empty__title">No classification yet</p>
          <p className="result-empty__hint">Upload an image and run inference to see the verdict.</p>
        </div>
      ) : null}

      {!isLoading && result ? (
        <div
          className={`result-card ${
            isUncertain ? 'result-card--uncertain' : isReal ? 'result-card--real' : 'result-card--fake'
          } ${showResult ? 'result-card--visible' : ''}`}
        >
          <div className="result-card__glow" aria-hidden="true" />

          <div className="result-verdict-row">
            <span
              className={`result-verdict ${
                isUncertain ? 'result-verdict--uncertain' : isReal ? 'result-verdict--real' : 'result-verdict--fake'
              }`}
            >
              <span className="result-verdict__icon" aria-hidden="true">
                {isUncertain ? '?' : isReal ? '✓' : '⚠'}
              </span>
              {result.predicted_class}
            </span>
          </div>

          {result.preprocessing ? (
            <p className="result-preprocess-hint" style={{ opacity: 0.85, fontSize: '0.9rem', marginBottom: '0.75rem' }}>
              Preprocess:{' '}
              {result.preprocessing.face_detected ? 'face detected (Haar)' : 'center crop fallback'}
              {result.preprocessing.method ? ` · ${result.preprocessing.method}` : ''}
            </p>
          ) : null}

          <div className="confidence-meter confidence-meter--primary">
            <div className="confidence-meter__head">
              <span>Prediction confidence</span>
              <span className="confidence-meter__value">{(Number(result.confidence) * 100).toFixed(1)}%</span>
            </div>
            <div className="confidence-track confidence-track--lg">
              <div
                className={`confidence-fill confidence-fill--${
                  isUncertain ? 'uncertain' : isReal ? 'real' : 'fake'
                }`}
                style={{ width: `${Math.min(100, Math.max(0, Number(result.confidence || 0) * 100))}%` }}
              />
            </div>
          </div>

          {result.probabilities ? (
            <div className="probability-section">
              <ConfidenceBar label="Real probability" value={result.probabilities.real} variant="real" />
              <ConfidenceBar label="Fake probability" value={result.probabilities.fake} variant="fake" />
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
