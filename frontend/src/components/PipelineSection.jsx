import React from 'react';

const steps = [
  {
    n: '01',
    title: 'Embedding',
    body: 'The image is resized and normalized, then encoded by the CNN into a fixed-length embedding vector.',
  },
  {
    n: '02',
    title: 'Classification',
    body: 'Embeddings are scored by a gradient-boosted classifier with scaling and calibration fit on validation data.',
  },
  {
    n: '03',
    title: 'Response',
    body: 'The API returns the predicted class, confidence, and per-class probabilities for manual review when needed.',
  },
];

const features = [
  {
    title: 'Hybrid architecture',
    body: 'CNN embeddings plus a tabular classifier often generalize better than either stage alone on deepfake-style shifts.',
  },
  {
    title: 'Calibrated outputs',
    body: 'Confidence is calibrated so uncertain predictions are easier to identify and escalate.',
  },
];

const stack = [
  { k: 'Frontend', v: 'React · Vite' },
  { k: 'API', v: 'Flask · REST' },
  { k: 'ML / Vision', v: 'TensorFlow · Keras · XGBoost' },
];

export default function PipelineSection() {
  return (
    <>
      <section id="how-it-works" className="section-block">
        <div className="section-head">
          <p className="section-eyebrow">Pipeline</p>
          <h2 className="section-title">How it works</h2>
          <p className="section-desc">
            Each request moves from upload to inference. The CNN produces embeddings, the hybrid head performs classification, and the response includes calibrated confidence.
          </p>
        </div>
        <div className="steps-grid">
          {steps.map((step) => (
            <article key={step.n} className="step-card glass-card-subtle">
              <span className="step-card__n">{step.n}</span>
              <h3 className="step-card__title">{step.title}</h3>
              <p className="step-card__body">{step.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section-block section-block--tight">
        <div className="section-head section-head--row-wrap">
          <div>
            <p className="section-eyebrow">Model</p>
            <h2 className="section-title">Why hybrid</h2>
          </div>
        </div>
        <div className="features-grid">
          {features.map((f) => (
            <article key={f.title} className="feature-card glass-card-subtle">
              <div className="feature-card__accent" aria-hidden="true" />
              <h3 className="feature-card__title">{f.title}</h3>
              <p className="feature-card__body">{f.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="stack" className="section-block section-block--last">
        <div className="section-head">
          <p className="section-eyebrow">Technology</p>
          <h2 className="section-title">Stack</h2>
        </div>
        <div className="stack-row glass-card-subtle">
          {stack.map((row, i) => (
            <div key={`${row.k}-${i}`} className="stack-item">
              <span className="stack-item__k">{row.k}</span>
              <span className="stack-item__v">{row.v}</span>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
