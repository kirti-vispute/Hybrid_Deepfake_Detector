import React from 'react';

import HeroVisual from './HeroVisual';

const metrics = [
  {
    title: 'Hybrid inference',
    hint: 'CNN embeddings are classified with a gradient-boosted model.',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M4 19V5M4 19h16M8 15l3-4 4 5 5-8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    title: 'Calibrated confidence',
    hint: 'Outputs are calibrated so scores track real uncertainty.',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M12 3v3M12 18v3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M3 12h3M18 12h3M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="1.2" />
      </svg>
    ),
  },
  {
    title: 'REST API',
    hint: 'Integrate classification and confidence into your own systems.',
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <path d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2M10 8h8a2 2 0 012 2v8a2 2 0 01-2 2h-8a2 2 0 01-2-2v-8a2 2 0 012-2z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
      </svg>
    ),
  },
];

export default function HeroSection() {
  return (
    <section className="hero-cinematic" aria-labelledby="hero-heading">
      <div className="hero-cinematic__wash" aria-hidden="true" />
      <div className="hero-cinematic__grid">
        <div className="hero-copy">
          <p className="hero-eyebrow">
            <span className="hero-eyebrow__rule" />
            <span className="hero-eyebrow__text">Hybrid deepfake detection</span>
          </p>

          <h1 id="hero-heading" className="hero-title">
            <span className="hero-title__line">Truth in pixels.</span>
            <span className="hero-title__line hero-title__line--emphasis">
              Skepticism <span className="hero-title__muted">engineered in.</span>
            </span>
          </h1>

          <p className="hero-lede hero-lede--cinema">
            We classify uploaded images as authentic or manipulated using CNN embeddings and hybrid scoring.
            <br />
            Each inference returns a label, calibrated confidence, and class probabilities.
          </p>

          <div className="hero-actions hero-actions--cinema">
            <a href="#analyze" className="btn btn-hero-primary">
              <span className="btn-hero-primary__shine" aria-hidden="true" />
              <span className="btn-hero-primary__label">Run inference</span>
              <svg className="btn-hero-primary__arrow" width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </a>
            <a href="#how-it-works" className="btn btn-hero-secondary">
              How it works
            </a>
          </div>
        </div>

        <div className="hero-visual-col">
          <HeroVisual />
        </div>
      </div>

      <ul className="hero-metrics" aria-label="Product highlights">
        {metrics.map((m) => (
          <li key={m.title} className="hero-metric-card">
            <div className="hero-metric-card__icon">{m.icon}</div>
            <div className="hero-metric-card__body">
              <strong className="hero-metric-card__title">{m.title}</strong>
              <span className="hero-metric-card__hint">{m.hint}</span>
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
