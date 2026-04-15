import React from 'react';

/**
 * Decorative hero focal — scan frame and readout (illustrative only).
 */
export default function HeroVisual() {
  return (
    <div className="hero-visual" aria-hidden="true">
      <div className="hero-visual__bloom" />
      <div className="hero-visual__shell">
        <header className="hero-visual__header">
          <span className="hero-visual__header-tag">Sample panel</span>
          <span className="hero-visual__header-live">
            <span className="hero-visual__live-dot" />
            Standby
          </span>
        </header>

        <div className="hero-visual__viewport">
          <svg className="hero-visual__frame" viewBox="0 0 400 260" preserveAspectRatio="xMidYMid slice">
            <path
              d="M 12 32 L 12 12 L 32 12"
              fill="none"
              stroke="rgba(130,175,255,0.65)"
              strokeWidth="1.25"
              strokeLinecap="round"
            />
            <path
              d="M 388 32 L 388 12 L 368 12"
              fill="none"
              stroke="rgba(130,175,255,0.65)"
              strokeWidth="1.25"
              strokeLinecap="round"
            />
            <path
              d="M 12 228 L 12 248 L 32 248"
              fill="none"
              stroke="rgba(130,175,255,0.65)"
              strokeWidth="1.25"
              strokeLinecap="round"
            />
            <path
              d="M 388 228 L 388 248 L 368 248"
              fill="none"
              stroke="rgba(130,175,255,0.65)"
              strokeWidth="1.25"
              strokeLinecap="round"
            />
            <rect x="24" y="24" width="352" height="212" rx="4" fill="none" stroke="rgba(120,160,255,0.12)" strokeWidth="1" strokeDasharray="6 10" />
          </svg>

          <div className="hero-visual__sensor">
            <div className="hero-visual__sensor-grid" />
            <div className="hero-visual__sensor-noise" />
          </div>

          <div className="hero-visual__scanbeam" />
          <div className="hero-visual__crosshair">
            <span />
            <span />
          </div>
        </div>

        <div className="hero-visual__readout">
          <div className="hero-visual__readout-top">
            <span className="hero-visual__readout-label">Calibrated confidence</span>
            <span className="hero-visual__readout-val">72%</span>
          </div>
          <div className="hero-visual__readout-track">
            <div className="hero-visual__readout-fill" />
          </div>
        </div>
      </div>

      <div className="hero-visual__badge">
        <span className="hero-visual__badge-icon" />
        Hybrid inference
      </div>
    </div>
  );
}
