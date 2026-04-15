import React from 'react';

export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="container-wide site-footer__inner">
        <div className="site-footer__brand">
          <span className="nav-brand__mark site-footer__mark" aria-hidden="true" />
          <div>
            <p className="site-footer__name">Hybrid Deepfake Detector</p>
            <p className="site-footer__tag">Built on CNN embeddings, hybrid classification, and calibrated confidence.</p>
          </div>
        </div>
        <p className="site-footer__note">
          Intended for research and education. Validate consequential decisions with independent human review.
        </p>
      </div>
    </footer>
  );
}
