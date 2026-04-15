import React from 'react';

const links = [
  { href: '#analyze', label: 'Inference' },
  { href: '#how-it-works', label: 'How it works' },
  { href: '#stack', label: 'Stack' },
];

export default function Navbar() {
  return (
    <header className="nav-shell nav-shell--cinema">
      <div className="nav-inner nav-inner--cinema">
        <a href="#" className="nav-brand nav-brand--cinema" aria-label="Hybrid Deepfake Detector home">
          <div className="nav-brand__glyph" aria-hidden="true">
            <span className="nav-brand__glyph-inner" />
          </div>
          <div className="nav-brand__lockup">
            <span className="nav-brand__name">Hybrid Deepfake</span>
            <span className="nav-brand__desc">Detector</span>
          </div>
        </a>

        <nav className="nav-links nav-links--cinema" aria-label="Primary">
          {links.map(({ href, label }) => (
            <a key={href} href={href} className="nav-link nav-link--cinema">
              {label}
            </a>
          ))}
        </nav>

        <a href="#analyze" className="btn-nav-cta btn-nav-cta--cinema">
          <span className="btn-nav-cta__glow" aria-hidden="true" />
          <span>Run inference</span>
        </a>
      </div>
    </header>
  );
}
