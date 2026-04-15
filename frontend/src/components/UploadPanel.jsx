import React, { useCallback, useRef, useState } from 'react';

const ACCEPT = 'image/png,image/jpeg,image/jpg,image/webp,image/bmp';

function pickFileFromList(fileList) {
  const f = fileList?.[0];
  if (!f || !f.type.startsWith('image/')) return null;
  return f;
}

export default function UploadPanel({
  file,
  previewUrl,
  onFileChange,
  onPredict,
  onReset,
  isLoading,
  apiError,
  modelsInfo,
}) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const available = modelsInfo?.available || {};
  const hybridReady = Boolean(available.hybrid || available.classical);

  const onInputChange = (event) => {
    const chosenFile = event.target.files?.[0] || null;
    onFileChange(chosenFile);
  };

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      if (isLoading) return;
      const dropped = pickFileFromList(e.dataTransfer?.files);
      if (dropped) onFileChange(dropped);
    },
    [isLoading, onFileChange],
  );

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  }, []);

  const openPicker = () => {
    if (!isLoading) inputRef.current?.click();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      openPicker();
    }
  };

  return (
    <section className="panel upload-panel glass-panel" aria-labelledby="upload-heading">
      <div className="panel-header">
        <h2 id="upload-heading">Image upload</h2>
        <p>Use PNG, JPEG, WebP, or BMP files for inference.</p>
      </div>

      <div
        className={`upload-zone ${dragActive ? 'upload-zone--drag' : ''}`}
        role="button"
        tabIndex={0}
        aria-label="Select an image by clicking or dropping a file."
        onClick={openPicker}
        onKeyDown={handleKeyDown}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          id="image-upload"
          type="file"
          accept={ACCEPT}
          onChange={onInputChange}
          disabled={isLoading}
          className="upload-zone__input"
        />
        {previewUrl ? (
          <div className="preview-wrap">
            <img src={previewUrl} alt="" className="preview" />
            <span className="preview-wrap__hint">Click or drop to replace the image.</span>
          </div>
        ) : (
          <div className="upload-empty">
            <span className="upload-icon" aria-hidden="true">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path
                  d="M12 16V8m0 0l-3 3m3-3l3 3M4 18h16"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </span>
            <strong>Drop an image here</strong>
            <span className="upload-empty__meta">Or choose a file from your device.</span>
          </div>
        )}
      </div>

      <div className="status-row">
        <span className={`status-pill ${hybridReady ? 'status-pill--ok' : 'status-pill--warn'}`}>
          <span className="status-pill__dot" aria-hidden="true" />
          {hybridReady
            ? (available.hybrid ? 'Hybrid model ready' : 'Hybrid model ready (classical fallback)')
            : 'Hybrid model unavailable'}
        </span>
      </div>

      <div className="actions">
        <button
          type="button"
          className="btn btn-primary"
          onClick={onPredict}
          disabled={isLoading || !file || !hybridReady}
        >
          {isLoading ? (
            <>
              <span className="btn-spinner" aria-hidden="true" />
              Running inference...
            </>
          ) : (
            <>Run inference</>
          )}
        </button>
        <button type="button" className="btn btn-secondary" onClick={onReset} disabled={isLoading}>
          Clear upload
        </button>
      </div>

      {file ? (
        <p className="file-name">
          <span className="file-name__label">Selected file</span>
          <strong className="file-name__value">{file.name}</strong>
        </p>
      ) : null}

      {apiError ? (
        <div className="error-banner" role="alert">
          {apiError}
        </div>
      ) : null}
    </section>
  );
}

