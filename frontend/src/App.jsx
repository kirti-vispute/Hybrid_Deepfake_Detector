import { useEffect, useMemo, useState } from 'react';

import Footer from './components/Footer';
import HeroSection from './components/HeroSection';
import Navbar from './components/Navbar';
import PipelineSection from './components/PipelineSection';
import ResultPanel from './components/ResultPanel';
import UploadPanel from './components/UploadPanel';
import { fetchModelAvailability, predictImage } from './services/api';

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [result, setResult] = useState(null);
  const [apiError, setApiError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [modelsInfo, setModelsInfo] = useState(null);

  useEffect(() => {
    let mounted = true;
    fetchModelAvailability()
      .then((payload) => {
        if (!mounted) return;
        setModelsInfo(payload);
      })
      .catch((error) => {
        if (!mounted) return;
        setApiError(error.message || 'Unable to reach the prediction API. Confirm the server is running.');
      });

    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl('');
      return () => {};
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const canPredict = useMemo(() => Boolean(file) && !isLoading, [file, isLoading]);

  const handlePredict = async () => {
    if (!canPredict) return;

    setApiError('');
    setIsLoading(true);
    setResult(null);

    try {
      const payload = await predictImage({ file });
      setResult(payload);
    } catch (error) {
      setApiError(error.message || 'The prediction request did not complete successfully.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setApiError('');
  };

  return (
    <>
      <Navbar />
      <main className="app-shell">
        <div className="background-mesh" aria-hidden="true" />
        <div className="container-wide">
          <HeroSection />

          <section id="analyze" className="analyze-section section-enter">
            <div className="analyze-section__head">
              <p className="section-eyebrow">Live inference</p>
              <h2 className="section-title">Classify an image</h2>
              <p className="section-desc">
                Upload a still; the service runs hybrid inference and returns the class, confidence, and class probabilities.
              </p>
            </div>
            <div className="grid-two">
              <UploadPanel
                file={file}
                previewUrl={previewUrl}
                onFileChange={setFile}
                onPredict={handlePredict}
                onReset={handleReset}
                isLoading={isLoading}
                apiError={apiError}
                modelsInfo={modelsInfo}
              />

              <ResultPanel result={result} isLoading={isLoading} />
            </div>
          </section>

          <PipelineSection />
        </div>
      </main>
      <Footer />
    </>
  );
}
