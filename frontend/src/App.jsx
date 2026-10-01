import { useEffect, useRef, useState } from 'react';
import { ApiError, messageFor, predict } from './api.js';
import AnalyzeButton from './components/AnalyzeButton.jsx';
import CaptionList, { MAX_CHARS, newCaption } from './components/CaptionList.jsx';
import ErrorBanner from './components/ErrorBanner.jsx';
import HeatmapOverlay from './components/HeatmapOverlay.jsx';
import ImageDropzone, { validateImageFile } from './components/ImageDropzone.jsx';
import ModelInfoModal from './components/ModelInfoModal.jsx';
import ResultCard from './components/ResultCard.jsx';
import StatusPill from './components/StatusPill.jsx';

const EXAMPLES = [
  {
    file: 'pizza.jpg',
    label: 'Pizza',
    captions: [
      'Large pizza covered in pepperoni, olives, peppers, onion, and mushrooms.',
      'Large apple covered in pepperoni, olives, peppers, onion, and mushrooms.',
      'A man riding a surfboard on a large wave.',
    ],
  },
  {
    file: 'coffee-note.jpg',
    label: 'Note on a coffee pot',
    captions: [
      'A green note on a coffee pot says that it has a defect.',
      'A blue book called Be Here Now sits with some other books on a shelf.',
    ],
  },
  { file: 'licence-plate.jpg', label: 'Licence plate', captions: [] },
];

function Logo() {
  return (
    <svg className="logo" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="2.5" y="2.5" width="12" height="12" fill="currentColor" />
      <rect x="9.5" y="9.5" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

export default function App() {
  const [image, setImage] = useState(null); // { file, url, width, height }
  const [captions, setCaptions] = useState(() => [newCaption()]);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null); // { data, hadCaptions }
  const [error, setError] = useState(null);
  const [aboutOpen, setAboutOpen] = useState(false);
  const abortRef = useRef(null);
  const zoneRef = useRef(null);
  const resultsRef = useRef(null);

  // Revoke the preview URL whenever the image changes or the app unmounts.
  useEffect(() => () => image && URL.revokeObjectURL(image.url), [image]);

  function selectImage(next) {
    setImage(next);
    setResponse(null); // results belong to the previous image
    setError(null);
  }

  function rejectImage(code, message) {
    setError(message || messageFor(code));
    setResponse(null);
  }

  async function loadExample(example) {
    if (loading) return;
    try {
      const blob = await (await fetch(`/examples/${example.file}`)).blob();
      const result = await validateImageFile(new File([blob], example.file, { type: blob.type || 'image/jpeg' }));
      if (result.error) return rejectImage(result.error, result.message);
      selectImage(result);
      setCaptions(example.captions.length ? example.captions.map((t) => newCaption(t)) : [newCaption()]);
    } catch {
      setError(messageFor('NETWORK'));
    }
  }

  async function analyze(e) {
    e.preventDefault();
    if (!image || loading) return;
    const texts = captions.map((c) => c.text.trim()).filter(Boolean);
    if (texts.some((t) => t.length > MAX_CHARS)) {
      setError(messageFor('CAPTION_TOO_LONG'));
      setResponse(null);
      return;
    }
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError(null);
    requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }));
    try {
      const data = await predict(image.file, texts, { signal: controller.signal });
      setResponse({ data, hadCaptions: texts.length > 0 });
    } catch (err) {
      if (err instanceof ApiError && err.code === 'CANCELLED') return;
      setError(err instanceof ApiError ? err.message : messageFor('UNKNOWN'));
      setResponse(null);
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setLoading(false);
      }
    }
  }

  function reset() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
    setImage(null);
    setCaptions([newCaption()]);
    setResponse(null);
    setError(null);
    zoneRef.current?.focus();
  }

  const data = response?.data;
  const matches = data?.results.filter((r) => r.verdict === 'MATCH').length ?? 0;
  const showResults = image && (loading || data);
  const captionCount = captions.filter((c) => c.text.trim()).length;

  return (
    <div className="app">
      <nav className="nav">
        <div className="nav-inner">
          <a className="brand" href="/" aria-label="Image–Text Match Checker home">
            <Logo />
            <span>Match Checker</span>
          </a>
          <div className="nav-right">
            <StatusPill />
            <button type="button" className="nav-link" onClick={() => setAboutOpen(true)}>
              About the model
            </button>
          </div>
        </div>
      </nav>

      <div className="shell">
        <header className="hero">
          <h1>Image–Text Match Checker</h1>
          <p className="hero-sub">
            Check whether an image and text agree. Describe a photo in your own words, or leave the caption empty and
            the text printed in the image is checked instead.
          </p>
          <p className="hero-spec">
            <span>CLIP ViT-B/32</span>
            <span>Cross-attention head</span>
            <span>Calibrated per mode</span>
            <span>English</span>
          </p>
        </header>

        <main>
          <form className="workspace card" onSubmit={analyze} noValidate aria-busy={loading || undefined}>
            <section className="ws-col" aria-labelledby="step-image">
              <div className="step-head">
                <span className="step-num">1</span>
                <h2 id="step-image">Image</h2>
              </div>
              <ImageDropzone
                image={image}
                onSelect={selectImage}
                onReject={rejectImage}
                disabled={loading}
                zoneRef={zoneRef}
              />
              <div className="examples">
                <span className="examples-label">Try an example</span>
                <div className="examples-list">
                  {EXAMPLES.map((ex) => (
                    <button
                      type="button"
                      key={ex.file}
                      className="example"
                      onClick={() => loadExample(ex)}
                      disabled={loading}
                      aria-label={`Load example: ${ex.label}`}
                      title={ex.label}
                    >
                      <img src={`/examples/${ex.file}`} alt="" loading="lazy" />
                    </button>
                  ))}
                </div>
              </div>
            </section>

            <section className="ws-col" aria-labelledby="step-captions">
              <div className="step-head">
                <span className="step-num">2</span>
                <h2 id="step-captions">
                  Captions <span className="step-optional">optional if the image has text</span>
                </h2>
              </div>
              <CaptionList captions={captions} onChange={setCaptions} disabled={loading} />

              <div className="actions">
                <AnalyzeButton loading={loading} disabled={!image} />
                <button type="button" className="btn btn-ghost" onClick={reset}>
                  Reset
                </button>
              </div>
              <p className="actions-hint">
                {image
                  ? captionCount
                    ? `Ready to compare ${captionCount} caption${captionCount > 1 ? 's' : ''} with the image.`
                    : 'No captions — we’ll check the text found in the image.'
                  : 'Add an image to start.'}
              </p>
              <ErrorBanner message={error} onDismiss={() => setError(null)} />
            </section>
          </form>

          <section
            className="results"
            aria-live="polite"
            aria-busy={loading || undefined}
            aria-labelledby="results-title"
            ref={resultsRef}
          >
            {showResults && (
              <>
                <div className="results-head">
                  <h2 id="results-title">{loading ? 'Analyzing…' : 'Results'}</h2>
                  {data && !loading && (
                    <p className="summary">
                      <span>
                        {matches} match{matches === 1 ? '' : 'es'}
                      </span>
                      <span>
                        {data.results.length - matches} mismatch{data.results.length - matches === 1 ? '' : 'es'}
                      </span>
                      <span>{(data.latency_ms / 1000).toFixed(1)} s</span>
                      <span title="Request ID">req {data.request_id}</span>
                    </p>
                  )}
                </div>

                <div className="results-body">
                    {loading && (
                      <>
                        <div className="loading-viewer">
                          <HeatmapOverlay src={image.url} heatmap={null} scanning alt={`Uploaded image, ${image.file.name}`} />
                        </div>
                        <div className="skeletons" aria-label="Loading results">
                          {[0, 1].map((i) => (
                            <div className="rcard skeleton" key={i}>
                              <span className="skeleton-line short" />
                              <span className="skeleton-line" />
                              <span className="skeleton-line tall" />
                            </div>
                          ))}
                        </div>
                      </>
                    )}

                    {data && !loading && (
                      <>
                        {data.text_detected && response.hadCaptions && (
                          <p className="note">
                            <svg viewBox="0 0 16 16" aria-hidden="true">
                              <circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" strokeWidth="1.3" />
                              <path d="M8 7.2v4M8 4.9v.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                            </svg>
                            Text detected in the image was removed before comparing, so the model judges the scene, not
                            the printed words.
                          </p>
                        )}
                        {data.warnings.map((w) => (
                          <p className="note note-warn" key={w.code + w.message}>
                            {w.message}
                          </p>
                        ))}
                        <ol className="result-list">
                          {data.results.map((r, i) => (
                            <li key={`${r.source}-${r.rank}-${r.text}`}>
                              <ResultCard
                                result={r}
                                index={i}
                                best={i === 0 && data.results.length > 1 && r.verdict === 'MATCH'}
                                imageUrl={image.url}
                              />
                            </li>
                          ))}
                        </ol>
                      </>
                    )}
                </div>
              </>
            )}
          </section>
        </main>

        <footer className="footer">
          <div className="footer-brand">
            <Logo />
            <span>Image–Text Match Checker</span>
          </div>
          <div className="footer-links">
            <button type="button" className="link-btn" onClick={() => setAboutOpen(true)}>
              About the model
            </button>
            <span className="muted">Images are analysed in memory and never stored.</span>
          </div>
        </footer>
      </div>

      <ModelInfoModal open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  );
}
