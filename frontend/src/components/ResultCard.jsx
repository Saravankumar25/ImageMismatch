import { useEffect, useId, useState } from 'react';
import HeatmapOverlay from './HeatmapOverlay.jsx';

const pct = (v) => `${Math.round(v * 100)}%`;

const MODE_LABELS = {
  A: 'Scene compared with your caption',
  'B-cap': 'Scene (printed text removed) compared with your caption',
  'B-ocr': 'Scene (printed text removed) compared with the text it contained',
};

const reducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

/** Animates 0 -> target once on mount. */
function useCountUp(target, duration = 900) {
  const [value, setValue] = useState(() => (reducedMotion() ? target : 0));
  useEffect(() => {
    if (reducedMotion()) {
      setValue(target);
      return undefined;
    }
    let raf;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      setValue(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    const done = setTimeout(() => setValue(target), duration + 50); // rAF is paused in hidden tabs
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(done);
    };
  }, [target, duration]);
  return value;
}

function VerdictBadge({ verdict }) {
  const match = verdict === 'MATCH';
  return (
    <span className={`badge ${match ? 'badge-match' : 'badge-mismatch'}`}>
      <svg viewBox="0 0 16 16" aria-hidden="true">
        {match ? (
          <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        ) : (
          <path d="M4.5 4.5l7 7m0-7l-7 7" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        )}
      </svg>
      {verdict}
    </span>
  );
}

function ProbabilityBar({ probability, threshold, match }) {
  const [width, setWidth] = useState(reducedMotion() ? probability : 0);
  useEffect(() => {
    const raf = requestAnimationFrame(() => setWidth(probability));
    return () => cancelAnimationFrame(raf);
  }, [probability]);
  return (
    <div
      className="prob-track"
      role="img"
      aria-label={`Match probability ${pct(probability)}; the threshold for a match is ${pct(threshold)}`}
    >
      <div
        className={`prob-fill ${match ? 'is-match' : 'is-mismatch'}`}
        style={{ width: `${Math.max(0, Math.min(1, width)) * 100}%` }}
      />
      <div className="prob-threshold" style={{ left: `${threshold * 100}%` }} aria-hidden="true">
        <span>{pct(threshold)}</span>
      </div>
    </div>
  );
}

export default function ResultCard({ result, best, imageUrl, index }) {
  const [selected, setSelected] = useState(false);
  const heatId = useId();
  const isOcr = result.source === 'ocr';
  const match = result.verdict === 'MATCH';
  const shown = useCountUp(result.match_probability * 100);

  return (
    <article
      className={`rcard ${match ? 'is-match' : 'is-mismatch'}${selected ? ' is-selected' : ''}`}
      style={{ '--delay': `${index * 70}ms` }}
    >
      <header className="rcard-head">
        <span className="rank">{String(result.rank).padStart(2, '0')}</span>
        <span className={`source-chip${isOcr ? ' is-ocr' : ''}`}>
          {isOcr ? (
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M3 4.5h10M8 4.5v8M5.5 12.5h5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          ) : (
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M3 12.5l1-3 6.5-6.5a1.4 1.4 0 0 1 2 2L6 11.5z" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
            </svg>
          )}
          {isOcr ? 'Text found in image' : 'Your caption'}
        </span>
        {best && <span className="best-chip">Best match</span>}
        <VerdictBadge verdict={result.verdict} />
      </header>

      <blockquote className={`rcard-text${isOcr ? ' is-ocr' : ''}`}>{result.text}</blockquote>

      <div className="rcard-score">
        <div className="score-number" aria-hidden="true">
          {Math.round(shown)}
          <span>%</span>
        </div>
        <div className="score-body">
          <div className="score-label">
            Match probability <span className="sr-only">{pct(result.match_probability)}</span>
          </div>
          <ProbabilityBar probability={result.match_probability} threshold={result.threshold} match={match} />
        </div>
      </div>

      <footer className="rcard-foot">
        <dl className="stats">
          <div>
            <dt>Similarity</dt>
            <dd>{result.similarity.toFixed(2)}</dd>
          </div>
          <div>
            <dt>Threshold</dt>
            <dd>{pct(result.threshold)}</dd>
          </div>
          <div title={MODE_LABELS[result.mode]}>
            <dt>Mode</dt>
            <dd>{result.mode}</dd>
          </div>
        </dl>
        <button
          type="button"
          className="btn btn-ghost btn-small heat-btn"
          aria-expanded={selected}
          aria-controls={heatId}
          onClick={() => setSelected((s) => !s)}
        >
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <rect x="2" y="2" width="12" height="12" rx="2.5" fill="none" stroke="currentColor" strokeWidth="1.3" />
            <circle cx="9.5" cy="6.5" r="2.4" fill="currentColor" opacity=".55" />
            <circle cx="6" cy="10" r="1.6" fill="currentColor" opacity=".3" />
          </svg>
          {selected ? 'Hide heatmap' : 'Show heatmap'}
        </button>
      </footer>
      <div id={heatId} className="rcard-heat" hidden={!selected}>
        {selected && imageUrl && (
          <HeatmapOverlay
            src={imageUrl}
            heatmap={result.heatmap}
            label={result.text}
            alt={`Uploaded image with the attention heatmap for “${result.text}”`}
          />
        )}
      </div>
    </article>
  );
}
