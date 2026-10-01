import { useEffect, useRef, useState } from 'react';
import { getModelInfo } from '../api.js';

const MODES = [
  ['A', 'Caption vs. photo', 'Images without text'],
  ['B-cap', 'Caption vs. scene', 'Printed text removed first'],
  ['B-ocr', 'Image text vs. scene', 'Does the printed text fit?'],
];
const fmt = (v, d = 2) => (typeof v === 'number' ? v.toFixed(d) : '–');
const pct = (v) => (typeof v === 'number' ? `${Math.round(v * 100)}%` : '–');

export default function ModelInfoModal({ open, onClose }) {
  const ref = useRef(null);
  const [info, setInfo] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const dlg = ref.current;
    if (!dlg) return;
    if (open && !dlg.open) dlg.showModal();
    if (!open && dlg.open) dlg.close();
  }, [open]);

  useEffect(() => {
    if (!open || info) return;
    setFailed(false);
    getModelInfo()
      .then(setInfo)
      .catch(() => setFailed(true));
  }, [open, info]);

  return (
    <dialog
      ref={ref}
      className="modal"
      aria-labelledby="model-info-title"
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()} // backdrop click
    >
      <div className="modal-body">
        <header className="modal-head">
          <div>
            <span className="eyebrow-text">Model card</span>
            <h2 id="model-info-title">About the model</h2>
          </div>
          <button type="button" className="icon-btn" aria-label="Close" onClick={onClose} autoFocus>
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 6l12 12M18 6L6 18" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
        </header>

        {failed && <p className="modal-error">Could not load model information. Please try again.</p>}
        {!info && !failed && (
          <div className="modal-skeleton" aria-label="Loading">
            <span className="skeleton-line" />
            <span className="skeleton-line short" />
          </div>
        )}

        {info && (
          <>
            <p className="modal-lede">
              A frozen <code>{info.clip_id}</code> encoder feeds a{' '}
              {(info.head_parameters / 1e6).toFixed(1)}M-parameter cross-attention head that decides whether an image
              and a piece of text agree. Scores are calibrated per mode. Printed text is detected with OCR and removed
              first, so the model judges the scene.
            </p>

            <div className="metric-grid">
              {MODES.map(([mode, title, sub]) => {
                const m = info.test_metrics?.[mode] || {};
                const base = info.baseline_cosine_roc_auc?.[mode];
                const delta = typeof m.roc_auc === 'number' && typeof base === 'number' ? m.roc_auc - base : null;
                return (
                  <div className="metric-card" key={mode}>
                    <div className="metric-top">
                      <span className="metric-mode">{mode}</span>
                      {delta != null && <span className="metric-delta">+{fmt(delta)} vs CLIP</span>}
                    </div>
                    <div className="metric-value">{fmt(m.roc_auc)}</div>
                    <div className="metric-name">ROC-AUC · {title}</div>
                    <div className="metric-sub">{sub}</div>
                    <div className="metric-row">
                      <span>Acc {pct(m.accuracy)}</span>
                      <span>F1 {fmt(m.f1)}</span>
                      <span>Thr {pct(info.thresholds?.[mode])}</span>
                    </div>
                  </div>
                );
              })}
            </div>
            <p className="muted small">
              Held-out test set of {info.counts?.test_pairs?.toLocaleString() ?? '—'} pairs, including hard negatives and
              swapped details (colours, numbers, people, animals, vehicles, food). “vs CLIP” compares against raw CLIP
              cosine similarity.
            </p>

            <h3>Reading a result</h3>
            <dl className="defs">
              <dt>Match probability</dt>
              <dd>Calibrated chance the text describes the image. MATCH once it reaches the threshold marker.</dd>
              <dt>Similarity</dt>
              <dd>Raw CLIP cosine, usually 0.1–0.4. Good for comparing texts, not a probability.</dd>
              <dt>Heatmap</dt>
              <dd>Image regions the model attended to while reading the text.</dd>
            </dl>

            <h3>Limitations</h3>
            <ul className="limitations">
              {info.limitations.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
            <p className="modal-foot">
              <span>Model {info.model_version}</span>
              <span>Core {info.mismatch_core_version}</span>
              <span>Images analysed in memory, never stored</span>
            </p>
          </>
        )}
      </div>
    </dialog>
  );
}
