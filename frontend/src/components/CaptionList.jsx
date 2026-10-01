import { useEffect, useId, useRef } from 'react';

export const MAX_CAPTIONS = 5;
export const MAX_CHARS = 300;

let nextId = 1;
export const newCaption = (text = '') => ({ id: nextId++, text });

export default function CaptionList({ captions, onChange, disabled }) {
  const inputs = useRef(new Map());
  const focusId = useRef(null);
  const hintId = useId();

  useEffect(() => {
    if (focusId.current != null) {
      inputs.current.get(focusId.current)?.focus();
      focusId.current = null;
    }
  });

  function update(id, text) {
    onChange(captions.map((c) => (c.id === id ? { ...c, text } : c)));
  }

  function add() {
    if (captions.length >= MAX_CAPTIONS) return;
    const c = newCaption();
    focusId.current = c.id;
    onChange([...captions, c]);
  }

  function remove(index) {
    if (captions.length === 1) {
      focusId.current = captions[0].id;
      onChange([{ ...captions[0], text: '' }]);
      return;
    }
    const next = captions.filter((_, i) => i !== index);
    focusId.current = next[Math.min(index, next.length - 1)].id;
    onChange(next);
  }

  const full = captions.length >= MAX_CAPTIONS;

  return (
    <fieldset className="captions" disabled={disabled}>
      <legend className="sr-only">Captions (optional if the image has text)</legend>
      <p className="field-hint" id={hintId}>
        Describe the image in your own words — compare up to 5 at once.
      </p>
      <ol className="caption-list">
        {captions.map((c, i) => {
          const len = c.text.trim().length;
          const counterId = `caption-counter-${c.id}`;
          return (
            <li key={c.id} className="caption-row">
              <span className="caption-index" aria-hidden="true">
                {i + 1}
              </span>
              <input
                ref={(el) => (el ? inputs.current.set(c.id, el) : inputs.current.delete(c.id))}
                type="text"
                className="caption-input"
                value={c.text}
                maxLength={MAX_CHARS}
                placeholder={i === 0 ? 'A dog running on the beach' : 'Another description to compare'}
                aria-label={`Caption ${i + 1}`}
                aria-describedby={`${i === 0 ? hintId + ' ' : ''}${counterId}`}
                onChange={(e) => update(c.id, e.target.value)}
              />
              <span
                id={counterId}
                className={`char-counter${len > MAX_CHARS - 30 ? ' near-limit' : ''}${len ? ' has-text' : ''}`}
                aria-label={`${len} of ${MAX_CHARS} characters`}
              >
                {len}/{MAX_CHARS}
              </span>
              <button
                type="button"
                className="caption-remove"
                aria-label={captions.length === 1 ? 'Clear caption' : `Remove caption ${i + 1}`}
                title={captions.length === 1 ? 'Clear' : 'Remove'}
                onClick={() => remove(i)}
                disabled={captions.length === 1 && !c.text}
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M7 7l10 10M17 7L7 17" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                </svg>
              </button>
            </li>
          );
        })}
      </ol>
      <button type="button" className="add-caption" onClick={add} disabled={full}>
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path d="M8 3.5v9M3.5 8h9" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        {full ? 'Maximum of 5 captions' : 'Add another caption'}
        <span className="add-count">
          {captions.length}/{MAX_CAPTIONS}
        </span>
      </button>
    </fieldset>
  );
}
