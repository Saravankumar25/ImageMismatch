import { useEffect, useId, useRef, useState } from 'react';

const MAX_MB = 10;
const MAX_BYTES = MAX_MB * 1024 * 1024;
const MAX_PIXELS = 40_000_000;
const TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const EXT_TYPES = { jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp' };

function fileType(file) {
  if (file.type) return file.type;
  const ext = file.name?.split('.').pop()?.toLowerCase();
  return EXT_TYPES[ext] || '';
}

function readDimensions(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = reject;
    img.src = url;
  });
}

/** Client-side checks (spec section 3). Resolves to {file,url,width,height} or {error, message?}. */
export async function validateImageFile(file) {
  if (!TYPES.includes(fileType(file))) return { error: 'UNSUPPORTED_FILE_TYPE' };
  if (file.size > MAX_BYTES) return { error: 'FILE_TOO_LARGE' };
  const url = URL.createObjectURL(file);
  try {
    const { width, height } = await readDimensions(url);
    if (width * height > MAX_PIXELS) {
      URL.revokeObjectURL(url);
      return { error: 'FILE_TOO_LARGE', message: 'Image is too large. Maximum is 40 megapixels.' };
    }
    return { file, url, width, height };
  } catch {
    URL.revokeObjectURL(url);
    return { error: 'UNSUPPORTED_FILE_TYPE' };
  }
}

function formatBytes(n) {
  return n < 1024 * 1024 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export default function ImageDropzone({ image, onSelect, onReject, disabled, zoneRef }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  const hintId = useId();

  async function accept(file) {
    if (!file || disabled) return;
    const result = await validateImageFile(file);
    if (result.error) onReject(result.error, result.message);
    else onSelect(result);
  }

  function open() {
    if (!disabled) inputRef.current?.click();
  }

  // Paste an image from the clipboard anywhere on the page.
  useEffect(() => {
    function onPaste(e) {
      if (disabled) return;
      const item = [...(e.clipboardData?.items || [])].find((i) => i.kind === 'file' && i.type.startsWith('image/'));
      if (item) {
        e.preventDefault();
        accept(item.getAsFile());
      }
    }
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  });

  const handlers = disabled
    ? {}
    : {
        onClick: open,
        onKeyDown: (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            open();
          }
        },
        onDragEnter: (e) => {
          e.preventDefault();
          dragDepth.current += 1;
          setDragging(true);
        },
        onDragOver: (e) => {
          e.preventDefault();
          e.dataTransfer.dropEffect = 'copy';
        },
        onDragLeave: () => {
          dragDepth.current = Math.max(0, dragDepth.current - 1);
          if (dragDepth.current === 0) setDragging(false);
        },
        onDrop: (e) => {
          e.preventDefault();
          dragDepth.current = 0;
          setDragging(false);
          accept(e.dataTransfer.files?.[0]);
        },
      };

  const className = ['dropzone', image && 'has-image', dragging && 'dragging', disabled && 'disabled']
    .filter(Boolean)
    .join(' ');

  return (
    <div
      ref={zoneRef}
      className={className}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled || undefined}
      aria-label={image ? `Selected image ${image.file.name}. Press Enter to choose a different image.` : 'Choose an image'}
      aria-describedby={hintId}
      {...handlers}
    >
      <input
        ref={inputRef}
        type="file"
        accept={TYPES.join(',')}
        hidden
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()} // don't bubble back into the zone's click handler
        onChange={(e) => {
          accept(e.target.files?.[0]);
          e.target.value = ''; // allow re-selecting the same file
        }}
      />
      {image ? (
        <div className="dz-preview">
          <img src={image.url} alt={`Preview of the selected image, ${image.file.name}`} />
          <div className="dz-meta" id={hintId}>
            <span className="dz-file">
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <rect x="2" y="2.5" width="12" height="11" rx="2" fill="none" stroke="currentColor" strokeWidth="1.3" />
                <path d="M2.5 11l3.2-3.2 2.6 2.6 1.8-1.8 3.4 3.4" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
              </svg>
              <span className="dz-name" title={image.file.name}>{image.file.name}</span>
            </span>
            <span className="dz-facts">
              {image.width}×{image.height} · {formatBytes(image.file.size)}
            </span>
            {!disabled && <span className="dz-replace">Replace</span>}
          </div>
          {dragging && <div className="dz-drop-veil">Drop to replace</div>}
        </div>
      ) : (
        <div className="dz-empty">
          <span className="dz-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path
                d="M12 15.5V4.5m0 0L7.5 9M12 4.5 16.5 9M4.5 15v2.5a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V15"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          <p className="dz-title">{dragging ? 'Release to upload' : 'Drop an image here'}</p>
          <p className="dz-sub" id={hintId}>
            or <span className="dz-browse">browse files</span> · paste with <kbd>Ctrl</kbd>
            <kbd>V</kbd>
          </p>
          <div className="dz-formats" aria-hidden="true">
            <span>JPEG</span>
            <span>PNG</span>
            <span>WebP</span>
            <span>≤ {MAX_MB} MB</span>
          </div>
        </div>
      )}
    </div>
  );
}
