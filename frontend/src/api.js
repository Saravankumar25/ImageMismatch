const API_BASE = (import.meta.env.VITE_API_BASE || '/api').replace(/\/$/, '');
const TIMEOUT_MS = 30000;

// Spec section 8.5 -- what the user sees for each backend error code.
const MESSAGES = {
  NO_TEXT_INPUT:
    'No text was provided and no text was found in the image. Add a caption or upload an image that contains text.',
  UNSUPPORTED_FILE_TYPE: 'Please upload a JPEG, PNG, or WebP image.',
  FILE_TOO_LARGE: 'Image is too large. Maximum is 10 MB.',
  TOO_MANY_CAPTIONS: 'You can add up to 5 captions.',
  CAPTION_TOO_LONG: 'Each caption can be at most 300 characters.',
  MISSING_IMAGE: 'Please upload an image.',
  MODEL_NOT_READY: 'The model is still loading. Please try again in a moment.',
  NETWORK: 'Could not reach the server. Please try again.',
  UNKNOWN: 'Something went wrong. Please try again.',
};

export function messageFor(code) {
  return MESSAGES[code] || MESSAGES.UNKNOWN;
}

export class ApiError extends Error {
  constructor(code, message) {
    super(message || messageFor(code));
    this.code = code;
  }
}

async function request(path, init = {}, { signal, timeoutMs = TIMEOUT_MS } = {}) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onCancel = () => controller.abort();
  signal?.addEventListener('abort', onCancel, { once: true });

  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, signal: controller.signal });
  } catch {
    if (signal?.aborted && !timedOut) throw new ApiError('CANCELLED', 'Request cancelled.');
    throw new ApiError('NETWORK');
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onCancel);
  }

  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON body, e.g. an Nginx error page */
  }
  if (res.ok && body) return body;

  const code =
    body?.error?.code ||
    (res.status === 413 ? 'FILE_TOO_LARGE' : res.status === 415 ? 'UNSUPPORTED_FILE_TYPE' : null) ||
    ([502, 503, 504].includes(res.status) ? 'NETWORK' : 'UNKNOWN');
  throw new ApiError(code);
}

/** POST /api/predict with the image and 0-5 captions. */
export function predict(file, captions, options) {
  const form = new FormData();
  form.append('image', file, file.name || 'image');
  for (const c of captions) form.append('captions', c);
  return request('/predict', { method: 'POST', body: form }, options);
}

/** GET /api/health -> 'online' | 'loading' | 'offline' (never throws). */
export async function getHealth() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    const body = await res.json().catch(() => null);
    if (res.ok && body?.model_loaded) return { state: 'online', version: body.model_version };
    if (res.status === 503 && body?.status === 'loading') return { state: 'loading', version: body.model_version };
    return { state: 'offline' };
  } catch {
    return { state: 'offline' };
  } finally {
    clearTimeout(timer);
  }
}

export function getModelInfo(options) {
  return request('/model-info', {}, { timeoutMs: 10000, ...options });
}
