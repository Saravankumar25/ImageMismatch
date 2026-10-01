import { useCallback, useEffect, useRef } from 'react';

// Monochrome "spotlight": low-attention regions are darkened, high-attention regions stay clear.
const MAX_SHADE = 0.82;

/** Bilinear upscale of the g×g grid into a black mask (patch centres at (i+0.5)/g). */
function renderHeat(grid, size) {
  const g = grid.length;
  const flat = grid.flat();
  const lo = Math.min(...flat);
  const range = Math.max(...flat) - lo || 1;
  const norm = grid.map((row) => row.map((v) => (v - lo) / range));
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  const clamp = (v) => Math.min(g - 1, Math.max(0, v));
  for (let y = 0; y < size; y++) {
    const gy = clamp(((y + 0.5) / size) * g - 0.5);
    const y0 = Math.floor(gy);
    const y1 = Math.min(g - 1, y0 + 1);
    const ty = gy - y0;
    for (let x = 0; x < size; x++) {
      const gx = clamp(((x + 0.5) / size) * g - 0.5);
      const x0 = Math.floor(gx);
      const x1 = Math.min(g - 1, x0 + 1);
      const tx = gx - x0;
      const top = norm[y0][x0] * (1 - tx) + norm[y0][x1] * tx;
      const bottom = norm[y1][x0] * (1 - tx) + norm[y1][x1] * tx;
      const v = top * (1 - ty) + bottom * ty;
      const i = (y * size + x) * 4;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = 0;
      img.data[i + 3] = Math.round(255 * 0.7 * Math.pow(1 - v, 1.6)); // peaks clear, lows dimmed
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/**
 * Image viewer that draws a 7×7 attention grid over the preview.
 * The model sees a centre square crop (short side resized to 224, then centre-cropped),
 * so the heatmap covers that square and the cropped-away edges are shaded.
 */
export default function HeatmapOverlay({ src, heatmap, alt, scanning, label }) {
  const imgRef = useRef(null);
  const canvasRef = useRef(null);

  const draw = useCallback(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !img.complete || !img.naturalWidth) return;
    const w = img.clientWidth;
    const h = img.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!heatmap) return;

    const side = Math.min(w, h);
    const x0 = (w - side) / 2;
    const y0 = (h - side) / 2;

    // Shade the parts of the image the model never saw.
    ctx.fillStyle = `rgba(0, 0, 0, ${MAX_SHADE})`;
    if (w > h) {
      ctx.fillRect(0, 0, x0, h);
      ctx.fillRect(x0 + side, 0, w - x0 - side, h);
    } else if (h > w) {
      ctx.fillRect(0, 0, w, y0);
      ctx.fillRect(0, y0 + side, w, h - y0 - side);
    }

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(renderHeat(heatmap, 168), x0, y0, side, side);

    if (w !== h) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.75)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(x0 + 0.5, y0 + 0.5, side - 1, side - 1);
    }
  }, [heatmap]);

  useEffect(() => {
    draw();
    const img = imgRef.current;
    if (!img || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(() => draw());
    ro.observe(img);
    return () => ro.disconnect();
  }, [draw]);

  return (
    <figure className="viewer">
      <div className={`viewer-stage${scanning ? ' is-scanning' : ''}${heatmap ? ' has-heat' : ''}`}>
        <img ref={imgRef} src={src} alt={alt} onLoad={draw} />
        <canvas ref={canvasRef} aria-hidden="true" />
        {scanning && (
          <div className="scan" aria-hidden="true">
            <div className="scan-grid" />
            <div className="scan-bar" />
          </div>
        )}
      </div>
      <figcaption className="viewer-caption">
        {scanning ? (
          <span className="viewer-status">
            <span className="pulse-dot" aria-hidden="true" /> Reading text, encoding with CLIP, scoring…
          </span>
        ) : heatmap ? (
          <>
            <span className="viewer-label">
              Attention for <q>{label}</q>
            </span>
            <span className="viewer-note">
              Brighter areas received more attention. Outside the dashed square was cropped away before analysis.
            </span>
          </>
        ) : (
          <span className="viewer-hint">Select “Show heatmap” on a result to see where the model looked.</span>
        )}
      </figcaption>
    </figure>
  );
}
