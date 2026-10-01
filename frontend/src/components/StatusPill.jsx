import { useEffect, useState } from 'react';
import { getHealth } from '../api.js';

const LABELS = {
  checking: 'Connecting…',
  loading: 'Model warming up',
  online: 'Model online',
  offline: 'Server offline',
};

/** Live backend status: polls /api/health quickly until online, then slowly. */
export default function StatusPill() {
  const [status, setStatus] = useState({ state: 'checking' });

  useEffect(() => {
    let alive = true;
    let timer;
    async function poll() {
      const next = await getHealth();
      if (!alive) return;
      setStatus(next);
      timer = setTimeout(poll, next.state === 'online' ? 30000 : 4000);
    }
    poll();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);

  return (
    <span className={`status-pill status-${status.state}`} role="status" aria-live="polite">
      <span className="status-dot" aria-hidden="true" />
      {LABELS[status.state]}
      {status.state === 'online' && status.version && <span className="status-version">{status.version}</span>}
    </span>
  );
}
