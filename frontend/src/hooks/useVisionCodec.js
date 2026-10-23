import { useEffect, useState } from 'react';
import { apiBaseUrl } from '../api/lydlrApi';

/** Poll the live temporal-DCT encoder. Each tick is one countable frame. */
export function useVisionCodec(enabled = true, intervalMs = 1200) {
  const [metric, setMetric] = useState(null);
  const [previews, setPreviews] = useState(null);

  useEffect(() => {
    if (!enabled) return undefined;
    let cancelled = false;

    const pull = async () => {
      try {
        const res = await fetch(`${apiBaseUrl()}/api/codec/frame/`);
        if (!res.ok) return;
        const body = await res.json();
        if (cancelled) return;
        setMetric(body.metric || null);
        setPreviews(body.previews || null);
      } catch (_) {
        /* API offline */
      }
    };

    pull();
    const id = setInterval(pull, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [enabled, intervalMs]);

  return { metric, previews };
}

export default useVisionCodec;
