import { useEffect, useRef, useCallback } from 'react';
import { apiBaseUrl } from '../api/lydlrApi';

/**
 * Periodically hit /api/demo/pulse/ so Visual / Dashboard stay alive
 * when ROS edge nodes are offline. Stops after first real metric if requested.
 */
export function useDemoPulse({
  enabled = true,
  intervalMs = 1500,
  onlyWhenIdle = true,
  lastMetricAtRef,
} = {}) {
  const timerRef = useRef(null);

  const pulse = useCallback(async () => {
    if (onlyWhenIdle && lastMetricAtRef?.current) {
      if (Date.now() - lastMetricAtRef.current < 4000) return;
    }
    try {
      await fetch(`${apiBaseUrl()}/api/demo/pulse/`, { method: 'POST' });
    } catch (_) {
      /* control plane down */
    }
  }, [onlyWhenIdle, lastMetricAtRef]);

  useEffect(() => {
    if (!enabled) return undefined;
    pulse();
    timerRef.current = setInterval(pulse, intervalMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [enabled, intervalMs, pulse]);
}

export default useDemoPulse;
