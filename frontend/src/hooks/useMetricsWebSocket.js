import { useEffect, useRef, useState, useCallback } from 'react';

const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

export function metricsWsUrl() {
  return `${WS_URL.replace(/\/$/, '')}/ws/metrics/`;
}

export function fleetWsUrl() {
  return `${WS_URL.replace(/\/$/, '')}/ws/fleet/`;
}

/**
 * Shared metrics WebSocket with reconnect.
 * onMetric receives the unwrapped metric document (node_id, compression_ratio, …).
 */
export function useMetricsWebSocket(onMetric, { enabled = true } = {}) {
  const [connected, setConnected] = useState(false);
  const onMetricRef = useRef(onMetric);
  const wsRef = useRef(null);
  const retryRef = useRef(null);

  useEffect(() => {
    onMetricRef.current = onMetric;
  }, [onMetric]);

  const disconnect = useCallback(() => {
    if (retryRef.current) {
      clearTimeout(retryRef.current);
      retryRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  useEffect(() => {
    if (!enabled) {
      disconnect();
      return undefined;
    }

    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      try {
        const ws = new WebSocket(metricsWsUrl());
        wsRef.current = ws;

        ws.onopen = () => {
          if (!cancelled) setConnected(true);
        };

        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'metrics_update' && msg.data) {
              onMetricRef.current?.(msg.data);
            }
          } catch (_) {
            /* ignore malformed */
          }
        };

        ws.onerror = () => {
          /* onclose handles reconnect */
        };

        ws.onclose = () => {
          setConnected(false);
          wsRef.current = null;
          if (!cancelled) {
            retryRef.current = setTimeout(connect, 3000);
          }
        };
      } catch (_) {
        if (!cancelled) {
          retryRef.current = setTimeout(connect, 3000);
        }
      }
    };

    connect();
    return () => {
      cancelled = true;
      disconnect();
    };
  }, [enabled, disconnect]);

  return { connected, disconnect };
}

export default useMetricsWebSocket;
