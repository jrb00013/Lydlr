import { useEffect, useRef, useState, useCallback } from 'react';
import { fleetWsUrl } from '../api/lydlrApi';

/**
 * Subscribe to fleet deployment / node command / link-spec events.
 */
export function useFleetEvents(onEvent, { enabled = true } = {}) {
  const [connected, setConnected] = useState(false);
  const onEventRef = useRef(onEvent);

  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  useEffect(() => {
    if (!enabled) return undefined;
    let cancelled = false;
    let ws;
    let retry;

    const connect = () => {
      if (cancelled) return;
      try {
        ws = new WebSocket(fleetWsUrl());
        ws.onopen = () => {
          if (!cancelled) setConnected(true);
        };
        ws.onmessage = (event) => {
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'fleet_event') {
              onEventRef.current?.(msg);
            }
          } catch (_) {
            /* ignore */
          }
        };
        ws.onclose = () => {
          setConnected(false);
          if (!cancelled) retry = setTimeout(connect, 4000);
        };
      } catch (_) {
        if (!cancelled) retry = setTimeout(connect, 4000);
      }
    };

    connect();
    return () => {
      cancelled = true;
      if (retry) clearTimeout(retry);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
      setConnected(false);
    };
  }, [enabled]);

  return { connected };
}

export default useFleetEvents;
