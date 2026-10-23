import { useEffect, useRef, useState, useCallback } from 'react';

/**
 * Smart polling hook with I/O scheduling
 * - Respects page visibility (pauses when tab is hidden)
 * - Exponential backoff on errors
 * - Configurable intervals
 * - Automatic cleanup
 */
export function useSmartPolling(callback, options = {}) {
  const {
    interval = 10000,
    enabled = true,
    immediate = true,
    onError = null,
    maxInterval = 60000,
    minInterval = 5000,
  } = options;

  const [isPolling, setIsPolling] = useState(enabled);
  const intervalRef = useRef(null);
  const timeoutRef = useRef(null);
  const currentIntervalRef = useRef(interval);
  const errorCountRef = useRef(0);
  const callbackRef = useRef(callback);
  const isVisibleRef = useRef(true);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  const scheduleNextPoll = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    if (!isVisibleRef.current) {
      return;
    }

    const backoffMultiplier = Math.min(Math.pow(1.5, errorCountRef.current), maxInterval / interval);
    const nextInterval = Math.min(
      Math.max(interval * backoffMultiplier, minInterval),
      maxInterval
    );
    currentIntervalRef.current = nextInterval;

    timeoutRef.current = setTimeout(async () => {
      if (!isVisibleRef.current || !isPolling) {
        return;
      }
      try {
        await callbackRef.current();
        errorCountRef.current = 0;
        currentIntervalRef.current = interval;
      } catch (error) {
        errorCountRef.current += 1;
        if (onError) {
          onError(error, errorCountRef.current);
        }
      }
      scheduleNextPoll();
    }, nextInterval);
  }, [interval, minInterval, maxInterval, isPolling, onError]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      isVisibleRef.current = !document.hidden;
      if (!document.hidden && isPolling) {
        scheduleNextPoll();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isPolling, scheduleNextPoll]);

  useEffect(() => {
    if (!enabled || !isPolling) {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      return undefined;
    }

    if (immediate) {
      callbackRef.current().catch((error) => {
        errorCountRef.current += 1;
        if (onError) {
          onError(error, errorCountRef.current);
        }
      });
    }

    scheduleNextPoll();
    const timeoutId = timeoutRef.current;
    const intervalId = intervalRef.current;
    return () => {
      if (timeoutId) clearTimeout(timeoutId);
      if (intervalId) clearInterval(intervalId);
    };
  }, [enabled, isPolling, interval, immediate, scheduleNextPoll, onError]);

  const pause = () => setIsPolling(false);
  const resume = () => {
    setIsPolling(true);
    if (isVisibleRef.current) {
      scheduleNextPoll();
    }
  };
  const reset = () => {
    errorCountRef.current = 0;
    currentIntervalRef.current = interval;
  };

  return {
    isPolling,
    pause,
    resume,
    reset,
    currentInterval: currentIntervalRef.current,
    errorCount: errorCountRef.current,
  };
}
