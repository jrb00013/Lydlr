import { useEffect, useRef, useState } from 'react';

/**
 * Smart polling hook with I/O scheduling
 * - Respects page visibility (pauses when tab is hidden)
 * - Exponential backoff on errors
 * - Configurable intervals
 * - Automatic cleanup
 */
export function useSmartPolling(callback, options = {}) {
  const {
    interval = 10000, // Default 10 seconds
    enabled = true,
    immediate = true,
    onError = null,
    maxInterval = 60000, // Max 60 seconds
    minInterval = 5000, // Min 5 seconds
  } = options;

  const [isPolling, setIsPolling] = useState(enabled);
  const intervalRef = useRef(null);
  const timeoutRef = useRef(null);
  const currentIntervalRef = useRef(interval);
  const errorCountRef = useRef(0);
  const callbackRef = useRef(callback);
  const isVisibleRef = useRef(true);

  // Update callback ref when it changes
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  // Handle page visibility
  useEffect(() => {
    const handleVisibilityChange = () => {
      isVisibleRef.current = !document.hidden;
      if (!document.hidden && isPolling) {
        // Resume polling when tab becomes visible
        scheduleNextPoll();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [isPolling]);

  const scheduleNextPoll = () => {
    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Don't schedule if tab is hidden
    if (!isVisibleRef.current) {
      return;
    }

    // Calculate backoff interval (exponential backoff on errors)
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
        // Reset error count on success
        errorCountRef.current = 0;
        currentIntervalRef.current = interval;
      } catch (error) {
        errorCountRef.current += 1;
        if (onError) {
          onError(error, errorCountRef.current);
        }
      }

      // Schedule next poll
      scheduleNextPoll();
    }, nextInterval);
  };

  useEffect(() => {
    if (!enabled || !isPolling) {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      return;
    }

    // Immediate execution if requested
    if (immediate) {
      callbackRef.current().catch((error) => {
        errorCountRef.current += 1;
        if (onError) {
          onError(error, errorCountRef.current);
        }
      });
    }

    // Schedule first poll
    scheduleNextPoll();

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [enabled, isPolling, interval, immediate]);

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

