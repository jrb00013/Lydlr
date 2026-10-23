/** Format byte counts for console telemetry. */

export function formatBytes(bytes, digits = 1) {
  const n = Number(bytes) || 0;
  if (n < 1024) return `${n.toFixed(0)} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(digits)} KB`;
  return `${(n / (1024 * 1024)).toFixed(digits)} MB`;
}

export function formatRatio(bytesIn, bytesOut) {
  const inn = Number(bytesIn) || 0;
  const out = Number(bytesOut) || 0;
  if (out <= 0) return '—';
  return `${(inn / out).toFixed(2)}×`;
}

export function formatSavingsPct(bytesIn, bytesOut) {
  const inn = Number(bytesIn) || 0;
  const out = Number(bytesOut) || 0;
  if (inn <= 0) return '0%';
  return `${(((inn - out) / inn) * 100).toFixed(1)}%`;
}
