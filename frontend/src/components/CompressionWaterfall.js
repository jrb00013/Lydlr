import React, { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import StorageIcon from '@mui/icons-material/Storage';
import { formatBytes } from '../utils/formatBytes';
import './CompressionWaterfall.css';

function CompressionWaterfall({ metric }) {
  const data = useMemo(() => {
    if (!metric) return null;

    const bytesIn = metric.bytes_in || 0;
    const bytesOut = metric.bytes_out || 0;
    const saved = Math.max(0, bytesIn - bytesOut);

    return [
      { name: 'Raw In', bytes: bytesIn, fill: 'var(--drone)' },
      { name: 'Compressed Out', bytes: bytesOut, fill: 'var(--accent-bright)' },
      { name: 'Bandwidth Saved', bytes: saved, fill: 'var(--emerald)' },
    ];
  }, [metric]);

  if (!data || data.every((d) => d.bytes === 0)) {
    return (
      <div className="card compression-waterfall">
        <h2>
          <StorageIcon /> Compression waterfall
        </h2>
        <p className="chart-empty">Awaiting compression telemetry…</p>
      </div>
    );
  }

  const totalIn = data[0].bytes;
  const ratio = totalIn > 0 ? data[1].bytes / totalIn : 0;
  const savingsPct = totalIn > 0 ? ((totalIn - data[1].bytes) / totalIn) * 100 : 0;

  return (
    <div className="card compression-waterfall">
      <div className="compression-waterfall__header">
        <h2>
          <StorageIcon /> Compression waterfall
        </h2>
        <div className="compression-waterfall__stats">
          <span className="badge badge--registered">
            {ratio.toFixed(2)}× ratio
          </span>
          <span className="badge badge--running">
            {savingsPct.toFixed(1)}% saved
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis dataKey="name" type="category" width={120} tick={{ fontSize: 11 }} />
          <Tooltip
            formatter={(value) => [formatBytes(value, 2), 'Bytes']}
            contentStyle={{
              background: 'rgba(10, 14, 22, 0.95)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-primary)',
            }}
          />
          <Legend />
          <Bar dataKey="bytes" barSize={32} radius={[6, 6, 6, 6]} />
        </BarChart>
      </ResponsiveContainer>

      {metric?.modality_bytes_in && metric?.modality_bytes_out && (
        <div className="compression-waterfall__breakdown">
          <h3>Modality breakdown</h3>
          <div className="compression-waterfall__rows">
            {Object.entries(metric.modality_bytes_in).map(([mod, inBytes]) => {
              const outBytes = (metric.modality_bytes_out[mod] || 0);
              const modSaved = Math.max(0, inBytes - outBytes);
              const modRatio = inBytes > 0 ? outBytes / inBytes : 0;
              return (
                <div key={mod} className="compression-waterfall__row">
                  <span className="compression-waterfall__mod">{mod}</span>
                  <span className="compression-waterfall__vals">
                    {formatBytes(inBytes)} → {formatBytes(outBytes)}
                    <span className="compression-waterfall__pct">
                      ({modSaved > 0 ? `-${Math.round((1 - modRatio) * 100)}%` : '0%'})
                    </span>
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default CompressionWaterfall;
