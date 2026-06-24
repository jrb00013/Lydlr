import React, { useState, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, ComposedChart, Line,
} from 'recharts';
import SignalCellularAltIcon from '@mui/icons-material/SignalCellularAlt';
import './LinkBudgetPanel.css';
import { useSmartPolling } from '../hooks/useSmartPolling';
import lydlrApi from '../api/lydlrApi';

function LinkBudgetPanel() {
  const [healthData, setHealthData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchHealth = useCallback(async () => {
    try {
      const data = await lydlrApi.fleetLinkHealth();
      setHealthData(data);
    } catch (e) {
      console.warn('Link budget health fetch failed:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useSmartPolling(fetchHealth, {
    interval: 10000,
    enabled: true,
    immediate: true,
    minInterval: 8000,
    maxInterval: 20000,
  });

  if (loading && !healthData) {
    return (
      <div className="card link-budget-panel">
        <h2><SignalCellularAltIcon /> Link budget health</h2>
        <p className="chart-empty">Loading budget telemetry…</p>
      </div>
    );
  }

  if (!healthData || !healthData.nodes || healthData.nodes.length === 0) {
    return (
      <div className="card link-budget-panel">
        <h2><SignalCellularAltIcon /> Link budget health</h2>
        <p className="chart-empty">No nodes reporting — start ROS2 edge compressors</p>
      </div>
    );
  }

  const { nodes, summary } = healthData;

  const chartData = nodes.map((n) => ({
    name: n.node_id,
    budget: n.uplink_budget_kbps,
    throughput: n.estimated_throughput_kbps,
    utilization: +(n.budget_utilization * 100).toFixed(1),
    compressionLevel: n.compression_ratio || 0,
    qualityScore: n.quality_score || 0,
    status: n.status,
  }));

  const statusColors = {
    over_budget: '#ef4444',
    at_budget: '#f59e0b',
    under_budget: '#22c55e',
  };

  return (
    <div className="card link-budget-panel">
      <h2><SignalCellularAltIcon /> Link budget health</h2>

      <div className="budget-summary-row">
        <span className={`budget-count budget-count--over`}>
          {summary.over_budget} over budget
        </span>
        <span className={`budget-count budget-count--at`}>
          {summary.at_budget} at budget
        </span>
        <span className={`budget-count budget-count--under`}>
          {summary.under_budget} under budget
        </span>
        {summary.nodes_with_quality_issues > 0 && (
          <span className="budget-count budget-count--quality">
            {summary.nodes_with_quality_issues} quality issues
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={250}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
          <XAxis dataKey="name" tick={{ fontSize: 11 }} />
          <YAxis yAxisId="left" label={{ value: 'kbps', angle: -90, position: 'insideLeft' }} />
          <YAxis yAxisId="right" orientation="right" label={{ value: '×', angle: 0, position: 'insideRight' }} />
          <Tooltip />
          <Legend />
          <Bar yAxisId="left" dataKey="budget" fill="#3b82f6" name="Budget (kbps)" opacity={0.6} />
          <Bar yAxisId="left" dataKey="throughput" fill="#60a5fa" name="Throughput (kbps)" />
          <Line yAxisId="right" type="monotone" dataKey="compressionLevel" stroke="#34d399" name="Compression ratio" strokeWidth={2} dot={{ r: 4 }} />
          <ReferenceLine yAxisId="left" x={0} stroke="rgba(148,163,184,0.3)" />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="budget-node-table">
        {nodes.map((n) => {
          const pct = +(n.budget_utilization * 100).toFixed(0);
          const barColor = pct > 95 ? '#ef4444' : pct > 50 ? '#f59e0b' : '#22c55e';
          return (
            <div key={n.node_id} className="budget-node-row">
              <div className="budget-node-info">
                <span className="budget-node-name">{n.node_id}</span>
                <span className={`budget-node-vertical badge badge--${n.vertical === 'iot' ? 'iot' : 'drone'}`}>
                  {n.vertical}
                </span>
              </div>
              <div className="budget-node-metrics">
                <span>{n.estimated_throughput_kbps} / {n.uplink_budget_kbps} kbps</span>
                <span className="budget-quality-dot" style={{ color: n.quality_ok ? '#22c55e' : '#ef4444' }}>
                  ● {n.quality_ok ? 'OK' : 'Low'}
                </span>
              </div>
              <div className="budget-bar-track">
                <div
                  className="budget-bar-fill"
                  style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: barColor }}
                />
                <span className="budget-bar-label">{pct}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default LinkBudgetPanel;
