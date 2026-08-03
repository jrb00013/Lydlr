import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import SignalOcean from './SignalOcean';
import { useMetricsWebSocket } from '../hooks/useMetricsWebSocket';
import { useSmartPolling } from '../hooks/useSmartPolling';
import { useDemoPulse } from '../hooks/useDemoPulse';
import { lydlrApi, previewMjpegUrl, previewJpegUrl } from '../api/lydlrApi';
import './VisualMonitoring.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const maxHistoryLength = 100;

function VisualMonitoring() {
  const [nodes, setNodes] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [topics, setTopics] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [compressionHistory, setCompressionHistory] = useState([]);
  const [modalitySeries, setModalitySeries] = useState([]);
  const [linkHealth, setLinkHealth] = useState(null);
  const [previewKey, setPreviewKey] = useState(0);
  const [useMjpeg, setUseMjpeg] = useState(true);
  const lastMetricAtRef = useRef(0);

  const selectedMetric = selectedNode ? metrics[selectedNode] : Object.values(metrics)[0];
  const fleetMetrics = useMemo(() => Object.values(metrics), [metrics]);

  const pushHistory = useCallback((metric) => {
    if (!metric?.node_id || metric.compression_ratio == null) return;
    lastMetricAtRef.current = Date.now();
    setCompressionHistory((prev) => {
      const updated = [
        ...prev,
        {
          timestamp: new Date().toISOString(),
          node: metric.node_id,
          compression: metric.compression_ratio,
          latency: metric.latency_ms,
          quality: (metric.quality_score || 0) * 100,
          throughput: metric.bandwidth_estimate,
        },
      ];
      return updated.slice(-maxHistoryLength);
    });

    const mIn = metric.modality_bytes_in || {};
    const mOut = metric.modality_bytes_out || {};
    setModalitySeries((prev) => {
      const point = {
        time: new Date().toLocaleTimeString(),
        node: metric.node_id,
        cam_in: mIn.camera || 0,
        lidar_in: mIn.lidar || 0,
        imu_in: mIn.imu || 0,
        audio_in: mIn.audio || 0,
        cam_out: mOut.camera || 0,
        lidar_out: mOut.lidar || 0,
        imu_out: mOut.imu || 0,
        audio_out: mOut.audio || 0,
      };
      return [...prev, point].slice(-40);
    });
  }, []);

  const onLiveMetric = useCallback(
    (metric) => {
      if (!metric?.node_id) return;
      setMetrics((prev) => ({ ...prev, [metric.node_id]: metric }));
      pushHistory(metric);
    },
    [pushHistory]
  );

  useMetricsWebSocket(onLiveMetric, { enabled: true });
  useDemoPulse({
    enabled: true,
    intervalMs: 1500,
    onlyWhenIdle: true,
    lastMetricAtRef,
  });

  useEffect(() => {
    const id = setInterval(() => setPreviewKey((k) => k + 1), 800);
    return () => clearInterval(id);
  }, []);

  const fetchNodes = useCallback(async () => {
    try {
      const data = await lydlrApi.nodes();
      setNodes(data);
      setSelectedNode((prev) => prev || (data[0] && data[0].node_id) || null);
    } catch (e) {
      console.error('Failed to fetch nodes:', e);
    }
  }, []);

  const fetchLinkHealth = useCallback(async () => {
    try {
      setLinkHealth(await lydlrApi.fleetLinkHealth());
    } catch (_) {
      /* optional */
    }
  }, []);

  const fetchTopics = useCallback(async (nodeId) => {
    if (!nodeId) return;
    try {
      const data = await lydlrApi.nodeTopics(nodeId);
      setTopics(data.topics || []);
    } catch (_) {
      setTopics([
        { name: `/lydlr/${nodeId}/transport/compressed`, type: 'compressed', node: nodeId },
        { name: `/lydlr/${nodeId}/transport/metrics`, type: 'metrics', node: nodeId },
        { name: `/lydlr/${nodeId}/preview/raw`, type: 'preview', node: nodeId },
        { name: `/lydlr/${nodeId}/preview/reconstructed`, type: 'preview', node: nodeId },
        { name: `/lydlr/${nodeId}/preview/heatmap`, type: 'preview', node: nodeId },
      ]);
    }
  }, []);

  const fetchRecent = useCallback(async () => {
    if (!selectedNode) return;
    try {
      const res = await fetch(`${API_URL}/api/metrics/?node_id=${selectedNode}&limit=40`);
      const data = await res.json();
      if (Array.isArray(data) && data[0]) {
        setMetrics((prev) => ({ ...prev, [selectedNode]: data[0] }));
        [...data].reverse().forEach(pushHistory);
      }
    } catch (_) {
      /* ignore */
    }
  }, [selectedNode, pushHistory]);

  const refresh = useCallback(async () => {
    await Promise.all([fetchNodes(), fetchLinkHealth()]);
  }, [fetchNodes, fetchLinkHealth]);

  useSmartPolling(refresh, { interval: 10000, immediate: true });

  useEffect(() => {
    fetchTopics(selectedNode);
    fetchRecent();
    setPreviewKey((k) => k + 1);
  }, [selectedNode, fetchTopics, fetchRecent]);

  const nodeHistory = useMemo(
    () => compressionHistory.filter((m) => !selectedNode || m.node === selectedNode),
    [compressionHistory, selectedNode]
  );

  const modalityForNode = useMemo(
    () => modalitySeries.filter((m) => !selectedNode || m.node === selectedNode),
    [modalitySeries, selectedNode]
  );

  const budgetRow = useMemo(() => {
    const list = linkHealth?.nodes || linkHealth?.health || [];
    if (!Array.isArray(list)) return null;
    return list.find((n) => n.node_id === selectedNode) || list[0] || null;
  }, [linkHealth, selectedNode]);

  const getQualityColor = (quality) => {
    if (quality >= 80) return 'var(--emerald, #10b981)';
    if (quality >= 60) return 'var(--amber, #f59e0b)';
    return 'var(--rose, #f43f5e)';
  };

  const sides = [
    { key: 'raw', label: 'Raw' },
    { key: 'reconstructed', label: 'Reconstructed' },
    { key: 'heatmap', label: 'Heatmap' },
  ];

  return (
    <div className="visual-monitoring">
      <section className="visual-hero">
        <SignalOcean
          metric={selectedMetric}
          fleetMetrics={fleetMetrics}
          linkHealth={linkHealth}
          selectedNode={selectedNode}
          onSelectNode={setSelectedNode}
          demoFallback
        />
        <div className="visual-hero__controls">
          <label htmlFor="vm-node">Node</label>
          <select
            id="vm-node"
            value={selectedNode || ''}
            onChange={(e) => setSelectedNode(e.target.value || null)}
            className="node-selector"
          >
            {nodes.map((node) => (
              <option key={node.node_id} value={node.node_id}>
                {node.node_id} ({node.status || 'unknown'})
              </option>
            ))}
          </select>
          <button
            type="button"
            className="preview-mode-btn"
            onClick={() => setUseMjpeg((v) => !v)}
          >
            {useMjpeg ? 'MJPEG' : 'JPEG poll'}
          </button>
        </div>
      </section>

      <div className="visual-below">
        <div className="live-strip">
          <div className="live-strip__item">
            <span className="live-strip__label">Compression</span>
            <span className="live-strip__value">
              {selectedMetric?.compression_ratio != null
                ? `${Number(selectedMetric.compression_ratio).toFixed(2)}×`
                : '—'}
            </span>
          </div>
          <div className="live-strip__item">
            <span className="live-strip__label">Latency</span>
            <span className="live-strip__value">
              {selectedMetric?.latency_ms != null
                ? `${Number(selectedMetric.latency_ms).toFixed(1)} ms`
                : '—'}
            </span>
          </div>
          <div className="live-strip__item">
            <span className="live-strip__label">Quality</span>
            <span
              className="live-strip__value"
              style={{
                color: getQualityColor((selectedMetric?.quality_score || 0) * 100),
              }}
            >
              {selectedMetric?.quality_score != null
                ? `${(selectedMetric.quality_score * 100).toFixed(1)}%`
                : '—'}
            </span>
          </div>
          <div className="live-strip__item live-strip__item--wide">
            <span className="live-strip__label">Uplink vs budget</span>
            <div className="budget-bar">
              <div
                className="budget-bar__fill"
                style={{
                  width: `${Math.min(
                    100,
                    budgetRow
                      ? ((budgetRow.estimated_throughput_kbps || 0) /
                          Math.max(budgetRow.uplink_budget_kbps || 1, 1)) *
                        100
                      : (selectedMetric?.bandwidth_estimate || 0) * 100
                  )}%`,
                }}
              />
            </div>
            <span className="live-strip__meta">
              {budgetRow
                ? `${Number(budgetRow.estimated_throughput_kbps || 0).toFixed(0)} / ${Number(
                    budgetRow.uplink_budget_kbps || 0
                  ).toFixed(0)} kbps`
                : 'awaiting link health'}
            </span>
          </div>
        </div>

        <div className="preview-panes">
          {sides.map(({ key, label }) => (
            <figure key={key} className="preview-pane">
              <figcaption>{label}</figcaption>
              {selectedNode ? (
                <img
                  key={`${selectedNode}-${key}-${useMjpeg ? 'm' : previewKey}`}
                  src={
                    useMjpeg
                      ? previewMjpegUrl(selectedNode, key)
                      : previewJpegUrl(selectedNode, key, previewKey)
                  }
                  alt={`${label} preview for ${selectedNode}`}
                  className="preview-pane__img"
                  onError={() => setUseMjpeg(false)}
                />
              ) : (
                <div className="preview-pane__empty">Select a node</div>
              )}
            </figure>
          ))}
        </div>

        <div className="monitoring-grid">
          <div className="chart-panel">
            <h2>Compression over time</h2>
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={nodeHistory}>
                <defs>
                  <linearGradient id="compFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(v) => new Date(v).toLocaleTimeString()}
                  stroke="#64748b"
                  fontSize={11}
                />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{ background: '#111827', border: '1px solid #334155' }}
                  labelFormatter={(v) => new Date(v).toLocaleString()}
                />
                <Area
                  type="monotone"
                  dataKey="compression"
                  stroke="#22d3ee"
                  fill="url(#compFill)"
                  name="Compression"
                  strokeWidth={2}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-panel">
            <h2>Modality bytes in</h2>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={modalityForNode}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                <XAxis dataKey="time" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip contentStyle={{ background: '#111827', border: '1px solid #334155' }} />
                <Legend />
                <Bar dataKey="cam_in" stackId="in" fill="#22d3ee" name="camera" />
                <Bar dataKey="lidar_in" stackId="in" fill="#10b981" name="lidar" />
                <Bar dataKey="imu_in" stackId="in" fill="#f59e0b" name="imu" />
                <Bar dataKey="audio_in" stackId="in" fill="#38bdf8" name="audio" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-panel">
            <h2>Latency & quality</h2>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={nodeHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(v) => new Date(v).toLocaleTimeString()}
                  stroke="#64748b"
                  fontSize={11}
                />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip contentStyle={{ background: '#111827', border: '1px solid #334155' }} />
                <Legend />
                <Line type="monotone" dataKey="latency" stroke="#f59e0b" name="Latency ms" dot={false} />
                <Line type="monotone" dataKey="quality" stroke="#10b981" name="Quality %" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="topics-panel">
            <h2>Topics</h2>
            <ul className="topics-list">
              {topics.map((topic) => (
                <li key={topic.name} className={`topic-item${topic.live ? ' topic-item--live' : ''}`}>
                  <span className="topic-name">{topic.name}</span>
                  <span className="topic-type">{topic.type}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VisualMonitoring;
