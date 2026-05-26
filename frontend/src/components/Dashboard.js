import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import FlightIcon from '@mui/icons-material/Flight';
import SensorsIcon from '@mui/icons-material/Sensors';
import SpeedIcon from '@mui/icons-material/Speed';
import SignalCellularAltIcon from '@mui/icons-material/SignalCellularAlt';
import HubIcon from '@mui/icons-material/Hub';
import MemoryIcon from '@mui/icons-material/Memory';
import PageHeader from './ui/PageHeader';
import LoadingSpinner from './ui/LoadingSpinner';
import './Dashboard.css';
import { useSmartPolling } from '../hooks/useSmartPolling';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

function Dashboard({ connected }) {
  const [stats, setStats] = useState({
    total_nodes: 0,
    active_nodes: 0,
    active_drones: 0,
    active_iot: 0,
    avg_compression_ratio: 0,
    avg_latency_ms: 0,
    avg_quality_score: 0,
    estimated_uplink_saved_kbps: 0,
    fleet_profile: 'drone_iot_edge',
    vertical: 'drone',
  });
  const [fleetNodes, setFleetNodes] = useState([]);
  const [metricsHistory, setMetricsHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  const pushMetric = useCallback((m) => {
    if (!m || m.compression_ratio == null) return;
    setMetricsHistory((prev) => {
      const label = m.node_id
        ? `${m.node_id.split('_').pop()} · ${new Date().toLocaleTimeString()}`
        : new Date().toLocaleTimeString();
      const point = {
        timestamp: label,
        compression: Number(m.compression_ratio) || 0,
        latency: Number(m.latency_ms) || 0,
        quality: Number(m.quality_score) || 0,
      };
      return [...prev, point].slice(-30);
    });
  }, []);

  const fetchStats = useCallback(async () => {
    const res = await fetch(`${API_URL}/api/stats/`);
    if (res.ok) setStats(await res.json());
  }, []);

  const fetchFleet = useCallback(async () => {
    const res = await fetch(`${API_URL}/api/nodes/`);
    if (res.ok) setFleetNodes(await res.json());
  }, []);

  const fetchRecentMetrics = useCallback(async () => {
    const res = await fetch(`${API_URL}/api/metrics/?limit=30`);
    if (!res.ok) return;
    const rows = await res.json();
    if (!rows.length) return;
    const history = [...rows].reverse().map((m) => ({
      timestamp: `${(m.node_id || '').replace('node_', '').replace('iot_gateway_', 'gw')} · ${new Date(m.timestamp).toLocaleTimeString()}`,
      compression: m.compression_ratio,
      latency: m.latency_ms,
      quality: m.quality_score,
    }));
    setMetricsHistory(history.slice(-30));
  }, []);

  const refreshAll = useCallback(async () => {
    try {
      await Promise.all([fetchStats(), fetchFleet(), fetchRecentMetrics()]);
    } catch (e) {
      console.warn('Dashboard refresh failed:', e);
    } finally {
      setLoading(false);
    }
  }, [fetchStats, fetchFleet, fetchRecentMetrics]);

  useEffect(() => {
    refreshAll();
    const ws = new WebSocket(`${WS_URL}/ws/metrics`);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'metrics_update') {
          pushMetric(data.data);
          fetchStats();
        }
      } catch (_) {
        /* ignore */
      }
    };
    return () => ws.close();
  }, [refreshAll, pushMetric, fetchStats]);

  useSmartPolling(refreshAll, {
    interval: 8000,
    enabled: connected,
    immediate: false,
    minInterval: 5000,
    maxInterval: 20000,
  });

  const droneNodes = fleetNodes.filter((n) => n.vertical === 'drone');
  const iotNodes = fleetNodes.filter((n) => n.vertical === 'iot');

  if (loading) {
    return <LoadingSpinner message="Loading edge fleet…" />;
  }

  return (
    <div className="dashboard page-enter">
      <PageHeader
        title="Drone & IoT Edge Console"
        subtitle="Bandwidth-adaptive multimodal compression for UAV downlinks and LPWAN edge gateways"
        icon={FlightIcon}
        badge={
          <span className="dashboard__profile-badge">
            {stats.fleet_profile || 'drone_iot_edge'}
          </span>
        }
      />

      <section className="dashboard__hero card">
        <div className="dashboard__hero-copy">
          <h2>Edge AI compression in flight</h2>
          <p>
            Compress camera, LiDAR, IMU, and telemetry at the edge before uplink.
            RL-tuned levels adapt to CPU load and link budget — built for drones on
            512&nbsp;kbps links and IoT gateways on 64&nbsp;kbps LPWAN.
          </p>
        </div>
        <div className="dashboard__hero-metrics">
          <div className="hero-stat">
            <span className="hero-stat__value">
              {stats.estimated_uplink_saved_kbps > 0
                ? `${stats.estimated_uplink_saved_kbps.toFixed(0)}`
                : '—'}
            </span>
            <span className="hero-stat__label">kbps saved (est.)</span>
          </div>
          <div className="hero-stat">
            <span className="hero-stat__value">
              {stats.avg_compression_ratio > 0
                ? `${stats.avg_compression_ratio.toFixed(1)}×`
                : '—'}
            </span>
            <span className="hero-stat__label">fleet compression</span>
          </div>
        </div>
      </section>

      <div className="stats-grid grid grid-4">
        <div className="stat-card card stat-card--drone">
          <FlightIcon className="stat-card__icon" />
          <div className="stat-value">
            {stats.active_drones}/{stats.total_nodes || '—'}
          </div>
          <div className="stat-label">UAV compressors</div>
        </div>
        <div className="stat-card card stat-card--iot">
          <SensorsIcon className="stat-card__icon" />
          <div className="stat-value">{stats.active_iot}</div>
          <div className="stat-label">IoT gateways</div>
        </div>
        <div className="stat-card card">
          <SpeedIcon className="stat-card__icon" />
          <div className="stat-value">
            {stats.avg_latency_ms > 0 ? `${stats.avg_latency_ms.toFixed(1)}` : '—'}
            <small>ms</small>
          </div>
          <div className="stat-label">edge latency</div>
        </div>
        <div className="stat-card card">
          <SignalCellularAltIcon className="stat-card__icon" />
          <div className="stat-value">
            {stats.avg_quality_score > 0
              ? `${(stats.avg_quality_score * 100).toFixed(0)}%`
              : '—'}
          </div>
          <div className="stat-label">perceptual quality</div>
        </div>
      </div>

      <div className="dashboard__fleet grid grid-2">
        <FleetPanel
          title="UAV fleet"
          icon={FlightIcon}
          variant="drone"
          nodes={droneNodes}
          emptyHint="Start ROS2 with --ros2 to stream UAV metrics"
        />
        <FleetPanel
          title="IoT edge"
          icon={SensorsIcon}
          variant="iot"
          nodes={iotNodes}
          emptyHint="iot_gateway_01 reports on LPWAN budget"
        />
      </div>

      <div className="charts-section">
        <div className="card chart-card">
          <h2>
            <HubIcon /> Live compression telemetry
          </h2>
          {metricsHistory.length === 0 ? (
            <p className="chart-empty">
              {connected
                ? 'Waiting for edge metrics — run ./start-lydlr.sh --build -d --ros2'
                : 'API offline — start the backend to see live data'}
            </p>
          ) : (
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={metricsHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
                <XAxis dataKey="timestamp" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="compression"
                  stroke="#60a5fa"
                  name="Compression ratio"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="latency"
                  stroke="#34d399"
                  name="Latency (ms)"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="quality"
                  stroke="#fbbf24"
                  name="Quality"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="card chart-card">
          <h2>
            <MemoryIcon /> Uplink efficiency
          </h2>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart
              data={[
                {
                  name: 'Raw uplink',
                  kbps: stats.estimated_uplink_saved_kbps + 400,
                },
                {
                  name: 'After Lydlr',
                  kbps: Math.max(400 - stats.estimated_uplink_saved_kbps, 80),
                },
              ]}
            >
              <defs>
                <linearGradient id="uplinkGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.15)" />
              <XAxis dataKey="name" />
              <YAxis unit=" kbps" />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="kbps"
                stroke="#60a5fa"
                fill="url(#uplinkGrad)"
                name="Uplink"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="system-info card">
        <h2>Edge deployment</h2>
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">Control plane</span>
            <span className={`info-value ${connected ? 'status-online' : ''}`}>
              {connected ? '● Live' : '○ Offline'}
            </span>
          </div>
          <div className="info-item">
            <span className="info-label">ROS 2 profile</span>
            <span className="info-value">{stats.vertical || 'drone'}</span>
          </div>
          <div className="info-item">
            <span className="info-label">Fleet</span>
            <span className="info-value">{stats.active_nodes} active nodes</span>
          </div>
          <div className="info-item">
            <span className="info-label">Quick start</span>
            <span className="info-value info-value--mono">
              ./start-lydlr.sh -d --ros2
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function FleetPanel({ title, icon: Icon, variant, nodes, emptyHint }) {
  return (
    <div className={`card fleet-panel fleet-panel--${variant}`}>
      <h2>
        <Icon /> {title}
      </h2>
      {nodes.length === 0 ? (
        <p className="fleet-panel__empty">{emptyHint}</p>
      ) : (
        <ul className="fleet-panel__list">
          {nodes.map((n) => (
            <li key={n.node_id} className="fleet-node">
              <div className="fleet-node__head">
                <span className="fleet-node__name">{n.display_name || n.node_id}</span>
                <span className={`fleet-node__status fleet-node__status--${n.status || 'unknown'}`}>
                  {n.status || 'unknown'}
                </span>
              </div>
              <div className="fleet-node__stats">
                <span>
                  {n.compression_ratio > 0 ? `${n.compression_ratio.toFixed(1)}×` : '—'} compress
                </span>
                <span>{n.latency_ms > 0 ? `${n.latency_ms.toFixed(0)} ms` : '—'}</span>
                <span>
                  {n.uplink_budget_kbps ? `${n.uplink_budget_kbps} kbps budget` : ''}
                </span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default Dashboard;
