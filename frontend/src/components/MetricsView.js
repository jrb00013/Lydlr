import React, { useState, useEffect, useCallback } from 'react';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import './MetricsView.css';
import PageHeader from './ui/PageHeader';
import LoadingSpinner from './ui/LoadingSpinner';
import DataTable from './ui/DataTable';
import lydlrApi from '../api/lydlrApi';
import { useSmartPolling } from '../hooks/useSmartPolling';

function MetricsView() {
  const [tab, setTab] = useState('samples');
  const [sampleRows, setSampleRows] = useState([]);
  const [rollupRows, setRollupRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [nodes, setNodes] = useState([]);
  const [selectedNode, setSelectedNode] = useState('');
  const [selectedVertical, setSelectedVertical] = useState('');
  const [loading, setLoading] = useState(true);

  const sampleColumns = [
    {
      key: 'timestamp',
      label: 'Time',
      sortable: true,
    },
    { key: 'node_id', label: 'Node' },
    {
      key: 'vertical',
      label: 'Vertical',
      render: (r) => (
        <span className={`badge badge--${r.vertical === 'iot' ? 'iot' : 'drone'}`}>
          {r.vertical || '—'}
        </span>
      ),
    },
    {
      key: 'compression_ratio',
      label: 'Ratio',
      render: (r) => <strong>{r.compression_ratio}×</strong>,
    },
    { key: 'latency_ms', label: 'Latency (ms)' },
    {
      key: 'quality_score',
      label: 'Quality',
      render: (r) => `${(r.quality_score * 100).toFixed(1)}%`,
    },
    { key: 'compression_level', label: 'Level' },
    { key: 'bandwidth_estimate', label: 'Bandwidth' },
  ];

  const rollupColumns = [
    { key: 'bucket_start', label: 'Hour bucket', sortable: true },
    { key: 'node_id', label: 'Node' },
    {
      key: 'vertical',
      label: 'Vertical',
      render: (r) => (
        <span className={`badge badge--${r.vertical === 'iot' ? 'iot' : 'drone'}`}>
          {r.vertical || '—'}
        </span>
      ),
    },
    { key: 'samples', label: 'Samples' },
    {
      key: 'avg_compression',
      label: 'Avg ratio',
      render: (r) => `${r.avg_compression}×`,
    },
    { key: 'avg_latency_ms', label: 'Avg latency' },
    {
      key: 'avg_quality',
      label: 'Avg quality',
      render: (r) => `${(r.avg_quality * 100).toFixed(1)}%`,
    },
    {
      key: 'min_compression',
      label: 'Min',
      render: (r) => (r.min_compression != null ? `${r.min_compression}×` : '—'),
    },
    {
      key: 'max_compression',
      label: 'Max',
      render: (r) => (r.max_compression != null ? `${r.max_compression}×` : '—'),
    },
  ];

  const fetchNodes = useCallback(async () => {
    try {
      const data = await lydlrApi.nodes();
      setNodes(data);
    } catch (e) {
      console.error(e);
    }
  }, []);

  const fetchSamples = useCallback(async () => {
    try {
      const params = { limit: 100 };
      if (selectedNode) params.node_id = selectedNode;
      if (selectedVertical) params.vertical = selectedVertical;
      const data = await lydlrApi.metricsTable(params);
      setSampleRows(
        (data.rows || []).map((r, i) => ({
          ...r,
          _rowKey: `${r.node_id}-${r.timestamp}-${i}`,
        }))
      );
      setTotal(data.total || 0);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [selectedNode, selectedVertical]);

  const fetchRollups = useCallback(async () => {
    try {
      const params = { limit: 72 };
      if (selectedNode) params.node_id = selectedNode;
      const data = await lydlrApi.metricsRollups(params);
      setRollupRows(data.rows || []);
    } catch (e) {
      console.error(e);
    }
  }, [selectedNode]);

  const refresh = useCallback(async () => {
    if (tab === 'samples') await fetchSamples();
    else await fetchRollups();
  }, [tab, fetchSamples, fetchRollups]);

  useEffect(() => {
    fetchNodes();
  }, [fetchNodes]);

  useEffect(() => {
    setLoading(true);
    refresh();
  }, [tab, selectedNode, selectedVertical, refresh]);

  useSmartPolling(refresh, {
    interval: 10000,
    enabled: true,
    immediate: false,
    minInterval: 8000,
    maxInterval: 30000,
  });

  if (loading && sampleRows.length === 0 && rollupRows.length === 0) {
    return <LoadingSpinner message="Loading telemetry…" />;
  }

  return (
    <div className="metrics-view page-enter">
      <PageHeader
        title="Telemetry data store"
        subtitle="Raw compression samples (7-day TTL) and hourly rollups — drone & IoT edge fleet"
        icon={AnalyticsIcon}
        badge={<span className="badge badge--registered">{total} samples</span>}
      />

      <div className="metrics-toolbar card">
        <div className="metrics-tabs">
          <button
            type="button"
            className={tab === 'samples' ? 'metrics-tab metrics-tab--active' : 'metrics-tab'}
            onClick={() => setTab('samples')}
          >
            Raw samples
          </button>
          <button
            type="button"
            className={tab === 'rollups' ? 'metrics-tab metrics-tab--active' : 'metrics-tab'}
            onClick={() => setTab('rollups')}
          >
            Hourly rollups
          </button>
        </div>
        <div className="metrics-filters">
          <select value={selectedNode} onChange={(e) => setSelectedNode(e.target.value)}>
            <option value="">All nodes</option>
            {nodes.map((n) => (
              <option key={n.node_id} value={n.node_id}>
                {n.display_name || n.node_id}
              </option>
            ))}
          </select>
          <select value={selectedVertical} onChange={(e) => setSelectedVertical(e.target.value)}>
            <option value="">All verticals</option>
            <option value="drone">Drone</option>
            <option value="iot">IoT</option>
          </select>
        </div>
      </div>

      <div className="card metrics-table-card">
        {tab === 'samples' ? (
          <DataTable
            columns={sampleColumns}
            rows={sampleRows}
            keyField="_rowKey"
            emptyMessage="No metrics yet — start ROS2 edge compressors"
            pageSize={20}
          />
        ) : (
          <DataTable
            columns={rollupColumns}
            rows={rollupRows}
            keyField="rollup_key"
            emptyMessage="Rollups populate after the first hour of samples"
            pageSize={24}
          />
        )}
      </div>
    </div>
  );
}

export default MetricsView;
