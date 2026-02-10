import React, { useState, useEffect, useCallback } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './MetricsView.css';
import { useSmartPolling } from '../hooks/useSmartPolling';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function MetricsView() {
  const [metrics, setMetrics] = useState([]);
  const [selectedNode, setSelectedNode] = useState('all');
  const [nodes, setNodes] = useState([]);

  const fetchNodes = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/nodes`);
      const data = await response.json();
      setNodes(data);
    } catch (error) {
      console.error('Failed to fetch nodes:', error);
    }
  }, []);

  const fetchMetrics = useCallback(async () => {
    try {
      const url = selectedNode === 'all' 
        ? `${API_URL}/api/metrics?limit=50`
        : `${API_URL}/api/metrics?node_id=${selectedNode}&limit=50`;
      
      const response = await fetch(url);
      const data = await response.json();
      setMetrics(data.reverse()); // Show oldest to newest
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  }, [selectedNode]);

  // Initial fetch
  useEffect(() => {
    fetchNodes();
    fetchMetrics();
  }, [fetchNodes, fetchMetrics]);

  // Smart polling for metrics
  useSmartPolling(fetchMetrics, {
    interval: 20000, // 20 seconds (was 5 seconds)
    enabled: true,
    immediate: false,
    minInterval: 15000,
    maxInterval: 45000,
    onError: (error) => {
      console.warn('Metrics polling error:', error);
    }
  });

  const chartData = metrics.map((m, idx) => ({
    index: idx,
    compression: m.compression_ratio,
    latency: m.latency_ms,
    quality: m.quality_score * 100,
    bandwidth: m.bandwidth_estimate
  }));

  return (
    <div className="metrics-view">
      <div className="metrics-header">
        <h1 className="page-title">Performance Metrics</h1>
        <select 
          className="node-selector"
          value={selectedNode}
          onChange={(e) => setSelectedNode(e.target.value)}
        >
          <option value="all">All Nodes</option>
          {nodes.map(node => (
            <option key={node.node_id} value={node.node_id}>
              {node.node_id}
            </option>
          ))}
        </select>
      </div>

      <div className="charts-container">
        <div className="chart-card card">
          <h2>Compression Ratio Over Time</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="index" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="compression" 
                stroke="#8884d8" 
                name="Compression Ratio"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card card">
          <h2>Latency (ms)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="index" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="latency" 
                stroke="#82ca9d" 
                name="Latency"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card card">
          <h2>Quality Score (%)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="index" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="quality" fill="#ffc658" name="Quality %" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card card">
          <h2>Bandwidth Estimate</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="index" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="bandwidth" 
                stroke="#ff7300" 
                name="Bandwidth"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

export default MetricsView;

