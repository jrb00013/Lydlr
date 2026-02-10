import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './Dashboard.css';
import { useSmartPolling } from '../hooks/useSmartPolling';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

function Dashboard() {
  const [stats, setStats] = useState({
    total_nodes: 0,
    active_nodes: 0,
    avg_compression_ratio: 0,
    avg_latency_ms: 0,
    avg_quality_score: 0
  });
  const [metricsHistory, setMetricsHistory] = useState([]);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_URL}/api/stats`);
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const updateMetrics = (newMetric) => {
    setMetricsHistory(prev => {
      const updated = [...prev, {
        timestamp: new Date().toLocaleTimeString(),
        compression: newMetric.compression_ratio,
        latency: newMetric.latency_ms,
        quality: newMetric.quality_score
      }];
      return updated.slice(-20); // Keep last 20 points
    });
  };

  useEffect(() => {
    // Fetch initial stats
    fetchStats();

    // Connect WebSocket for real-time updates
    const websocket = new WebSocket(`${WS_URL}/ws/metrics`);
    
    websocket.onopen = () => {
      console.log('WebSocket Connected');
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'metrics_update') {
        updateMetrics(data.data);
      }
    };

    websocket.onerror = (error) => {
      console.error('WebSocket Error:', error);
    };

    return () => {
      if (websocket) websocket.close();
    };
  }, []);

  // Smart polling for stats - less frequent since WebSocket handles real-time updates
  useSmartPolling(fetchStats, {
    interval: 30000, // 30 seconds (was 5 seconds) - WebSocket handles real-time
    enabled: true,
    immediate: false,
    minInterval: 20000,
    maxInterval: 60000,
    onError: (error) => {
      console.warn('Stats polling error:', error);
    }
  });

  return (
    <div className="dashboard">
      <h1 className="page-title">Dashboard</h1>
      
      <div className="stats-grid grid grid-4">
        <div className="stat-card card">
          <div className="stat-icon"></div>
          <div className="stat-value">{stats.active_nodes}/{stats.total_nodes}</div>
          <div className="stat-label">Active Nodes</div>
        </div>

        <div className="stat-card card">
          <div className="stat-icon"></div>
          <div className="stat-value">{stats.avg_compression_ratio.toFixed(2)}x</div>
          <div className="stat-label">Avg Compression</div>
        </div>

        <div className="stat-card card">
          <div className="stat-icon"></div>
          <div className="stat-value">{stats.avg_latency_ms.toFixed(1)}ms</div>
          <div className="stat-label">Avg Latency</div>
        </div>

        <div className="stat-card card">
          <div className="stat-icon"></div>
          <div className="stat-value">{(stats.avg_quality_score * 100).toFixed(1)}%</div>
          <div className="stat-label">Avg Quality</div>
        </div>
      </div>

      <div className="charts-section">
        <div className="card">
          <h2>Real-Time Metrics</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={metricsHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="compression" 
                stroke="#8884d8" 
                name="Compression Ratio"
                strokeWidth={2}
              />
              <Line 
                yAxisId="right"
                type="monotone" 
                dataKey="latency" 
                stroke="#82ca9d" 
                name="Latency (ms)"
                strokeWidth={2}
              />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="quality" 
                stroke="#ffc658" 
                name="Quality Score"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="system-info card">
        <h2>System Information</h2>
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">System Status:</span>
            <span className="info-value status-online">● Online</span>
          </div>
          <div className="info-item">
            <span className="info-label">Version:</span>
            <span className="info-value">1.0.0</span>
          </div>
          <div className="info-item">
            <span className="info-label">Uptime:</span>
            <span className="info-value">Real-time</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;

