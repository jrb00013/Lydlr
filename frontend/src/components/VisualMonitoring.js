import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, Cell } from 'recharts';
import './VisualMonitoring.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

function VisualMonitoring() {
  const [nodes, setNodes] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [topics, setTopics] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [compressionHistory, setCompressionHistory] = useState([]);
  const [bandwidthData, setBandwidthData] = useState([]);
  const [qualityHeatmap, setQualityHeatmap] = useState([]);
  const wsRef = useRef(null);
  const maxHistoryLength = 100;

  const fetchNodes = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/api/nodes`);
      const data = await response.json();
      setNodes(data);
      if (data.length > 0 && !selectedNode) {
        setSelectedNode(data[0].node_id);
      }
    } catch (error) {
      console.error('Failed to fetch nodes:', error);
    }
  }, [selectedNode]);

  const fetchTopics = useCallback(async () => {
    try {
      // This would need a backend endpoint to get ROS2 topics
      // For now, we'll simulate with node-based topics
      const topicList = [];
      nodes.forEach(node => {
        topicList.push(
          { name: `/${node.node_id}/compressed`, type: 'compressed', node: node.node_id },
          { name: `/${node.node_id}/metrics`, type: 'metrics', node: node.node_id },
          { name: `/${node.node_id}/decompressed`, type: 'decompressed', node: node.node_id }
        );
      });
      setTopics(topicList);
    } catch (error) {
      console.error('Failed to fetch topics:', error);
    }
  }, [nodes]);

  const fetchMetrics = useCallback(async (nodeId) => {
    try {
      const response = await fetch(`${API_URL}/api/metrics?node_id=${nodeId}&limit=50`);
      const data = await response.json();
      
      if (data && data.length > 0) {
        const latest = data[0];
        setMetrics(prev => ({
          ...prev,
          [nodeId]: latest
        }));

        // Update compression history
        setCompressionHistory(prev => {
          const updated = [...prev, {
            timestamp: new Date().toISOString(),
            node: nodeId,
            compression: latest.compression_ratio,
            latency: latest.latency_ms,
            quality: latest.quality_score * 100,
            bandwidth: latest.bandwidth_estimate
          }];
          return updated.slice(-maxHistoryLength);
        });

        // Update bandwidth data
        setBandwidthData(prev => {
          const updated = [...prev, {
            time: new Date().toLocaleTimeString(),
            bandwidth: latest.bandwidth_estimate,
            compression: latest.compression_ratio
          }];
          return updated.slice(-50);
        });
      }
    } catch (error) {
      console.error(`Failed to fetch metrics for ${nodeId}:`, error);
    }
  }, []);

  useEffect(() => {
    fetchNodes();
    const interval = setInterval(() => {
      fetchNodes();
    }, 10000);
    return () => clearInterval(interval);
  }, [fetchNodes]);

  useEffect(() => {
    if (nodes.length > 0) {
      fetchTopics();
      nodes.forEach(node => {
        fetchMetrics(node.node_id);
      });
    }
  }, [nodes, fetchTopics, fetchMetrics]);

  useEffect(() => {
    // WebSocket connection for real-time updates
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket(`${WS_URL}/ws/metrics`);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('WebSocket connected for visual monitoring');
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'metrics_update' && data.data) {
              const metric = data.data;
              const nodeId = metric.node_id;
              
              setMetrics(prev => ({
                ...prev,
                [nodeId]: metric
              }));

              setCompressionHistory(prev => {
                const updated = [...prev, {
                  timestamp: new Date().toISOString(),
                  node: nodeId,
                  compression: metric.compression_ratio,
                  latency: metric.latency_ms,
                  quality: metric.quality_score * 100,
                  bandwidth: metric.bandwidth_estimate
                }];
                return updated.slice(-maxHistoryLength);
              });
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
          console.log('WebSocket closed, reconnecting...');
          setTimeout(connectWebSocket, 3000);
        };
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Generate quality heatmap data
  useEffect(() => {
    const heatmap = [];
    nodes.forEach((node, idx) => {
      const metric = metrics[node.node_id];
      if (metric) {
        heatmap.push({
          x: idx,
          y: metric.quality_score * 100,
          z: metric.compression_ratio,
          node: node.node_id,
          quality: metric.quality_score * 100
        });
      }
    });
    setQualityHeatmap(heatmap);
  }, [nodes, metrics]);

  const getQualityColor = (quality) => {
    if (quality >= 80) return '#00ff00';
    if (quality >= 60) return '#ffff00';
    if (quality >= 40) return '#ff8800';
    return '#ff0000';
  };

  const nodeHistory = compressionHistory.filter(m => m.node === selectedNode);

  return (
    <div className="visual-monitoring">
      <div className="monitoring-header">
        <h1 className="page-title">Visual Monitoring Dashboard</h1>
        <div className="node-selector-container">
          <label>Select Node:</label>
          <select 
            value={selectedNode || ''} 
            onChange={(e) => setSelectedNode(e.target.value)}
            className="node-selector"
          >
            <option value="">All Nodes</option>
            {nodes.map(node => (
              <option key={node.node_id} value={node.node_id}>
                {node.node_id} ({node.status})
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="monitoring-grid">
        {/* Real-time Metrics Cards */}
        <div className="metrics-cards">
          {nodes.map(node => {
            const metric = metrics[node.node_id];
            if (!metric) return null;
            
            return (
              <div key={node.node_id} className={`metric-card ${selectedNode === node.node_id ? 'selected' : ''}`}>
                <div className="metric-card-header">
                  <h3>{node.node_id}</h3>
                  <span className={`status-badge status-${node.status}`}>{node.status}</span>
                </div>
                <div className="metric-values">
                  <div className="metric-item">
                    <span className="metric-label">Compression:</span>
                    <span className="metric-value highlight">{metric.compression_ratio.toFixed(2)}x</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Latency:</span>
                    <span className="metric-value">{metric.latency_ms.toFixed(1)}ms</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Quality:</span>
                    <span className="metric-value" style={{ color: getQualityColor(metric.quality_score * 100) }}>
                      {(metric.quality_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">Bandwidth:</span>
                    <span className="metric-value">{(metric.bandwidth_estimate * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Compression Ratio Over Time */}
        <div className="chart-card card">
          <h2>Compression Ratio Over Time</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={nodeHistory.length > 0 ? nodeHistory : compressionHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis />
              <Tooltip 
                labelFormatter={(value) => new Date(value).toLocaleString()}
              />
              <Legend />
              {selectedNode ? (
                <Line 
                  type="monotone" 
                  dataKey="compression" 
                  stroke="#8884d8" 
                  name="Compression Ratio"
                  strokeWidth={2}
                  dot={false}
                />
              ) : (
                nodes.map((node, idx) => {
                  const nodeData = compressionHistory.filter(m => m.node === node.node_id);
                  const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00ff00'];
                  return (
                    <Line 
                      key={node.node_id}
                      type="monotone" 
                      dataKey="compression" 
                      data={nodeData}
                      stroke={colors[idx % colors.length]}
                      name={node.node_id}
                      strokeWidth={2}
                      dot={false}
                    />
                  );
                })
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Quality vs Compression Scatter */}
        <div className="chart-card card">
          <h2>Quality vs Compression Analysis</h2>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={qualityHeatmap}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" name="Node Index" />
              <YAxis dataKey="y" name="Quality Score (%)" />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                content={({ active, payload }) => {
                  if (active && payload && payload[0]) {
                    const data = payload[0].payload;
                    return (
                      <div className="custom-tooltip">
                        <p>Node: {data.node}</p>
                        <p>Quality: {data.quality.toFixed(1)}%</p>
                        <p>Compression: {data.z.toFixed(2)}x</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter name="Quality" dataKey="y" fill="#8884d8">
                {qualityHeatmap.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getQualityColor(entry.quality)} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Bandwidth Utilization */}
        <div className="chart-card card">
          <h2>Bandwidth Utilization</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={bandwidthData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="bandwidth" fill="#82ca9d" name="Bandwidth %" />
              <Bar dataKey="compression" fill="#8884d8" name="Compression Ratio" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Topics Overview */}
        <div className="topics-card card">
          <h2>Active Topics</h2>
          <div className="topics-list">
            {topics.map((topic, idx) => (
              <div key={idx} className="topic-item">
                <span className="topic-name">{topic.name}</span>
                <span className="topic-type">{topic.type}</span>
                <span className="topic-node">{topic.node}</span>
              </div>
            ))}
          </div>
        </div>

        {/* System Performance Summary */}
        <div className="summary-card card">
          <h2>System Performance Summary</h2>
          <div className="summary-stats">
            <div className="summary-item">
              <span className="summary-label">Total Nodes:</span>
              <span className="summary-value">{nodes.length}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Active Nodes:</span>
              <span className="summary-value">
                {nodes.filter(n => n.status === 'active').length}
              </span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Avg Compression:</span>
              <span className="summary-value">
                {Object.values(metrics).length > 0
                  ? (Object.values(metrics).reduce((sum, m) => sum + m.compression_ratio, 0) / Object.values(metrics).length).toFixed(2)
                  : '0.00'}x
              </span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Avg Quality:</span>
              <span className="summary-value">
                {Object.values(metrics).length > 0
                  ? (Object.values(metrics).reduce((sum, m) => sum + m.quality_score, 0) / Object.values(metrics).length * 100).toFixed(1)
                  : '0.0'}%
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VisualMonitoring;

