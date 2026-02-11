import React, { useState, useEffect, useContext } from 'react';
import './NodesView.css';
import { NotificationContext, ConfirmContext } from '../App';
import { useSmartPolling } from '../hooks/useSmartPolling';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function NodesView() {
  const notification = useContext(NotificationContext);
  const confirm = useContext(ConfirmContext);
  const [nodes, setNodes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddNode, setShowAddNode] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [selectedNodeForLogs, setSelectedNodeForLogs] = useState(null);
  const [nodeLogs, setNodeLogs] = useState([]);
  const [models, setModels] = useState([]);
  const [showDeployModal, setShowDeployModal] = useState(false);
  const [selectedNodeForDeploy, setSelectedNodeForDeploy] = useState(null);
  const [selectedModelVersion, setSelectedModelVersion] = useState('');
  const [newNodeId, setNewNodeId] = useState('');
  const [nodeConfig, setNodeConfig] = useState({
    target_node_count: 2,
    auto_scale: false,
    min_nodes: 1,
    max_nodes: 10
  });

  const fetchNodes = async () => {
    try {
      const response = await fetch(`${API_URL}/api/nodes/`);
      const data = await response.json();
      setNodes(data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch nodes:', error);
      setLoading(false);
    }
  };

  const fetchNodeConfig = async () => {
    try {
      const response = await fetch(`${API_URL}/api/nodes/config/`);
      const data = await response.json();
      setNodeConfig(data);
    } catch (error) {
      console.error('Failed to fetch node config:', error);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await fetch(`${API_URL}/api/models/`);
      const data = await response.json();
      setModels(data);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const fetchNodeLogs = async () => {
    if (!selectedNodeForLogs) return;
    try {
      const response = await fetch(`${API_URL}/api/nodes/${selectedNodeForLogs}/logs/?lines=200`);
      const data = await response.json();
      setNodeLogs(data.logs || []);
    } catch (error) {
      console.error('Failed to fetch node logs:', error);
    }
  };


  // Initial fetch
  useEffect(() => {
    fetchNodes();
    fetchNodeConfig();
    fetchModels();
  }, []);

  // Smart polling for nodes - only when tab is visible, with backoff
  useSmartPolling(fetchNodes, {
    interval: 15000, // 15 seconds (was 3 seconds)
    enabled: true,
    immediate: false,
    minInterval: 10000,
    maxInterval: 30000,
    onError: (error) => {
      console.warn('Node polling error:', error);
    }
  });

  // Smart polling for logs - only when modal is open and visible
  useSmartPolling(() => {
    if (showLogs && selectedNodeForLogs) {
      fetchNodeLogs();
    }
  }, {
    interval: 5000, // 5 seconds for logs (was 2 seconds)
    enabled: showLogs && selectedNodeForLogs !== null,
    immediate: true,
    minInterval: 3000,
    maxInterval: 10000,
  });

  const handleAddNode = async () => {
    try {
      const payload = newNodeId ? { node_id: newNodeId } : {};
      const response = await fetch(`${API_URL}/api/nodes/create/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      
      if (response.ok) {
        const data = await response.json();
        notification.showSuccess(`Node ${data.node_id} created successfully!`);
        setShowAddNode(false);
        setNewNodeId('');
        fetchNodes();
      } else {
        const error = await response.json();
        notification.showError(`Failed to create node: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to create node:', error);
      notification.showError('Failed to create node');
    }
  };

  const handleDeleteNode = async (nodeId) => {
    const confirmed = await confirm.confirm({
      title: 'Delete Node',
      message: `Are you sure you want to delete node ${nodeId}? This action cannot be undone.`,
      confirmText: 'Delete',
      cancelText: 'Cancel',
      type: 'danger'
    });

    if (!confirmed) {
      return;
    }

    try {
      const response = await fetch(`${API_URL}/api/nodes/${nodeId}/delete/`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        notification.showSuccess(`Node ${nodeId} deleted successfully!`);
        fetchNodes();
      } else {
        const error = await response.json();
        notification.showError(`Failed to delete node: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to delete node:', error);
      notification.showError('Failed to delete node');
    }
  };

  const handleUpdateConfig = async () => {
    try {
      const response = await fetch(`${API_URL}/api/nodes/config/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(nodeConfig),
      });
      
      if (response.ok) {
        notification.showSuccess('Node configuration updated successfully!');
        setShowConfig(false);
        fetchNodes();
      } else {
        const error = await response.json();
        notification.showError(`Failed to update config: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to update config:', error);
      notification.showError('Failed to update configuration');
    }
  };

  const handleNodeAction = async (nodeId, action, event) => {
    let button = null;
    try {
      // Disable button during operation
      if (event && event.target) {
        button = event.target;
        button.disabled = true;
        const originalText = button.textContent;
        button.textContent = `${action === 'start' ? 'Starting' : action === 'stop' ? 'Stopping' : 'Restarting'}...`;
        
        // Store original text for restoration
        button.dataset.originalText = originalText;
      }

      const response = await fetch(`${API_URL}/api/nodes/${nodeId}/${action}/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({})
      });
      
      const data = await response.json();
      
      if (response.ok) {
        if (action === 'start' && data.status === 'started') {
          notification.showSuccess(`Node ${nodeId} started successfully! PID: ${data.pid || 'N/A'}`);
        } else if (action === 'start' && data.status === 'already_running') {
          notification.showInfo(`Node ${nodeId} is already running (PID: ${data.pid || 'N/A'})`);
        } else if (action === 'stop' && data.status === 'stopped') {
          notification.showSuccess(`Node ${nodeId} stopped successfully`);
        } else if (action === 'stop' && data.status === 'not_running') {
          notification.showInfo(`Node ${nodeId} is not running`);
        } else if (action === 'restart' && data.status === 'started') {
          notification.showSuccess(`Node ${nodeId} restarted successfully! PID: ${data.pid || 'N/A'}`);
        } else {
          notification.showSuccess(data.message || `${action.charAt(0).toUpperCase() + action.slice(1)} command completed`);
        }
        // Refresh nodes list after a short delay to see updated status
        setTimeout(() => {
          fetchNodes();
        }, 500);
      } else {
        const errorMsg = data.detail || data.error || `Failed to ${action} node: Unknown error`;
        notification.showError(errorMsg);
        console.error(`Failed to ${action} node ${nodeId}:`, data);
      }
    } catch (error) {
      console.error(`Failed to ${action} node:`, error);
      notification.showError(`Failed to ${action} node: ${error.message || 'Network error'}`);
    } finally {
      // Re-enable button
      if (button) {
        button.disabled = false;
        button.textContent = button.dataset.originalText || (action.charAt(0).toUpperCase() + action.slice(1));
      }
    }
  };

  const handleDeployToNode = async (nodeId, modelVersion) => {
    if (!modelVersion) {
      notification.showWarning('Please select a model version');
      return;
    }

    try {
      const response = await fetch(`${API_URL}/api/nodes/${nodeId}/deploy/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_version: modelVersion })
      });

      const data = await response.json();

      if (response.ok) {
        notification.showSuccess(`Model ${modelVersion} deployed to ${nodeId} successfully!`);
        setShowDeployModal(false);
        setSelectedNodeForDeploy(null);
        setSelectedModelVersion('');
        fetchNodes();
      } else {
        notification.showError(data.detail || 'Failed to deploy model');
      }
    } catch (error) {
      console.error('Failed to deploy model:', error);
      notification.showError(`Failed to deploy model: ${error.message}`);
    }
  };

  const openLogsModal = (nodeId) => {
    setSelectedNodeForLogs(nodeId);
    setShowLogs(true);
    fetchNodeLogs();
  };

  const openDeployModal = (nodeId) => {
    setSelectedNodeForDeploy(nodeId);
    setShowDeployModal(true);
  };

  if (loading) {
    return <div className="loading">Loading nodes...</div>;
  }

  return (
    <div className="nodes-view">
      <div className="page-header">
        <h1 className="page-title">Edge Nodes</h1>
        <div className="header-actions">
          <button 
            className="btn btn-primary"
            onClick={() => setShowAddNode(true)}
          >
            + Add Node
          </button>
          <button 
            className="btn btn-secondary"
            onClick={() => setShowConfig(true)}
          >
            Configure
          </button>
        </div>
      </div>

      {/* Add Node Modal */}
      {showAddNode && (
        <div className="modal-overlay" onClick={() => setShowAddNode(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Add New Node</h2>
            <div className="form-group">
              <label>Node ID (optional - auto-generated if empty):</label>
              <input
                type="text"
                value={newNodeId}
                onChange={(e) => setNewNodeId(e.target.value)}
                placeholder="e.g., node_5"
              />
            </div>
            <div className="modal-actions">
              <button className="btn btn-primary" onClick={handleAddNode}>
                Create Node
              </button>
              <button className="btn btn-secondary" onClick={() => setShowAddNode(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Deploy Modal */}
      {showDeployModal && (
        <div className="modal-overlay" onClick={() => setShowDeployModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Deploy Model to {selectedNodeForDeploy}</h2>
            <div className="form-group">
              <label>Select Model Version:</label>
              <select
                value={selectedModelVersion}
                onChange={(e) => setSelectedModelVersion(e.target.value)}
                className="form-select"
              >
                <option value="">-- Select Model --</option>
                {models.map(model => (
                  <option key={model.version} value={model.version}>
                    {model.version} ({model.size_mb?.toFixed(2) || 'N/A'} MB)
                  </option>
                ))}
              </select>
            </div>
            <div className="modal-actions">
              <button 
                className="btn btn-primary" 
                onClick={() => handleDeployToNode(selectedNodeForDeploy, selectedModelVersion)}
                disabled={!selectedModelVersion}
              >
                Deploy
              </button>
              <button className="btn btn-secondary" onClick={() => {
                setShowDeployModal(false);
                setSelectedNodeForDeploy(null);
                setSelectedModelVersion('');
              }}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Logs Modal */}
      {showLogs && (
        <div className="modal-overlay" onClick={() => setShowLogs(false)}>
          <div className="modal-content logs-modal" onClick={(e) => e.stopPropagation()}>
            <div className="logs-header">
              <h2>Logs: {selectedNodeForLogs}</h2>
              <button className="btn btn-secondary" onClick={() => setShowLogs(false)}>Close</button>
            </div>
            <div className="logs-content">
              {nodeLogs.length === 0 ? (
                <p className="no-logs">No logs available yet</p>
              ) : (
                <pre className="logs-text">
                  {nodeLogs.join('')}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Configuration Modal */}
      {showConfig && (
        <div className="modal-overlay" onClick={() => setShowConfig(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Node Configuration</h2>
            <div className="form-group">
              <label>Target Node Count:</label>
              <input
                type="number"
                min="0"
                max="100"
                value={nodeConfig.target_node_count}
                onChange={(e) => setNodeConfig({
                  ...nodeConfig,
                  target_node_count: parseInt(e.target.value)
                })}
              />
            </div>
            <div className="form-group">
              <label>
                <input
                  type="checkbox"
                  checked={nodeConfig.auto_scale}
                  onChange={(e) => setNodeConfig({
                    ...nodeConfig,
                    auto_scale: e.target.checked
                  })}
                />
                Auto-scale nodes
              </label>
            </div>
            <div className="form-group">
              <label>Min Nodes:</label>
              <input
                type="number"
                min="0"
                value={nodeConfig.min_nodes}
                onChange={(e) => setNodeConfig({
                  ...nodeConfig,
                  min_nodes: parseInt(e.target.value)
                })}
              />
            </div>
            <div className="form-group">
              <label>Max Nodes:</label>
              <input
                type="number"
                min="1"
                value={nodeConfig.max_nodes}
                onChange={(e) => setNodeConfig({
                  ...nodeConfig,
                  max_nodes: parseInt(e.target.value)
                })}
              />
            </div>
            <div className="modal-actions">
              <button className="btn btn-primary" onClick={handleUpdateConfig}>
                Save Configuration
              </button>
              <button className="btn btn-secondary" onClick={() => setShowConfig(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="nodes-grid grid grid-2">
        {nodes.length === 0 ? (
          <div className="card">
            <p>No nodes connected. Click "Add Node" to create one.</p>
          </div>
        ) : (
          nodes.map(node => (
            <div key={node.node_id} className="node-card card">
              <div className="node-header">
                <h3>{node.node_id}</h3>
                <span className={`node-status status-${node.status}`}>
                  {node.status}
                </span>
              </div>

              {/* Process Info */}
              {node.process_info && (
                <div className="process-info">
                  <div className="process-detail">
                    <span className="process-label">PID:</span>
                    <span className="process-value">{node.process_info.pid || 'N/A'}</span>
                  </div>
                  {node.process_info.log_file && (
                    <div className="process-detail">
                      <span className="process-label">Log:</span>
                      <span className="process-value log-path">{node.process_info.log_file.split('/').pop()}</span>
                    </div>
                  )}
                </div>
              )}

              <div className="node-metrics">
                <div className="metric">
                  <span className="metric-label">Model:</span>
                  <span className="metric-value">{node.model_version || 'N/A'}</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Compression:</span>
                  <span className="metric-value">{(node.compression_ratio || 0).toFixed(2)}x</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Latency:</span>
                  <span className="metric-value">{(node.latency_ms || 0).toFixed(1)}ms</span>
                </div>
                <div className="metric">
                  <span className="metric-label">Quality:</span>
                  <span className="metric-value">{((node.quality_score || 0) * 100).toFixed(1)}%</span>
                </div>
              </div>

              <div className="node-actions">
                <div className="action-row">
                  {node.status === 'running' ? (
                    <button 
                      className="btn btn-danger"
                      onClick={(e) => handleNodeAction(node.node_id, 'stop', e)}
                    >
                      Stop
                    </button>
                  ) : (
                    <button 
                      className="btn btn-primary"
                      onClick={(e) => handleNodeAction(node.node_id, 'start', e)}
                    >
                      Start
                    </button>
                  )}
                  <button 
                    className="btn btn-secondary"
                    onClick={(e) => handleNodeAction(node.node_id, 'restart', e)}
                  >
                    Restart
                  </button>
                </div>
                <div className="action-row">
                  <button 
                    className="btn btn-info"
                    onClick={() => openDeployModal(node.node_id)}
                  >
                    Deploy Model
                  </button>
                  <button 
                    className="btn btn-secondary"
                    onClick={() => openLogsModal(node.node_id)}
                  >
                    View Logs
                  </button>
                </div>
                <div className="action-row">
                  <button 
                    className="btn btn-danger"
                    onClick={() => handleDeleteNode(node.node_id)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default NodesView;
