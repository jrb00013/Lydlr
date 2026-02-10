import React, { useState, useEffect, useContext } from 'react';
import './DeploymentView.css';
import { NotificationContext } from '../App';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function DeploymentView() {
  const notification = useContext(NotificationContext);
  const [models, setModels] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [deployments, setDeployments] = useState([]);
  const [deploying, setDeploying] = useState(false);

  useEffect(() => {
    fetchModels();
    fetchNodes();
    fetchDeployments();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch(`${API_URL}/api/models`);
      const data = await response.json();
      setModels(data);
    } catch (error) {
      console.error('Failed to fetch models:', error);
    }
  };

  const fetchNodes = async () => {
    try {
      const response = await fetch(`${API_URL}/api/nodes`);
      const data = await response.json();
      setNodes(data);
    } catch (error) {
      console.error('Failed to fetch nodes:', error);
    }
  };

  const fetchDeployments = async () => {
    try {
      const response = await fetch(`${API_URL}/api/deployments`);
      const data = await response.json();
      setDeployments(data);
    } catch (error) {
      console.error('Failed to fetch deployments:', error);
    }
  };

  const handleDeploy = async () => {
    if (!selectedModel || selectedNodes.length === 0) {
      notification.showWarning('Please select a model and at least one node');
      return;
    }

    setDeploying(true);
    try {
      const response = await fetch(`${API_URL}/api/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_version: selectedModel,
          node_ids: selectedNodes
        })
      });
      if (response.ok) {
        notification.showSuccess('Deployment initiated!');
        fetchDeployments();
        setSelectedModel('');
        setSelectedNodes([]);
      } else {
        const error = await response.json();
        notification.showError(`Deployment failed: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to deploy:', error);
      notification.showError('Deployment failed');
    } finally {
      setDeploying(false);
    }
  };

  const toggleNodeSelection = (nodeId) => {
    setSelectedNodes(prev => 
      prev.includes(nodeId)
        ? prev.filter(id => id !== nodeId)
        : [...prev, nodeId]
    );
  };

  return (
    <div className="deployment-view">
      <h1 className="page-title">Model Deployment</h1>

      <div className="deployment-grid">
        <div className="card deploy-card">
          <h2>Deploy New Model</h2>
          
          <div className="form-group">
            <label>Select Model:</label>
            <select 
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="form-select"
            >
              <option value="">-- Choose Model --</option>
              {models.map(model => (
                <option key={model.version} value={model.version}>
                  {model.version} ({model.size_mb.toFixed(2)} MB)
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Select Target Nodes:</label>
            <div className="nodes-checklist">
              {nodes.length === 0 ? (
                <p className="no-nodes">No nodes available</p>
              ) : (
                nodes.map(node => (
                  <label key={node.node_id} className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={selectedNodes.includes(node.node_id)}
                      onChange={() => toggleNodeSelection(node.node_id)}
                    />
                    <span>{node.node_id}</span>
                    <span className={`node-badge status-${node.status}`}>
                      {node.status}
                    </span>
                  </label>
                ))
              )}
            </div>
          </div>

          <button 
            className="btn btn-primary deploy-btn"
            onClick={handleDeploy}
            disabled={deploying || !selectedModel || selectedNodes.length === 0}
          >
            {deploying ? 'Deploying...' : 'Deploy Model'}
          </button>
        </div>

        <div className="card history-card">
          <h2>Deployment History</h2>
          <div className="deployments-list">
            {deployments.length === 0 ? (
              <p className="no-deployments">No deployments yet</p>
            ) : (
              deployments.map((deployment, idx) => (
                <div key={idx} className="deployment-item">
                  <div className="deployment-header">
                    <span className="deployment-model">
                      {deployment.model_version}
                    </span>
                    <span className={`deployment-status status-${deployment.status}`}>
                      {deployment.status}
                    </span>
                  </div>
                  <div className="deployment-details">
                    <span>Nodes: {deployment.node_ids.join(', ')}</span>
                    <span className="deployment-time">
                      {new Date(deployment.deployed_at).toLocaleString()}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default DeploymentView;

