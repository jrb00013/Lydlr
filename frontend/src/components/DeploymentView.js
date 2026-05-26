import React, { useState, useEffect, useContext, useCallback } from 'react';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import FlightIcon from '@mui/icons-material/Flight';
import SensorsIcon from '@mui/icons-material/Sensors';
import './DeploymentView.css';
import { NotificationContext } from '../App';
import PageHeader from './ui/PageHeader';
import LoadingSpinner from './ui/LoadingSpinner';
import lydlrApi from '../api/lydlrApi';

function DeploymentView() {
  const notification = useContext(NotificationContext);
  const [models, setModels] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [deployments, setDeployments] = useState([]);
  const [deployStrategy, setDeployStrategy] = useState('fleet');
  const [deploying, setDeploying] = useState(false);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const [modelData, nodeData, deployData] = await Promise.all([
        lydlrApi.models(),
        lydlrApi.nodes(),
        lydlrApi.deployments(),
      ]);
      setModels(modelData);
      setNodes(nodeData);
      setDeployments(deployData);
    } catch (error) {
      console.error('Failed to load deployment data:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleDeploy = async () => {
    if (!selectedModel || selectedNodes.length === 0) {
      notification.showWarning('Select a model and at least one node');
      return;
    }

    setDeploying(true);
    try {
      const result = await lydlrApi.deploy({
        model_version: selectedModel,
        node_ids: selectedNodes,
        strategy: deployStrategy,
      });
      const rosCount = (result.ros_deployed || []).length;
      notification.showSuccess(
        `Deployed ${selectedModel} to ${result.successful_nodes?.length || selectedNodes.length} node(s)` +
          (rosCount ? ` — ROS notified on ${rosCount}` : '')
      );
      await refresh();
      setSelectedModel('');
      setSelectedNodes([]);
    } catch (error) {
      notification.showError(error.message || 'Deployment failed');
    } finally {
      setDeploying(false);
    }
  };

  const toggleNodeSelection = (nodeId) => {
    setSelectedNodes((prev) =>
      prev.includes(nodeId) ? prev.filter((id) => id !== nodeId) : [...prev, nodeId]
    );
  };

  const droneNodes = nodes.filter((n) => n.vertical === 'drone');
  const iotNodes = nodes.filter((n) => n.vertical === 'iot');

  if (loading) {
    return <LoadingSpinner message="Loading deploy console…" />;
  }

  return (
    <div className="deployment-view page-enter">
      <PageHeader
        title="Model deployment"
        subtitle="Push compressor weights to UAV and IoT edge nodes — files copied + ROS deploy topic"
        icon={RocketLaunchIcon}
      />

      <div className="deployment-grid">
        <div className="card deploy-card">
          <h2>Deploy to fleet</h2>

          <div className="form-group">
            <label htmlFor="deploy-model">Model version</label>
            <select
              id="deploy-model"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="form-select"
            >
              <option value="">— Choose model —</option>
              {models.map((model) => (
                <option key={model.version} value={model.version}>
                  {model.version} ({model.size_mb?.toFixed(2)} MB)
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Target nodes</label>
            {nodes.length === 0 ? (
              <p className="no-nodes">No fleet nodes — start stack with ./start-lydlr.sh -d --ros2</p>
            ) : (
              <>
                {droneNodes.length > 0 && (
                  <div className="nodes-checklist">
                    <span className="nodes-checklist__label">
                      <FlightIcon fontSize="small" /> UAV
                    </span>
                    {droneNodes.map((node) => (
                      <NodeCheckbox
                        key={node.node_id}
                        node={node}
                        checked={selectedNodes.includes(node.node_id)}
                        onToggle={toggleNodeSelection}
                      />
                    ))}
                  </div>
                )}
                {iotNodes.length > 0 && (
                  <div className="nodes-checklist">
                    <span className="nodes-checklist__label">
                      <SensorsIcon fontSize="small" /> IoT
                    </span>
                    {iotNodes.map((node) => (
                      <NodeCheckbox
                        key={node.node_id}
                        node={node}
                        checked={selectedNodes.includes(node.node_id)}
                        onToggle={toggleNodeSelection}
                      />
                    ))}
                  </div>
                )}
              </>
            )}
          </div>

          <div className="form-group">
            <label htmlFor="deploy-strategy">Deploy strategy</label>
            <select
              id="deploy-strategy"
              value={deployStrategy}
              onChange={(e) => setDeployStrategy(e.target.value)}
              className="form-select"
            >
              <option value="fleet">Fleet — all selected nodes</option>
              <option value="canary">Canary — first node only</option>
            </select>
          </div>

          <button
            type="button"
            className="btn btn-primary deploy-btn"
            onClick={handleDeploy}
            disabled={deploying || !selectedModel || selectedNodes.length === 0}
          >
            {deploying ? 'Deploying…' : 'Deploy model'}
          </button>
        </div>

        <div className="card history-card">
          <h2>Deployment history</h2>
          <div className="deployments-list">
            {deployments.length === 0 ? (
              <p className="no-deployments">No deployments yet</p>
            ) : (
              deployments.map((deployment) => (
                <div key={deployment._id || deployment.deployment_id} className="deployment-item">
                  <div className="deployment-header">
                    <span className="deployment-model">{deployment.model_version}</span>
                    <span className={`deployment-status status-${deployment.status}`}>
                      {deployment.status}
                    </span>
                  </div>
                  <div className="deployment-details">
                    <span>Nodes: {(deployment.node_ids || []).join(', ')}</span>
                    <span className="deployment-time">
                      {deployment.deployed_at
                        ? new Date(deployment.deployed_at).toLocaleString()
                        : '—'}
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

function NodeCheckbox({ node, checked, onToggle }) {
  return (
    <label className="checkbox-label">
      <input type="checkbox" checked={checked} onChange={() => onToggle(node.node_id)} />
      <span>{node.display_name || node.node_id}</span>
      <span className={`node-badge status-${node.status}`}>{node.status}</span>
    </label>
  );
}

export default DeploymentView;
