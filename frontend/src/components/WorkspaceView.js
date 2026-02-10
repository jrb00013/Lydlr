import React, { useState, useEffect, useContext } from 'react';
import { NotificationContext } from '../App';
import './WorkspaceView.css';

const WorkspaceView = () => {
  const notification = useContext(NotificationContext);
  const [workspaceInfo, setWorkspaceInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [building, setBuilding] = useState(false);
  const [buildOutput, setBuildOutput] = useState([]);
  const [showOutput, setShowOutput] = useState(false);

  const fetchWorkspaceInfo = async () => {
    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/workspace/`);
      if (response.ok) {
        const data = await response.json();
        setWorkspaceInfo(data);
      } else {
        notification.showError('Failed to fetch workspace info');
      }
    } catch (error) {
      console.error('Error fetching workspace info:', error);
      notification.showError('Failed to fetch workspace info');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWorkspaceInfo();
    // Refresh every 10 seconds
    const interval = setInterval(fetchWorkspaceInfo, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleBuild = async () => {
    setBuilding(true);
    setBuildOutput([]);
    setShowOutput(true);
    notification.showInfo('Building workspace... This may take a few minutes.');

    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/api/workspace/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action: 'build' }),
      });

      const data = await response.json();
      
      if (response.ok && data.status === 'success') {
        notification.showSuccess('Workspace built successfully!');
        setBuildOutput(data.output || []);
        // Refresh workspace info
        await fetchWorkspaceInfo();
      } else {
        notification.showError(data.error || 'Build failed');
        setBuildOutput(data.output || [data.error] || ['Build failed']);
      }
    } catch (error) {
      console.error('Error building workspace:', error);
      notification.showError('Failed to build workspace');
      setBuildOutput([`Error: ${error.message}`]);
    } finally {
      setBuilding(false);
    }
  };

  if (loading) {
    return (
      <div className="workspace-view">
        <div className="loading">Loading workspace information...</div>
      </div>
    );
  }

  if (!workspaceInfo) {
    return (
      <div className="workspace-view">
        <div className="error-message">Failed to load workspace information</div>
      </div>
    );
  }

  return (
    <div className="workspace-view">
      <div className="workspace-header">
        <h1>Workspace Management</h1>
        <button
          className={`btn btn-primary ${building ? 'btn-loading' : ''}`}
          onClick={handleBuild}
          disabled={building}
        >
          {building ? 'Building...' : 'Build Workspace'}
        </button>
      </div>

      <div className="workspace-info-grid">
        <div className="info-card">
          <h3>Workspace Status</h3>
          <div className="status-badge">
            <span className={`status-indicator ${workspaceInfo.is_built ? 'built' : 'not-built'}`}>
              {workspaceInfo.is_built ? '● Built' : '○ Not Built'}
            </span>
          </div>
          <div className="info-item">
            <label>Path:</label>
            <span className="monospace">{workspaceInfo.workspace_path || 'Not found'}</span>
          </div>
          <div className="info-item">
            <label>ROS2 Distribution:</label>
            <span>{workspaceInfo.ros2_distro || 'Not found'}</span>
          </div>
          <div className="info-item">
            <label>Execution Mode:</label>
            <span>{workspaceInfo.use_docker ? `Docker (${workspaceInfo.container})` : 'Local'}</span>
          </div>
        </div>

        <div className="info-card">
          <h3>Workspace Structure</h3>
          <div className="structure-item">
            <span className={`check ${workspaceInfo.has_src ? 'yes' : 'no'}`}>
              {workspaceInfo.has_src ? '✓' : '✗'}
            </span>
            <span>src/ directory</span>
          </div>
          <div className="structure-item">
            <span className={`check ${workspaceInfo.has_install ? 'yes' : 'no'}`}>
              {workspaceInfo.has_install ? '✓' : '✗'}
            </span>
            <span>install/ directory</span>
          </div>
          <div className="structure-item">
            <span className={`check ${workspaceInfo.is_built ? 'yes' : 'no'}`}>
              {workspaceInfo.is_built ? '✓' : '✗'}
            </span>
            <span>Workspace built</span>
          </div>
        </div>

        <div className="info-card">
          <h3>Packages</h3>
          {workspaceInfo.packages && workspaceInfo.packages.length > 0 ? (
            <div className="packages-list">
              {workspaceInfo.packages.map((pkg, idx) => (
                <div key={idx} className="package-tag">
                  {pkg}
                </div>
              ))}
            </div>
          ) : (
            <div className="no-packages">No packages found</div>
          )}
        </div>
      </div>

      {showOutput && buildOutput.length > 0 && (
        <div className="build-output-card">
          <div className="build-output-header">
            <h3>Build Output</h3>
            <button
              className="btn btn-small btn-secondary"
              onClick={() => setShowOutput(false)}
            >
              Hide
            </button>
          </div>
          <div className="build-output">
            {buildOutput.map((line, idx) => (
              <div key={idx} className="output-line">
                {line}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="workspace-actions">
        <div className="action-card">
          <h3>Quick Actions</h3>
          <div className="action-buttons">
            <button
              className="btn btn-secondary"
              onClick={fetchWorkspaceInfo}
              disabled={loading}
            >
              Refresh Status
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => setShowOutput(!showOutput)}
            >
              {showOutput ? 'Hide' : 'Show'} Build Output
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WorkspaceView;

