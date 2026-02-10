import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import Dashboard from './components/Dashboard';
import NodesView from './components/NodesView';
import ModelsView from './components/ModelsView';
import MetricsView from './components/MetricsView';
import DeploymentView from './components/DeploymentView';
import DevicesView from './components/DevicesView';
import WorkspaceView from './components/WorkspaceView';
import VisualMonitoring from './components/VisualMonitoring';
import NotificationContainer from './components/NotificationContainer';
import ConfirmModal from './components/ConfirmModal';
import { useNotification } from './hooks/useNotification';
import { useConfirm } from './hooks/useConfirm';

// Create a context to share notifications and confirm across components
export const NotificationContext = React.createContext();
export const ConfirmContext = React.createContext();

function AppContent() {
  const [connected, setConnected] = useState(false);
  const notification = useNotification();
  const confirm = useConfirm();

  useEffect(() => {
    // Check backend connectivity
    fetch(process.env.REACT_APP_API_URL || 'http://localhost:8000/health')
      .then(() => setConnected(true))
      .catch(() => setConnected(false));
  }, []);

  return (
    <NotificationContext.Provider value={notification}>
      <ConfirmContext.Provider value={confirm}>
        <div className="App">
          <nav className="navbar">
            <div className="nav-brand">
              <h1>Lydlr</h1>
              <span className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
                {connected ? '● Connected' : '○ Disconnected'}
              </span>
            </div>
            <div className="nav-links">
              <Link to="/">Dashboard</Link>
              <Link to="/devices">Devices</Link>
              <Link to="/nodes">Nodes</Link>
              <Link to="/workspace">Workspace</Link>
              <Link to="/models">Models</Link>
              <Link to="/metrics">Metrics</Link>
              <Link to="/visual">Visual</Link>
              <Link to="/deploy">Deploy</Link>
            </div>
          </nav>

          <div className="content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/nodes" element={<NodesView />} />
              <Route path="/devices" element={<DevicesView />} />
              <Route path="/workspace" element={<WorkspaceView />} />
              <Route path="/models" element={<ModelsView />} />
              <Route path="/metrics" element={<MetricsView />} />
              <Route path="/visual" element={<VisualMonitoring />} />
              <Route path="/deploy" element={<DeploymentView />} />
            </Routes>
          </div>

          <footer className="footer">
            <p>Lydlr © 2026</p>
          </footer>

          <NotificationContainer 
            notifications={notification.notifications}
            removeNotification={notification.removeNotification}
          />
          <ConfirmModal
            isOpen={confirm.confirmState.isOpen}
            title={confirm.confirmState.title}
            message={confirm.confirmState.message}
            onConfirm={confirm.confirmState.onConfirm}
            onCancel={confirm.confirmState.onCancel}
            confirmText={confirm.confirmState.confirmText}
            cancelText={confirm.confirmState.cancelText}
            type={confirm.confirmState.type}
          />
        </div>
      </ConfirmContext.Provider>
    </NotificationContext.Provider>
  );
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}

export default App;

