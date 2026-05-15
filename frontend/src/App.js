import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import DashboardIcon from '@mui/icons-material/Dashboard';
import DevicesIcon from '@mui/icons-material/Devices';
import HubIcon from '@mui/icons-material/Hub';
import FolderIcon from '@mui/icons-material/Folder';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import VisibilityIcon from '@mui/icons-material/Visibility';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';
import CompressIcon from '@mui/icons-material/Compress';
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

export const NotificationContext = React.createContext();
export const ConfirmContext = React.createContext();

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: DashboardIcon, end: true },
  { to: '/devices', label: 'Devices', icon: DevicesIcon },
  { to: '/nodes', label: 'Nodes', icon: HubIcon },
  { to: '/workspace', label: 'Workspace', icon: FolderIcon },
  { to: '/models', label: 'Models', icon: ModelTrainingIcon },
  { to: '/metrics', label: 'Metrics', icon: AnalyticsIcon },
  { to: '/visual', label: 'Visual', icon: VisibilityIcon },
  { to: '/deploy', label: 'Deploy', icon: RocketLaunchIcon },
];

function AppShell() {
  const [connected, setConnected] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const notification = useNotification();
  const confirm = useConfirm();
  const location = useLocation();

  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    const checkHealth = () => {
      fetch(process.env.REACT_APP_API_URL || 'http://localhost:8000/health')
        .then((res) => setConnected(res.ok))
        .catch(() => setConnected(false));
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <NotificationContext.Provider value={notification}>
      <ConfirmContext.Provider value={confirm}>
        <div className="app-layout">
          <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
            <div className="sidebar__brand">
              <div className="sidebar__logo">
                <CompressIcon />
              </div>
              <div className="sidebar__brand-text">
                <span className="sidebar__name">Lydlr</span>
                <span className="sidebar__tagline">Drone · IoT · Edge AI</span>
              </div>
            </div>

            <nav className="sidebar__nav">
              {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={end}
                  className={({ isActive }) =>
                    `sidebar__link ${isActive ? 'sidebar__link--active' : ''}`
                  }
                >
                  <Icon className="sidebar__link-icon" />
                  <span>{label}</span>
                </NavLink>
              ))}
            </nav>

            <div className="sidebar__footer">
              <div className={`connection-pill ${connected ? 'connection-pill--on' : 'connection-pill--off'}`}>
                <span className="connection-pill__dot" />
                {connected ? 'Control plane live' : 'API offline'}
              </div>
            </div>
          </aside>

          {sidebarOpen && (
            <button
              type="button"
              className="sidebar-backdrop"
              onClick={() => setSidebarOpen(false)}
              aria-label="Close menu"
            />
          )}

          <div className="app-main">
            <header className="topbar">
              <button
                type="button"
                className="topbar__menu-btn"
                onClick={() => setSidebarOpen((o) => !o)}
                aria-label={sidebarOpen ? 'Close menu' : 'Open menu'}
              >
                {sidebarOpen ? <CloseIcon /> : <MenuIcon />}
              </button>
              <div className="topbar__status">
                <span className={`topbar__badge ${connected ? 'topbar__badge--live' : ''}`}>
                  {connected ? 'Live' : 'Offline'}
                </span>
              </div>
            </header>

            <main className="content page-enter">
              <Routes>
                <Route path="/" element={<Dashboard connected={connected} />} />
                <Route path="/nodes" element={<NodesView />} />
                <Route path="/devices" element={<DevicesView />} />
                <Route path="/workspace" element={<WorkspaceView />} />
                <Route path="/models" element={<ModelsView />} />
                <Route path="/metrics" element={<MetricsView />} />
                <Route path="/visual" element={<VisualMonitoring />} />
                <Route path="/deploy" element={<DeploymentView />} />
              </Routes>
            </main>

            <footer className="app-footer">
              <span>Lydlr © 2026</span>
              <span className="app-footer__sep">·</span>
              <span>Drone & IoT edge compression</span>
            </footer>
          </div>

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
      <AppShell />
    </Router>
  );
}

export default App;
