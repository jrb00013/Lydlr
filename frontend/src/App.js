import React, { useState, useEffect, useMemo } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  NavLink,
  useLocation,
  Navigate,
} from 'react-router-dom';
import DashboardIcon from '@mui/icons-material/Dashboard';
import DevicesIcon from '@mui/icons-material/Devices';
import HubIcon from '@mui/icons-material/Hub';
import FolderIcon from '@mui/icons-material/Folder';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import VisibilityIcon from '@mui/icons-material/Visibility';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import GroupsIcon from '@mui/icons-material/Groups';
import MenuIcon from '@mui/icons-material/Menu';
import CloseIcon from '@mui/icons-material/Close';
import CompressIcon from '@mui/icons-material/Compress';
import './App.css';
import Dashboard from './components/Dashboard';
import NodesView from './components/NodesView';
import ModelsView from './components/ModelsView';
import MetricsView from './components/MetricsView';
import DeploymentView from './components/DeploymentView';
import FederatedView from './components/FederatedView';
import DevicesView from './components/DevicesView';
import WorkspaceView from './components/WorkspaceView';
import VisualMonitoring from './components/VisualMonitoring';
import NotFound from './components/NotFound';
import NotificationContainer from './components/NotificationContainer';
import ConfirmModal from './components/ConfirmModal';
import { useNotification } from './hooks/useNotification';
import { useConfirm } from './hooks/useConfirm';
import { apiBaseUrl } from './api/lydlrApi';

export const NotificationContext = React.createContext();
export const ConfirmContext = React.createContext();

const NAV_SECTIONS = [
  {
    label: 'Operate',
    items: [
      { to: '/', label: 'Dashboard', icon: DashboardIcon, end: true },
      { to: '/visual', label: 'Visual', icon: VisibilityIcon },
      { to: '/metrics', label: 'Metrics', icon: AnalyticsIcon },
    ],
  },
  {
    label: 'Fleet',
    items: [
      { to: '/devices', label: 'Devices', icon: DevicesIcon },
      { to: '/nodes', label: 'Nodes', icon: HubIcon },
      { to: '/deploy', label: 'Deploy', icon: RocketLaunchIcon },
    ],
  },
  {
    label: 'Models',
    items: [
      { to: '/models', label: 'Registry', icon: ModelTrainingIcon },
      { to: '/federated', label: 'Federated', icon: GroupsIcon },
      { to: '/workspace', label: 'Workspace', icon: FolderIcon },
    ],
  },
];

const TITLE_BY_PATH = {
  '/': 'Dashboard',
  '/devices': 'Devices',
  '/nodes': 'Nodes',
  '/workspace': 'Workspace',
  '/models': 'Models',
  '/metrics': 'Metrics',
  '/visual': 'Visual',
  '/deploy': 'Deploy',
  '/federated': 'Federated',
};

function pageTitle(pathname) {
  if (TITLE_BY_PATH[pathname]) return TITLE_BY_PATH[pathname];
  const hit = Object.keys(TITLE_BY_PATH).find(
    (p) => p !== '/' && pathname.startsWith(p)
  );
  return hit ? TITLE_BY_PATH[hit] : 'Console';
}

function AppShell() {
  const [connected, setConnected] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const notification = useNotification();
  const confirm = useConfirm();
  const location = useLocation();
  const title = useMemo(() => pageTitle(location.pathname), [location.pathname]);

  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    const base = apiBaseUrl();
    const checkHealth = () => {
      fetch(`${base}/health`)
        .then((res) => setConnected(res.ok))
        .catch(() => setConnected(false));
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    document.title = `${title} · Lydlr`;
  }, [title]);

  return (
    <NotificationContext.Provider value={notification}>
      <ConfirmContext.Provider value={confirm}>
        <div className="app-layout">
          <a href="#main-content" className="skip-link">
            Skip to content
          </a>

          <aside
            className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}
            aria-label="Primary"
          >
            <div className="sidebar__brand">
              <div className="sidebar__logo" aria-hidden="true">
                <CompressIcon />
              </div>
              <div className="sidebar__brand-text">
                <span className="sidebar__name">Lydlr</span>
                <span className="sidebar__tagline">Edge compression</span>
              </div>
            </div>

            <nav className="sidebar__nav">
              {NAV_SECTIONS.map((section) => (
                <div key={section.label} className="sidebar__section">
                  <p className="sidebar__section-label">{section.label}</p>
                  {section.items.map(({ to, label, icon: Icon, end }) => (
                    <NavLink
                      key={to}
                      to={to}
                      end={end}
                      className={({ isActive }) =>
                        `sidebar__link ${isActive ? 'sidebar__link--active' : ''}`
                      }
                    >
                      <Icon className="sidebar__link-icon" fontSize="small" />
                      <span>{label}</span>
                    </NavLink>
                  ))}
                </div>
              ))}
            </nav>

            <div className="sidebar__footer">
              <div
                className={`connection-pill ${
                  connected ? 'connection-pill--on' : 'connection-pill--off'
                }`}
              >
                <span className="connection-pill__dot" aria-hidden="true" />
                <span>{connected ? 'Control plane live' : 'API offline'}</span>
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
                aria-expanded={sidebarOpen}
              >
                {sidebarOpen ? <CloseIcon /> : <MenuIcon />}
              </button>
              <div className="topbar__title-block">
                <h1 className="topbar__title">{title}</h1>
                <p className="topbar__crumb">Lydlr · Drone &amp; IoT</p>
              </div>
              <div className="topbar__status">
                <span
                  className={`topbar__badge ${connected ? 'topbar__badge--live' : ''}`}
                >
                  {connected ? 'Live' : 'Offline'}
                </span>
              </div>
            </header>

            <main id="main-content" className="content page-enter" tabIndex={-1}>
              <Routes>
                <Route path="/" element={<Dashboard connected={connected} />} />
                <Route path="/nodes" element={<NodesView />} />
                <Route path="/devices" element={<DevicesView />} />
                <Route path="/workspace" element={<WorkspaceView />} />
                <Route path="/models" element={<ModelsView />} />
                <Route path="/metrics" element={<MetricsView />} />
                <Route path="/visual" element={<VisualMonitoring />} />
                <Route path="/deploy" element={<DeploymentView />} />
                <Route path="/federated" element={<FederatedView />} />
                <Route path="/home" element={<Navigate to="/" replace />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </main>

            <footer className="app-footer">
              <span>Lydlr © 2026</span>
              <span className="app-footer__sep" aria-hidden="true">
                ·
              </span>
              <span>Countable bits · matched-rate edge compression</span>
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
