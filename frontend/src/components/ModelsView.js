import React, { useState, useEffect, useContext, useCallback } from 'react';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import SyncIcon from '@mui/icons-material/Sync';
import UploadIcon from '@mui/icons-material/Upload';
import UndoIcon from '@mui/icons-material/Undo';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import './ModelsView.css';
import { NotificationContext } from '../App';
import PageHeader from './ui/PageHeader';
import LoadingSpinner from './ui/LoadingSpinner';
import DataTable from './ui/DataTable';
import lydlrApi from '../api/lydlrApi';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function ModelsView() {
  const notification = useContext(NotificationContext);
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [rollingBack, setRollingBack] = useState(false);
  const [deploying, setDeploying] = useState(false);
  const [selected, setSelected] = useState(null);

  const modelColumns = [
    {
      key: 'version',
      label: 'Version',
      render: (r) => (
        <button type="button" className="link-btn" onClick={() => setSelected(r)}>
          <strong>{r.version}</strong>
        </button>
      ),
    },
    {
      key: 'model_type',
      label: 'Type',
      render: (r) => (
        <span className="badge badge--multimodal">{r.model_type?.replace(/_/g, ' ')}</span>
      ),
    },
    { key: 'architecture', label: 'Architecture' },
    {
      key: 'status',
      label: 'Status',
      render: (r) => (
        <span className={`badge badge--${r.status === 'production' ? 'production' : 'registered'}`}>
          {r.status}
        </span>
      ),
    },
    {
      key: 'size_mb',
      label: 'Size (MB)',
      render: (r) => (r.size_mb != null ? r.size_mb.toFixed(2) : '—'),
    },
    {
      key: 'vertical_targets',
      label: 'Verticals',
      render: (r) =>
        (r.vertical_targets || []).map((v) => (
          <span key={v} className={`badge badge--${v}`} style={{ marginRight: 4 }}>
            {v}
          </span>
        )),
    },
    {
      key: 'compression_ratio',
      label: 'Compress',
      render: (r) => (r.compression_ratio != null ? `${r.compression_ratio.toFixed(1)}×` : '—'),
    },
    {
      key: 'quality_score',
      label: 'Quality',
      render: (r) =>
        r.quality_score != null ? `${(r.quality_score * 100).toFixed(0)}%` : '—',
    },
    {
      key: 'inference_ms',
      label: 'Infer (ms)',
      render: (r) => (r.inference_ms != null ? r.inference_ms.toFixed(1) : '—'),
    },
    {
      key: 'deployed_node_count',
      label: 'Deployed',
      render: (r) =>
        r.deployed_node_count > 0 ? (
          <span title={(r.deployed_nodes || []).join(', ')}>{r.deployed_node_count} nodes</span>
        ) : (
          '—'
        ),
    },
    { key: 'filename', label: 'File' },
    { key: 'updated_at', label: 'Updated' },
  ];

  const loadTable = useCallback(async () => {
    try {
      const data = await lydlrApi.modelsRegistryTable();
      setRows(data.rows || []);
    } catch (e) {
      console.error(e);
      notification?.showError?.('Failed to load model registry');
    } finally {
      setLoading(false);
    }
  }, [notification]);

  useEffect(() => {
    loadTable();
  }, [loadTable]);

  const handleSync = async () => {
    setSyncing(true);
    try {
      await lydlrApi.modelsSync();
      notification?.showSuccess?.('Registry synced from disk');
      await loadTable();
    } catch (e) {
      notification?.showError?.(e.message || 'Sync failed');
    } finally {
      setSyncing(false);
    }
  };

  const handleDeploySelected = async () => {
    if (!selected?.version) return;
    const nodeIds = selected.deployed_nodes?.length
      ? selected.deployed_nodes
      : (await lydlrApi.nodes()).map((n) => n.node_id);
    if (!nodeIds.length) {
      notification?.showWarning?.('No fleet nodes available');
      return;
    }
    setDeploying(true);
    try {
      const result = await lydlrApi.deploy({
        model_version: selected.version,
        node_ids: nodeIds,
      });
      notification?.showSuccess?.(
        `Deployed ${selected.version} to ${result.successful_nodes?.length || nodeIds.length} node(s)`
      );
      await loadTable();
    } catch (e) {
      notification?.showError?.(e.message || 'Deploy failed');
    } finally {
      setDeploying(false);
    }
  };

  const handleRollback = async () => {
    const nodeIds = selected?.deployed_nodes || [];
    setRollingBack(true);
    try {
      const result = await lydlrApi.rollback(nodeIds.length ? nodeIds : undefined);
      const count = result.rolled_back?.length || 0;
      notification?.showSuccess?.(`Rolled back ${count} node(s)`);
      await loadTable();
    } catch (e) {
      notification?.showError?.(e.message || 'Rollback failed — need a previous deployment');
    } finally {
      setRollingBack(false);
    }
  };

  const handleUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch(`${API_URL}/api/models/upload/`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Upload failed');
      notification?.showSuccess?.('Model uploaded');
      await lydlrApi.modelsSync();
      await loadTable();
    } catch (e) {
      notification?.showError?.('Upload failed');
    } finally {
      setUploading(false);
    }
  };

  if (loading) {
    return <LoadingSpinner message="Loading model registry…" />;
  }

  return (
    <div className="models-view page-enter">
      <PageHeader
        title="Model registry"
        subtitle="PyTorch artifacts, training metadata, and fleet assignments — synced from disk into MongoDB"
        icon={ModelTrainingIcon}
        actions={
          <>
            <button type="button" className="btn btn-secondary" onClick={handleSync} disabled={syncing}>
              <SyncIcon fontSize="small" /> {syncing ? 'Syncing…' : 'Sync disk'}
            </button>
            <label htmlFor="model-upload" className="btn btn-primary">
              <UploadIcon fontSize="small" /> {uploading ? 'Uploading…' : 'Upload .pth'}
            </label>
            <input
              id="model-upload"
              type="file"
              accept=".pth"
              hidden
              disabled={uploading}
              onChange={handleUpload}
            />
          </>
        }
      />

      <div className="card models-table-card">
        <DataTable
          columns={modelColumns}
          rows={rows}
          keyField="artifact_id"
          emptyMessage="No artifacts — train models or run Sync disk"
          pageSize={12}
        />
      </div>

      {selected && (
        <div className="card model-detail-panel">
          <h3>{selected.artifact_id}</h3>
          <div className="model-detail-grid">
            <div>
              <h4>Architecture</h4>
              <p>{selected.architecture}</p>
            </div>
            <div>
              <h4>Training</h4>
              <p>Source: {selected.training_source || 'synthetic'}</p>
            </div>
            <div>
              <h4>Deployed nodes</h4>
              <p>{(selected.deployed_nodes || []).join(', ') || 'none'}</p>
            </div>
          </div>
          <div className="model-detail-actions">
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleDeploySelected}
              disabled={deploying}
            >
              <RocketLaunchIcon fontSize="small" /> {deploying ? 'Deploying…' : 'Deploy to fleet'}
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleRollback}
              disabled={rollingBack || !(selected.deployed_nodes || []).length}
            >
              <UndoIcon fontSize="small" /> {rollingBack ? 'Rolling back…' : 'Rollback nodes'}
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => setSelected(null)}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default ModelsView;
