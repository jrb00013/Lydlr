import React, { useState, useContext, useCallback } from 'react';
import GroupsIcon from '@mui/icons-material/Groups';
import DownloadIcon from '@mui/icons-material/Download';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import './FederatedView.css';
import { NotificationContext } from '../App';
import PageHeader from './ui/PageHeader';
import LoadingSpinner from './ui/LoadingSpinner';
import { useSmartPolling } from '../hooks/useSmartPolling';
import lydlrApi from '../api/lydlrApi';

const STATUS_CLASS = {
  pending: 'fed-status--pending',
  aggregating: 'fed-status--active',
  merged: 'fed-status--merged',
  failed: 'fed-status--failed',
};

function participantRows(round) {
  if (round.participants?.length) {
    return round.participants;
  }
  const statusMap = round.participant_status || {};
  return Object.entries(statusMap).map(([node_id, info]) => ({
    node_id,
    status: info.status || 'pending',
    modality_bytes_out: info.modality_bytes_out || 0,
    delta_sha256: info.checksum_sha256 || '',
  }));
}

function FederatedView() {
  const notification = useContext(NotificationContext);
  const [rounds, setRounds] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [baseVersion, setBaseVersion] = useState('');
  const [maxDeltaKbps, setMaxDeltaKbps] = useState(128);
  const [inferenceBackend, setInferenceBackend] = useState('onnx');
  const [expandedRound, setExpandedRound] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const [roundData, nodeData, modelData] = await Promise.all([
        lydlrApi.federatedRounds(),
        lydlrApi.nodes(),
        lydlrApi.models(),
      ]);
      setRounds(roundData);
      setNodes(nodeData);
      setModels(modelData);
      if (!baseVersion && modelData.length) {
        setBaseVersion(modelData[0].version || '');
      }
    } catch (error) {
      console.error('Failed to load federated data:', error);
    } finally {
      setLoading(false);
    }
  }, [baseVersion]);

  useSmartPolling(refresh, {
    interval: 12000,
    enabled: true,
    immediate: true,
    minInterval: 8000,
    maxInterval: 20000,
  });

  const toggleNode = (nodeId) => {
    setSelectedNodes((prev) =>
      prev.includes(nodeId) ? prev.filter((id) => id !== nodeId) : [...prev, nodeId]
    );
  };

  const handleStartRound = async () => {
    if (!baseVersion || selectedNodes.length < 2) {
      notification.showWarning('Pick a base model and at least two nodes');
      return;
    }
    setStarting(true);
    try {
      const round = await lydlrApi.federatedStartRound({
        participant_node_ids: selectedNodes,
        base_version: baseVersion,
        max_delta_kbps: maxDeltaKbps,
        inference_backend: inferenceBackend,
      });
      notification.showSuccess(`Started round ${round.round_id}`);
      setSelectedNodes([]);
      await refresh();
      setExpandedRound(round.round_id);
    } catch (error) {
      notification.showError(error.message || 'Failed to start round');
    } finally {
      setStarting(false);
    }
  };

  const handleExport = async () => {
    try {
      const blob = await lydlrApi.federatedRoundsExport();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'federated_rounds.csv';
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      notification.showError(error.message || 'Export failed');
    }
  };

  const handleDeployMerged = async (round) => {
    const version = round.merged_version;
    if (!version) {
      notification.showWarning('No merged version yet');
      return;
    }
    try {
      await lydlrApi.deploy({
        model_version: version,
        node_ids: round.participant_node_ids,
        strategy: 'fleet',
        inference_backend: round.inference_backend || 'onnx',
      });
      notification.showSuccess(`Deployed merged ${version} to fleet`);
    } catch (error) {
      notification.showError(error.message || 'Deploy failed');
    }
  };

  if (loading) {
    return <LoadingSpinner message="Loading federated rounds…" />;
  }

  return (
    <div className="federated-view page-enter">
      <PageHeader
        title="Federated learning"
        subtitle="Fleet FedAvg rounds — delta upload, merge, deploy via control plane"
        icon={GroupsIcon}
      />

      <div className="federated-grid">
        <div className="card fed-start-card">
          <h2>Start round</h2>
          <div className="form-group">
            <label htmlFor="fed-base">Base model version</label>
            <select
              id="fed-base"
              value={baseVersion}
              onChange={(e) => setBaseVersion(e.target.value)}
            >
              <option value="">Select model…</option>
              {models.map((m) => (
                <option key={m.version} value={m.version}>
                  {m.version}
                </option>
              ))}
            </select>
          </div>
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="fed-delta">Max delta (kbps)</label>
              <input
                id="fed-delta"
                type="number"
                min={32}
                max={2048}
                value={maxDeltaKbps}
                onChange={(e) => setMaxDeltaKbps(Number(e.target.value))}
              />
            </div>
            <div className="form-group">
              <label htmlFor="fed-backend">Inference backend</label>
              <select
                id="fed-backend"
                value={inferenceBackend}
                onChange={(e) => setInferenceBackend(e.target.value)}
              >
                <option value="torch">PyTorch</option>
                <option value="onnx">ONNX</option>
                <option value="trt">TensorRT</option>
              </select>
            </div>
          </div>
          <div className="form-group">
            <span className="fed-label">Participants</span>
            <div className="fed-node-picks">
              {nodes.map((n) => (
                <label key={n.node_id} className="fed-node-chip">
                  <input
                    type="checkbox"
                    checked={selectedNodes.includes(n.node_id)}
                    onChange={() => toggleNode(n.node_id)}
                  />
                  {n.display_name || n.node_id}
                </label>
              ))}
            </div>
          </div>
          <button
            type="button"
            className="btn btn-primary"
            disabled={starting}
            onClick={handleStartRound}
          >
            {starting ? 'Starting…' : 'Start FedAvg round'}
          </button>
        </div>

        <div className="card fed-rounds-card">
          <div className="fed-rounds-header">
            <h2>Round timeline</h2>
            <button type="button" className="btn btn-secondary btn-sm" onClick={handleExport}>
              <DownloadIcon fontSize="small" /> Export CSV
            </button>
          </div>
          {rounds.length === 0 ? (
            <p className="fed-empty">No rounds yet — start one to coordinate fleet learning.</p>
          ) : (
            <div className="fed-timeline">
              {rounds.map((round) => {
                const isOpen = expandedRound === round.round_id;
                const statusClass = STATUS_CLASS[round.status] || 'fed-status--pending';
                return (
                  <div key={round.round_id} className={`fed-round ${isOpen ? 'fed-round--open' : ''}`}>
                    <button
                      type="button"
                      className="fed-round-summary"
                      onClick={() =>
                        setExpandedRound(isOpen ? null : round.round_id)
                      }
                    >
                      <span className={`fed-status ${statusClass}`}>{round.status}</span>
                      <span className="fed-round-id">{round.round_id}</span>
                      <span className="fed-round-meta">
                        {round.base_version}
                        {round.merged_version ? ` → ${round.merged_version}` : ''}
                      </span>
                      <span className="fed-round-time">
                        {(round.created_at || round.started_at || '').toString().slice(0, 19)}
                      </span>
                    </button>
                    {isOpen && (
                      <div className="fed-round-detail">
                        <div className="fed-detail-row">
                          <span>Max delta</span>
                          <strong>{round.max_delta_kbps} kbps</strong>
                        </div>
                        <div className="fed-detail-row">
                          <span>Modality bytes out</span>
                          <strong>{round.modality_bytes_out_total ?? 0}</strong>
                        </div>
                        <div className="fed-participants">
                          {participantRows(round).map((p) => (
                            <div key={p.node_id} className="fed-participant">
                              <span>{p.node_id}</span>
                              <span className={`fed-status ${STATUS_CLASS[p.status] || ''}`}>
                                {p.status}
                              </span>
                              <span className="fed-bytes">{p.modality_bytes_out || 0} B</span>
                            </div>
                          ))}
                        </div>
                        {round.merged_version && (
                          <button
                            type="button"
                            className="btn btn-primary btn-sm fed-deploy-btn"
                            onClick={() => handleDeployMerged(round)}
                          >
                            <RocketLaunchIcon fontSize="small" />
                            Deploy {round.merged_version}
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default FederatedView;
