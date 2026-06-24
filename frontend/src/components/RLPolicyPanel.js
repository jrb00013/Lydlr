import React, { useState, useCallback } from 'react';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import PsychologyIcon from '@mui/icons-material/Psychology';
import './RLPolicyPanel.css';
import { useSmartPolling } from '../hooks/useSmartPolling';
import lydlrApi from '../api/lydlrApi';

function RLPolicyPanel() {
  const [rlMode, setRlMode] = useState('heuristic');
  const [rlStatus, setRlStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchStatus = useCallback(async () => {
    try {
      const data = await lydlrApi.rlPolicyStatus();
      setRlStatus(data);
      if (data.mode) setRlMode(data.mode);
    } catch (e) {
      console.warn('RL policy status fetch failed:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useSmartPolling(fetchStatus, {
    interval: 10000,
    enabled: true,
    immediate: true,
    minInterval: 8000,
    maxInterval: 20000,
  });

  const handleModeChange = async (_, newMode) => {
    if (!newMode) return;
    try {
      await lydlrApi.rlPolicySetMode(newMode);
      setRlMode(newMode);
    } catch (e) {
      console.warn('RL mode set failed:', e);
    }
  };

  return (
    <div className="card rl-policy-panel">
      <h2><PsychologyIcon /> RL compression policy</h2>

      <div className="rl-mode-toggle">
        <span className="rl-label">Controller mode:</span>
        <ToggleButtonGroup
          value={rlMode}
          exclusive
          onChange={handleModeChange}
          size="small"
          color="primary"
        >
          <ToggleButton value="heuristic">Heuristic</ToggleButton>
          <ToggleButton value="ppo">PPO</ToggleButton>
        </ToggleButtonGroup>
      </div>

      {rlStatus && (
        <div className="rl-metrics">
          <div className="rl-metric">
            <span className="rl-metric-label">Action</span>
            <span className="rl-metric-value">
              {rlStatus.rl_action != null ? rlStatus.rl_action.toFixed(4) : '—'}
            </span>
          </div>
          <div className="rl-metric">
            <span className="rl-metric-label">Reward</span>
            <span className="rl-metric-value">
              {rlStatus.rl_reward != null ? rlStatus.rl_reward.toFixed(2) : '—'}
            </span>
          </div>
          <div className="rl-metric">
            <span className="rl-metric-label">Steps</span>
            <span className="rl-metric-value">{rlStatus.step || 0}</span>
          </div>
        </div>
      )}

      {loading && !rlStatus && (
        <p className="rl-empty">Loading RL policy status…</p>
      )}
    </div>
  );
}

export default RLPolicyPanel;
