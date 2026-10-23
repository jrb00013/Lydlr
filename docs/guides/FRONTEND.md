# Frontend console notes

The edge console lives under `frontend/` (Create React App + MUI + Recharts).

## Key views

| Route | Component | Role |
|-------|-----------|------|
| `/` | Dashboard | Fleet stats, link budget, RL policy, compression waterfall |
| `/visual` | VisualMonitoring | Live previews + waterfall + modality series |
| `/metrics` | MetricsView | Historical charts |
| `/devices` `/nodes` | Devices / Nodes | Fleet inventory |
| `/deploy` `/models` `/federated` | Deploy / Registry / Federated | Model ops |

## Telemetry helpers

- `utils/formatBytes.js` — KB/MB formatting for waterfall tooltips
- `hooks/useLatestMetric.js` — pick newest metric from a node map
- `hooks/useMetricsWebSocket.js` / `useDemoPulse.js` — live vs demo fill

## Dev

```bash
cd frontend && npm install && npm start
# API base from REACT_APP_API_URL (see .env.example)
```
