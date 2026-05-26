const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

async function request(path, options = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const lydlrApi = {
  stats: () => request('/api/stats/'),
  nodes: () => request('/api/nodes/'),
  metricsTable: (params = {}) => {
    const q = new URLSearchParams({ format: 'table', ...params });
    return request(`/api/metrics/?${q}`);
  },
  metricsRollups: (params = {}) => {
    const q = new URLSearchParams(params);
    return request(`/api/metrics/rollups/?${q}`);
  },
  metricsFleet: () => request('/api/metrics/fleet/'),
  modelsRegistryTable: () => request('/api/models/registry/table/'),
  modelsSync: () => request('/api/models/sync/', { method: 'POST' }),
  modelsRegistry: (sync = true) =>
    request(`/api/models/registry/?sync=${sync}`),
  models: () => request('/api/models/'),
  deploy: (body) =>
    request('/api/deploy/', { method: 'POST', body: JSON.stringify(body) }),
  deployments: () => request('/api/deployments/'),
  rollback: (nodeIds) =>
    request('/api/deploy/rollback/', {
      method: 'POST',
      body: JSON.stringify(nodeIds?.length ? { node_ids: nodeIds } : {}),
    }),
  fleetLinkPolicy: () => request('/api/fleet/link-policy/'),
  nodeLinkSpec: (nodeId) => request(`/api/nodes/${nodeId}/link-spec/`),
  updateNodeLinkSpec: (nodeId, body) =>
    request(`/api/nodes/${nodeId}/link-spec/`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),
  metricsExport: (params = {}) => {
    const q = new URLSearchParams(params);
    return request(`/api/metrics/export/?${q}`);
  },
};

export default lydlrApi;
