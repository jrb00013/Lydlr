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
  deploy: (body) =>
    request('/api/deploy/', { method: 'POST', body: JSON.stringify(body) }),
};

export default lydlrApi;
