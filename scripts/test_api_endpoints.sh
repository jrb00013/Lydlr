#!/usr/bin/env bash
# Smoke-test Lydlr API endpoints used by the frontend.
set -euo pipefail

API="${API_URL:-http://localhost:8000}"
PASS=0
FAIL=0

check() {
  local name="$1"
  local method="$2"
  local path="$3"
  local data="${4:-}"
  local expect="${5:-200}"

  local code
  if [ "$method" = "POST" ] && [ -n "$data" ]; then
    code=$(curl -s -o /tmp/lydlr_test_body.json -w '%{http_code}' \
      -X POST -H 'Content-Type: application/json' -d "$data" "${API}${path}")
  elif [ "$method" = "POST" ]; then
    code=$(curl -s -o /tmp/lydlr_test_body.json -w '%{http_code}' -X POST "${API}${path}")
  else
    code=$(curl -s -o /tmp/lydlr_test_body.json -w '%{http_code}' "${API}${path}")
  fi

  if [ "$code" = "$expect" ] || { [ "$expect" = "2xx" ] && [ "${code:0:1}" = "2" ]; }; then
    echo "  OK  [$code] $method $path"
    PASS=$((PASS + 1))
  else
    echo "  FAIL [$code] $method $path (expected $expect)"
    head -c 200 /tmp/lydlr_test_body.json 2>/dev/null; echo
    FAIL=$((FAIL + 1))
  fi
}

echo "=== Lydlr API smoke test ==="
echo "API: $API"
echo ""

check "health" GET /health/
check "stats" GET /api/stats/
check "nodes" GET /api/nodes/
check "devices" GET /api/devices/
check "models disk" GET /api/models/
check "models registry table" GET /api/models/registry/table/
check "metrics" GET '/api/metrics/?limit=5'
check "metrics table" GET '/api/metrics/?format=table&limit=5'
check "metrics rollups" GET /api/metrics/rollups/
check "metrics fleet" GET /api/metrics/fleet/
check "deployments" GET /api/deployments/
check "workspace" GET /api/workspace/
check "orchestration" GET /api/orchestration/status/
check "models sync" POST /api/models/sync/

# POST metrics sample
check "metrics ingest" POST /api/metrics/ \
  '{"node_id":"node_0","compression_ratio":8.5,"latency_ms":12.3,"quality_score":0.88,"compression_level":0.8,"bandwidth_estimate":256.0,"vertical":"drone"}' \
  200

# Deploy + rollback (needs nodes + model on disk)
NODE_IDS=$(curl -s "${API}/api/nodes/" | python3 -c "import sys,json; d=json.load(sys.stdin); print(','.join(repr(n['node_id']) for n in d[:1]))" 2>/dev/null || echo "")
MODEL_VER=$(curl -s "${API}/api/models/" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0]['version'] if d else '')" 2>/dev/null || echo "")

if [ -n "$MODEL_VER" ] && [ -n "$NODE_IDS" ]; then
  NODE_JSON=$(curl -s "${API}/api/nodes/" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps([d[0]['node_id']]))")
  check "deploy" POST /api/deploy/ "{\"model_version\":\"$MODEL_VER\",\"node_ids\":$NODE_JSON}" 2xx
  check "rollback" POST /api/deploy/rollback/ "{\"node_ids\":$NODE_JSON}" 2xx
else
  echo "  SKIP deploy/rollback (no model or nodes)"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
