"""
MongoDB collection registry and index specifications.

Architecture
------------
Control plane (MongoDB) stores fleet state, model registry, telemetry, deployments.
Data plane (ROS2 + .pth on disk) executes compression; artifacts sync into model_artifacts.

Collections
-----------
nodes                 Edge compressor instances (UAV / IoT)
devices               Physical hardware attached to nodes
sensors               Sensor channels on devices
model_artifacts       Registered ML weights + metadata (source of truth for UI tables)
node_model_assignments Active model version per node
metrics               Raw time-series samples (TTL 7d)
metrics_rollups       Hourly aggregates per node (long retention)
deployments           Deploy audit trail
federated_rounds      Federated FedAvg round state + merged artifacts
system_config         Fleet / orchestration settings
"""

COLLECTIONS = {
    "NODES": "nodes",
    "DEVICES": "devices",
    "SENSORS": "sensors",
    "MODELS": "models",  # legacy alias; prefer model_artifacts
    "MODEL_ARTIFACTS": "model_artifacts",
    "NODE_MODEL_ASSIGNMENTS": "node_model_assignments",
    "METRICS": "metrics",
    "METRICS_ROLLUPS": "metrics_rollups",
    "DEPLOYMENTS": "deployments",
    "FEDERATED_ROUNDS": "federated_rounds",
    "SYSTEM_CONFIG": "system_config",
}

# Indexes applied at startup via ensure_indexes()
INDEX_SPECS = [
    (COLLECTIONS["NODES"], [("node_id", 1)], {"unique": True}),
    (COLLECTIONS["NODES"], [("status", 1)]),
    (COLLECTIONS["NODES"], [("vertical", 1)]),
    (COLLECTIONS["NODES"], [("last_update", -1)]),
    (COLLECTIONS["DEVICES"], [("device_id", 1)], {"unique": True}),
    (COLLECTIONS["DEVICES"], [("node_id", 1)]),
    (COLLECTIONS["DEVICES"], [("vertical", 1)]),
    (COLLECTIONS["SENSORS"], [("sensor_id", 1), ("device_id", 1)], {"unique": True}),
    (COLLECTIONS["SENSORS"], [("device_id", 1)]),
    (COLLECTIONS["MODEL_ARTIFACTS"], [("artifact_id", 1)], {"unique": True}),
    (COLLECTIONS["MODEL_ARTIFACTS"], [("version", 1)]),
    (COLLECTIONS["MODEL_ARTIFACTS"], [("model_type", 1)]),
    (COLLECTIONS["MODEL_ARTIFACTS"], [("status", 1)]),
    (COLLECTIONS["MODEL_ARTIFACTS"], [("vertical_targets", 1)]),
    (COLLECTIONS["NODE_MODEL_ASSIGNMENTS"], [("node_id", 1)], {"unique": True}),
    (COLLECTIONS["METRICS"], [("node_id", 1), ("timestamp", -1)]),
    (COLLECTIONS["METRICS"], [("timestamp", -1)]),
    (COLLECTIONS["METRICS"], [("vertical", 1), ("timestamp", -1)]),
    (COLLECTIONS["METRICS_ROLLUPS"], [("rollup_key", 1)], {"unique": True}),
    (COLLECTIONS["METRICS_ROLLUPS"], [("node_id", 1), ("bucket_start", -1)]),
    (COLLECTIONS["DEPLOYMENTS"], [("deployment_id", 1)]),
    (COLLECTIONS["DEPLOYMENTS"], [("deployed_at", -1)]),
    (COLLECTIONS["DEPLOYMENTS"], [("node_ids", 1)]),
    (COLLECTIONS["FEDERATED_ROUNDS"], [("round_id", 1)], {"unique": True}),
    (COLLECTIONS["FEDERATED_ROUNDS"], [("created_at", -1)]),
    (COLLECTIONS["FEDERATED_ROUNDS"], [("status", 1)]),
    (COLLECTIONS["SYSTEM_CONFIG"], [("type", 1)], {"unique": True}),
    # Legacy models collection
    (COLLECTIONS["MODELS"], [("version", 1)], {"unique": True}),
]

# Raw metrics TTL: 7 days
METRICS_TTL_SECONDS = 7 * 24 * 3600
