// MongoDB Initialization Script
db = db.getSiblingDB('lydlr_db');

// Create collections with indexes
db.createCollection('nodes');
db.nodes.createIndex({ "node_id": 1 }, { unique: true });
db.nodes.createIndex({ "status": 1 });
db.nodes.createIndex({ "last_update": -1 });

db.createCollection('models');
db.models.createIndex({ "version": 1 }, { unique: true });
db.models.createIndex({ "uploaded_at": -1 });

db.createCollection('metrics');
db.metrics.createIndex({ "node_id": 1, "timestamp": -1 });
db.metrics.createIndex({ "timestamp": -1 });
db.metrics.createIndex({ "compression_ratio": 1 });

db.createCollection('deployments');
db.deployments.createIndex({ "deployment_id": 1 });
db.deployments.createIndex({ "deployed_at": -1 });
db.deployments.createIndex({ "node_ids": 1 });

db.createCollection('devices');
db.devices.createIndex({ "device_id": 1 }, { unique: true });
db.devices.createIndex({ "node_id": 1 });
db.devices.createIndex({ "status": 1 });
db.devices.createIndex({ "device_type": 1 });
db.devices.createIndex({ "last_update": -1 });

db.createCollection('sensors');
db.sensors.createIndex({ "sensor_id": 1, "device_id": 1 }, { unique: true });
db.sensors.createIndex({ "device_id": 1 });
db.sensors.createIndex({ "sensor_type": 1 });
db.sensors.createIndex({ "status": 1 });
db.sensors.createIndex({ "last_update": -1 });

db.createCollection('system_config');
db.system_config.createIndex({ "type": 1 }, { unique: true });

// Model registry (control-plane source of truth for UI tables)
db.createCollection('model_artifacts');
db.model_artifacts.createIndex({ "artifact_id": 1 }, { unique: true });
db.model_artifacts.createIndex({ "version": 1 });
db.model_artifacts.createIndex({ "model_type": 1 });
db.model_artifacts.createIndex({ "status": 1 });
db.model_artifacts.createIndex({ "vertical_targets": 1 });

db.createCollection('node_model_assignments');
db.node_model_assignments.createIndex({ "node_id": 1 }, { unique: true });

// Hourly telemetry rollups (long retention)
db.createCollection('metrics_rollups');
db.metrics_rollups.createIndex({ "rollup_key": 1 }, { unique: true });
db.metrics_rollups.createIndex({ "node_id": 1, "bucket_start": -1 });

db.metrics.createIndex({ "vertical": 1, "timestamp": -1 });
db.metrics.createIndex({ "timestamp": 1 }, { expireAfterSeconds: 604800, name: "metrics_ttl_7d" });

// Create admin user
db.createUser({
    user: 'lydlr_admin',
    pwd: 'lydlr_admin_pass',
    roles: [
        { role: 'readWrite', db: 'lydlr_db' },
        { role: 'dbAdmin', db: 'lydlr_db' }
    ]
});

// Edge fleet seed — drones + IoT gateways (drone / IoT / Edge AI vertical)
const now = new Date();
db.nodes.insertMany([
  {
    node_id: 'node_0',
    status: 'active',
    vertical: 'drone',
    display_name: 'UAV Alpha',
    model_version: 'vv1.0',
    uplink_budget_kbps: 512,
    sensors: ['camera', 'lidar', 'imu', 'audio'],
    last_update: now,
  },
  {
    node_id: 'node_1',
    status: 'active',
    vertical: 'drone',
    display_name: 'UAV Bravo',
    model_version: 'vv1.0',
    uplink_budget_kbps: 512,
    sensors: ['camera', 'lidar', 'imu', 'audio'],
    last_update: now,
  },
  {
    node_id: 'iot_gateway_01',
    status: 'active',
    vertical: 'iot',
    display_name: 'Field Gateway 01',
    model_version: 'vv1.0',
    uplink_budget_kbps: 64,
    sensors: ['camera', 'imu', 'lidar'],
    last_update: now,
  },
]);

db.devices.insertMany([
  {
    device_id: 'uav_alpha_fc',
    device_type: 'flight_controller',
    node_id: 'node_0',
    status: 'online',
    vertical: 'drone',
    description: 'Primary UAV — long-range video + LiDAR downlink',
    last_update: now,
  },
  {
    device_id: 'uav_bravo_fc',
    device_type: 'flight_controller',
    node_id: 'node_1',
    status: 'online',
    vertical: 'drone',
    description: 'Secondary UAV — formation telemetry',
    last_update: now,
  },
  {
    device_id: 'iot_gw_01',
    device_type: 'edge_gateway',
    node_id: 'iot_gateway_01',
    status: 'online',
    vertical: 'iot',
    description: 'Solar edge gateway — LPWAN uplink',
    last_update: now,
  },
]);

db.system_config.insertOne({
  type: 'node_configuration',
  target_node_count: 3,
  vertical: 'drone',
  fleet_profile: 'drone_iot_edge',
  auto_scale: false,
  min_nodes: 2,
  max_nodes: 10,
  updated_at: now,
});

db.system_config.insertOne({
  type: 'fleet_profile',
  name: 'drone_iot_edge',
  description: 'Dual-UAV + IoT gateway — bandwidth-adaptive multimodal compression',
  verticals: ['drone', 'iot'],
  default_uplink_kbps: { drone: 512, iot: 64 },
  updated_at: now,
});

db.model_artifacts.insertMany([
  {
    artifact_id: 'multimodal_compressor_vv1.0',
    version: 'vv1.0',
    model_type: 'multimodal_compressor',
    architecture: 'EnhancedMultimodalCompressor',
    filename: 'lydlr_compressor_vv1.0.pth',
    status: 'production',
    vertical_targets: ['drone', 'iot'],
    modalities: ['camera', 'lidar', 'imu', 'audio'],
    training: { data_source: 'synthetic_multimodal', use_synthetic: true },
    performance: { compression_ratio: 12.0, quality_score: 0.88, inference_time_ms: 8.5 },
    created_at: now,
    updated_at: now,
  },
]);

db.node_model_assignments.insertMany([
  { node_id: 'node_0', model_version: 'vv1.0', artifact_id: 'multimodal_compressor_vv1.0', status: 'active', assigned_at: now },
  { node_id: 'node_1', model_version: 'vv1.0', artifact_id: 'multimodal_compressor_vv1.0', status: 'active', assigned_at: now },
  { node_id: 'iot_gateway_01', model_version: 'vv1.0', artifact_id: 'multimodal_compressor_vv1.0', status: 'active', assigned_at: now },
]);

print('✅ MongoDB initialized with drone / IoT edge fleet + model registry for Lydlr');

