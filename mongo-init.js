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

// Create admin user
db.createUser({
    user: 'lydlr_admin',
    pwd: 'lydlr_admin_pass',
    roles: [
        { role: 'readWrite', db: 'lydlr_db' },
        { role: 'dbAdmin', db: 'lydlr_db' }
    ]
});

print('✅ MongoDB initialized successfully for Lydlr');

