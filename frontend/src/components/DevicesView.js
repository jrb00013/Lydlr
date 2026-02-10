import React, { useState, useEffect, useContext } from 'react';
import './DevicesView.css';
import { NotificationContext, ConfirmContext } from '../App';
import { useSmartPolling } from '../hooks/useSmartPolling';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Device type labels mapping for ROS2 hardware devices
const DEVICE_TYPE_CONFIG = {
  camera: { label: 'Camera' },
  stereo_camera: { label: 'Stereo Camera' },
  lidar: { label: 'LiDAR' },
  imu: { label: 'IMU' },
  gps: { label: 'GPS / GNSS' },
  motor_controller: { label: 'Motor Controller' },
  servo_motor: { label: 'Servo Motor' },
  stepper_motor: { label: 'Stepper Motor' },
  encoder: { label: 'Encoder' },
  ultrasonic_sensor: { label: 'Ultrasonic Sensor' },
  infrared_sensor: { label: 'Infrared Sensor' },
  force_torque_sensor: { label: 'Force/Torque Sensor' },
  actuator: { label: 'Actuator' },
  power_supply: { label: 'Power Supply' },
  edge_computer: { label: 'Edge Computer' },
  gateway: { label: 'Gateway' },
  network_switch: { label: 'Network Switch' },
  radar: { label: 'Radar' },
  depth_camera: { label: 'Depth Camera' },
  thermal_camera: { label: 'Thermal Camera' },
  other: { label: 'Other' }
};

const getDeviceTypeLabel = (deviceType) => {
  const config = DEVICE_TYPE_CONFIG[deviceType] || DEVICE_TYPE_CONFIG.other;
  return config.label;
};

const getTopicDescription = (topic) => {
  const topicLower = topic.toLowerCase();
  const descriptions = {
    "status": "Device status updates (active/inactive/error)",
    "metadata": "Device metadata and configuration",
    "heartbeat": "Periodic device heartbeat signal",
    "location": "Device location updates",
    "image_raw": "Raw image data from camera",
    "image_compressed": "Compressed image data",
    "camera_info": "Camera calibration and info",
    "points": "Point cloud data from LiDAR",
    "imu/data": "IMU sensor data (accel, gyro, mag)",
    "gps/fix": "GPS position fix data",
    "range": "Range/distance measurements",
    "wrench": "Force and torque measurements",
    "position": "Position data",
    "velocity": "Velocity data",
    "current": "Current measurements",
    "voltage": "Voltage measurements",
    "power": "Power consumption",
    "temperature": "Temperature readings",
    "cmd/enable": "Enable device command",
    "cmd/disable": "Disable device command",
    "cmd/reset": "Reset device command",
    "cmd/configure": "Configure device command",
    "cmd/velocity": "Set velocity command",
    "cmd/position": "Set position command",
    "diagnostics": "Device diagnostics information",
    "telemetry": "Device telemetry data",
    "errors": "Device error messages",
    "warnings": "Device warning messages"
  };
  
  for (const [key, desc] of Object.entries(descriptions)) {
    if (topicLower.includes(key)) {
      return desc;
    }
  }
  
  return "Device communication topic";
};

function DevicesView() {
  const notification = useContext(NotificationContext);
  const confirm = useContext(ConfirmContext);
  const [devices, setDevices] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddDevice, setShowAddDevice] = useState(false);
  const [showAddSensor, setShowAddSensor] = useState(false);
  const [showConnectionModal, setShowConnectionModal] = useState(false);
  const [selectedDevice, setSelectedDevice] = useState(null);
  const [connectionData, setConnectionData] = useState({
    device_id: '',
    node_id: '',
    pub_topics: [],
    sub_topics: []
  });
  const [pubTopicInput, setPubTopicInput] = useState('');
  const [subTopicInput, setSubTopicInput] = useState('');
  const [newDevice, setNewDevice] = useState({
    device_id: '',
    device_name: '',
    device_type: 'camera',
    node_id: '',
    ip_address: '',
    location: ''
  });
  const [newSensor, setNewSensor] = useState({
    sensor_id: '',
    sensor_type: 'camera',
    device_id: '',
    data_rate: '',
    resolution: ''
  });

  const fetchDevices = async () => {
    try {
      const response = await fetch(`${API_URL}/api/devices/`);
      const data = await response.json();
      setDevices(data);
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch devices:', error);
      setLoading(false);
    }
  };

  const fetchNodes = async () => {
    try {
      const response = await fetch(`${API_URL}/api/nodes/`);
      const data = await response.json();
      setNodes(data);
    } catch (error) {
      console.error('Failed to fetch nodes:', error);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchDevices();
    fetchNodes();
  }, []);

  // Smart polling for devices - only when tab is visible
  useSmartPolling(fetchDevices, {
    interval: 15000, // 15 seconds (was 3 seconds)
    enabled: true,
    immediate: false,
    minInterval: 10000,
    maxInterval: 30000,
    onError: (error) => {
      console.warn('Device polling error:', error);
    }
  });

  const handleAddDevice = async () => {
    try {
      const payload = {
        device_id: newDevice.device_id || undefined,
        device_name: newDevice.device_name || undefined,
        device_type: newDevice.device_type,
        node_id: newDevice.node_id || undefined,
        ip_address: newDevice.ip_address || undefined,
        location: newDevice.location || undefined
      };
      
      const response = await fetch(`${API_URL}/api/devices/create/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      
      if (response.ok) {
        const data = await response.json();
        notification.showSuccess(`Device ${data.device_id} created successfully!`);
        setShowAddDevice(false);
        setNewDevice({
          device_id: '',
          device_name: '',
          device_type: 'camera',
          node_id: '',
          ip_address: '',
          location: ''
        });
        fetchDevices();
      } else {
        const error = await response.json();
        notification.showError(`Failed to create device: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to create device:', error);
      notification.showError('Failed to create device');
    }
  };

  const handleDeleteDevice = async (deviceId) => {
    const confirmed = await confirm.confirm({
      title: 'Delete Device',
      message: `Are you sure you want to delete device ${deviceId}? This will also delete all associated sensors. This action cannot be undone.`,
      confirmText: 'Delete',
      cancelText: 'Cancel',
      type: 'danger'
    });

    if (!confirmed) {
      return;
    }

    try {
      const response = await fetch(`${API_URL}/api/devices/${deviceId}/delete/`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
        notification.showSuccess(`Device ${deviceId} deleted successfully!`);
        fetchDevices();
      } else {
        const error = await response.json();
        notification.showError(`Failed to delete device: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to delete device:', error);
      notification.showError('Failed to delete device');
    }
  };

  const handleAddSensor = async () => {
    if (!newSensor.device_id || !newSensor.sensor_id) {
      notification.showWarning('Please fill in device ID and sensor ID');
      return;
    }

    try {
      const payload = {
        sensor_id: newSensor.sensor_id,
        sensor_type: newSensor.sensor_type,
        device_id: newSensor.device_id,
        data_rate: newSensor.data_rate ? parseFloat(newSensor.data_rate) : undefined,
        resolution: newSensor.resolution || undefined
      };
      
      const response = await fetch(`${API_URL}/api/sensors/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      
      if (response.ok) {
        notification.showSuccess(`Sensor ${newSensor.sensor_id} added successfully!`);
        setShowAddSensor(false);
        setNewSensor({
          sensor_id: '',
          sensor_type: 'camera',
          device_id: '',
          data_rate: '',
          resolution: ''
        });
        fetchDevices();
      } else {
        const error = await response.json();
        notification.showError(`Failed to add sensor: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to add sensor:', error);
      notification.showError('Failed to add sensor');
    }
  };

  const openAddSensorModal = (deviceId) => {
    setNewSensor({
      ...newSensor,
      device_id: deviceId
    });
    setSelectedDevice(deviceId);
    setShowAddSensor(true);
  };

  const openConnectionModal = (device) => {
    setConnectionData({
      device_id: device.device_id,
      node_id: device.node_id || '',
      pub_topics: device.pub_topics || [],
      sub_topics: device.sub_topics || []
    });
    setPubTopicInput('');
    setSubTopicInput('');
    setShowConnectionModal(true);
  };

  const handleConnectDevice = async () => {
    if (!connectionData.node_id) {
      notification.showWarning('Please select a node to connect');
      return;
    }

    try {
      const response = await fetch(`${API_URL}/api/connections/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(connectionData),
      });

      if (response.ok) {
        notification.showSuccess(`Device ${connectionData.device_id} connected to node ${connectionData.node_id}`);
        setShowConnectionModal(false);
        fetchDevices();
      } else {
        const error = await response.json();
        notification.showError(`Failed to connect device: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to connect device:', error);
      notification.showError('Failed to connect device');
    }
  };

  const handleDisconnectDevice = async (deviceId) => {
    const confirmed = await confirm.confirm({
      title: 'Disconnect Device',
      message: `Are you sure you want to disconnect this device from its node?`,
      confirmText: 'Disconnect',
      cancelText: 'Cancel',
      type: 'warning'
    });

    if (!confirmed) {
      return;
    }

    try {
      const response = await fetch(`${API_URL}/api/connections/`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ device_id: deviceId }),
      });

      if (response.ok) {
        notification.showSuccess(`Device ${deviceId} disconnected`);
        fetchDevices();
      } else {
        const error = await response.json();
        notification.showError(`Failed to disconnect device: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to disconnect device:', error);
      notification.showError('Failed to disconnect device');
    }
  };

  const addTopic = (type) => {
    const topicValue = type === 'pub' ? pubTopicInput : subTopicInput;
    
    if (!topicValue.trim()) {
      notification.showWarning('Please enter a topic name');
      return;
    }

    if (type === 'pub') {
      setConnectionData({
        ...connectionData,
        pub_topics: [...connectionData.pub_topics, topicValue.trim()]
      });
      setPubTopicInput('');
    } else {
      setConnectionData({
        ...connectionData,
        sub_topics: [...connectionData.sub_topics, topicValue.trim()]
      });
      setSubTopicInput('');
    }
  };

  const removeTopic = (type, index) => {
    if (type === 'pub') {
      setConnectionData({
        ...connectionData,
        pub_topics: connectionData.pub_topics.filter((_, i) => i !== index)
      });
    } else {
      setConnectionData({
        ...connectionData,
        sub_topics: connectionData.sub_topics.filter((_, i) => i !== index)
      });
    }
  };

  if (loading) {
    return <div className="loading">Loading devices...</div>;
  }

  return (
    <div className="devices-view">
      <div className="page-header">
        <h1 className="page-title">Devices & Sensors</h1>
        <div className="header-actions">
          <button 
            className="btn btn-primary"
            onClick={() => setShowAddDevice(true)}
          >
            + Add Device
          </button>
        </div>
      </div>

      {/* Add Device Modal */}
      {showAddDevice && (
        <div className="modal-overlay" onClick={() => setShowAddDevice(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Add New Device</h2>
            <div className="form-group">
              <label>Device ID (optional - auto-generated if empty):</label>
              <input
                type="text"
                value={newDevice.device_id}
                onChange={(e) => setNewDevice({...newDevice, device_id: e.target.value})}
                placeholder="e.g., device_0"
              />
            </div>
            <div className="form-group">
              <label>Device Name:</label>
              <input
                type="text"
                value={newDevice.device_name}
                onChange={(e) => setNewDevice({...newDevice, device_name: e.target.value})}
                placeholder="e.g., Camera 1"
              />
            </div>
            <div className="form-group">
              <label>Device Type:</label>
              <select
                value={newDevice.device_type}
                onChange={(e) => setNewDevice({...newDevice, device_type: e.target.value})}
                className="form-select"
              >
                <option value="camera">Camera</option>
                <option value="stereo_camera">Stereo Camera</option>
                <option value="depth_camera">Depth Camera</option>
                <option value="thermal_camera">Thermal Camera</option>
                <option value="lidar">LiDAR</option>
                <option value="radar">Radar</option>
                <option value="imu">IMU</option>
                <option value="gps">GPS / GNSS</option>
                <option value="ultrasonic_sensor">Ultrasonic Sensor</option>
                <option value="infrared_sensor">Infrared Sensor</option>
                <option value="force_torque_sensor">Force/Torque Sensor</option>
                <option value="encoder">Encoder</option>
                <option value="motor_controller">Motor Controller</option>
                <option value="servo_motor">Servo Motor</option>
                <option value="stepper_motor">Stepper Motor</option>
                <option value="actuator">Actuator</option>
                <option value="power_supply">Power Supply</option>
                <option value="edge_computer">Edge Computer</option>
                <option value="gateway">Gateway</option>
                <option value="network_switch">Network Switch</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div className="form-group">
              <label>Connected Node (optional):</label>
              <select
                value={newDevice.node_id}
                onChange={(e) => setNewDevice({...newDevice, node_id: e.target.value})}
                className="form-select"
              >
                <option value="">-- No Node --</option>
                {nodes.map(node => (
                  <option key={node.node_id} value={node.node_id}>
                    {node.node_id} ({node.status})
                  </option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label>IP Address (optional):</label>
              <input
                type="text"
                value={newDevice.ip_address}
                onChange={(e) => setNewDevice({...newDevice, ip_address: e.target.value})}
                placeholder="e.g., 192.168.1.100"
              />
            </div>
            <div className="form-group">
              <label>Location (optional):</label>
              <input
                type="text"
                value={newDevice.location}
                onChange={(e) => setNewDevice({...newDevice, location: e.target.value})}
                placeholder="e.g., Lab A, Room 101"
              />
            </div>
            <div className="modal-actions">
              <button className="btn btn-primary" onClick={handleAddDevice}>
                Create Device
              </button>
              <button className="btn btn-secondary" onClick={() => setShowAddDevice(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Connection Modal */}
      {showConnectionModal && (
        <div className="modal-overlay" onClick={() => setShowConnectionModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '600px' }}>
            <h2>Connect Device to Node</h2>
            
            <div className="form-group">
              <label>Select Node:</label>
              <select
                value={connectionData.node_id}
                onChange={(e) => setConnectionData({...connectionData, node_id: e.target.value})}
                className="form-select"
              >
                <option value="">-- Select Node --</option>
                {nodes.map(node => (
                  <option key={node.node_id} value={node.node_id}>
                    {node.node_id} ({node.status})
                  </option>
                ))}
              </select>
            </div>

            <div className="topics-section">
              <div className="topics-header">
                <h3>Publish Topics (Device → Node)</h3>
                <button 
                  className="btn btn-small btn-secondary"
                  onClick={() => {
                    // Auto-generate topics based on device type
                    const device = devices.find(d => d.device_id === connectionData.device_id);
                    if (device) {
                      notification.showInfo('Topics will be auto-generated based on device type when connecting');
                    }
                  }}
                  title="Topics are auto-generated based on device type"
                >
                  Auto-generate
                </button>
              </div>
              <p className="topics-hint">
                Topics are automatically generated based on device type. You can add custom topics below.
              </p>
              <div className="topics-input-group">
                <input
                  type="text"
                  value={pubTopicInput}
                  onChange={(e) => setPubTopicInput(e.target.value)}
                  placeholder="e.g., /device/custom_topic"
                  className="form-input"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      addTopic('pub');
                    }
                  }}
                />
                <button className="btn btn-primary" onClick={() => addTopic('pub')}>
                  Add Custom
                </button>
              </div>
              <div className="topics-list">
                {connectionData.pub_topics && connectionData.pub_topics.length > 0 ? (
                  connectionData.pub_topics.map((topic, idx) => (
                    <span key={idx} className="topic-tag topic-pub" title={getTopicDescription(topic)}>
                      {topic}
                      <button className="topic-remove" onClick={() => removeTopic('pub', idx)}>×</button>
                    </span>
                  ))
                ) : (
                  <p className="no-topics-hint">No publish topics configured. Topics will be auto-generated when connecting.</p>
                )}
              </div>
            </div>

            <div className="topics-section">
              <div className="topics-header">
                <h3>Subscribe Topics (Node → Device)</h3>
              </div>
              <p className="topics-hint">
                Control and command topics for the device.
              </p>
              <div className="topics-input-group">
                <input
                  type="text"
                  value={subTopicInput}
                  onChange={(e) => setSubTopicInput(e.target.value)}
                  placeholder="e.g., /device/cmd/enable"
                  className="form-input"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      addTopic('sub');
                    }
                  }}
                />
                <button className="btn btn-primary" onClick={() => addTopic('sub')}>
                  Add Custom
                </button>
              </div>
              <div className="topics-list">
                {connectionData.sub_topics && connectionData.sub_topics.length > 0 ? (
                  connectionData.sub_topics.map((topic, idx) => (
                    <span key={idx} className="topic-tag topic-sub" title={getTopicDescription(topic)}>
                      {topic}
                      <button className="topic-remove" onClick={() => removeTopic('sub', idx)}>×</button>
                    </span>
                  ))
                ) : (
                  <p className="no-topics-hint">No subscribe topics configured. Topics will be auto-generated when connecting.</p>
                )}
              </div>
            </div>

            <div className="modal-actions">
              <button className="btn btn-primary" onClick={handleConnectDevice}>
                {connectionData.node_id ? 'Update Connection' : 'Connect'}
              </button>
              <button className="btn btn-secondary" onClick={() => setShowConnectionModal(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add Sensor Modal */}
      {showAddSensor && (
        <div className="modal-overlay" onClick={() => setShowAddSensor(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2>Add Sensor to {selectedDevice}</h2>
            <div className="form-group">
              <label>Sensor ID:</label>
              <input
                type="text"
                value={newSensor.sensor_id}
                onChange={(e) => setNewSensor({...newSensor, sensor_id: e.target.value})}
                placeholder="e.g., camera_0"
                required
              />
            </div>
            <div className="form-group">
              <label>Sensor Type:</label>
              <select
                value={newSensor.sensor_type}
                onChange={(e) => setNewSensor({...newSensor, sensor_type: e.target.value})}
                className="form-select"
              >
                <option value="camera">Camera</option>
                <option value="lidar">LiDAR</option>
                <option value="imu">IMU</option>
                <option value="gps">GPS</option>
                <option value="ultrasonic">Ultrasonic</option>
                <option value="temperature">Temperature</option>
                <option value="pressure">Pressure</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div className="form-group">
              <label>Data Rate (Hz, optional):</label>
              <input
                type="number"
                value={newSensor.data_rate}
                onChange={(e) => setNewSensor({...newSensor, data_rate: e.target.value})}
                placeholder="e.g., 30"
              />
            </div>
            <div className="form-group">
              <label>Resolution (optional):</label>
              <input
                type="text"
                value={newSensor.resolution}
                onChange={(e) => setNewSensor({...newSensor, resolution: e.target.value})}
                placeholder="e.g., 1920x1080"
              />
            </div>
            <div className="modal-actions">
              <button className="btn btn-primary" onClick={handleAddSensor}>
                Add Sensor
              </button>
              <button className="btn btn-secondary" onClick={() => setShowAddSensor(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="devices-grid grid grid-2">
        {devices.length === 0 ? (
          <div className="card">
            <p>No devices connected. Click "Add Device" to create one.</p>
          </div>
        ) : (
          devices.map(device => (
            <div key={device.device_id} className="device-card card">
              <div className="device-header">
                <div>
                  <h3>{device.device_name || device.device_id}</h3>
                  <p className="device-id">{device.device_id}</p>
                </div>
                <div className="device-badges">
                  <span className={`device-type type-${device.device_type}`}>
                    {getDeviceTypeLabel(device.device_type)}
                  </span>
                  <span className={`device-status status-${device.status || 'active'}`}>
                    {device.status || 'active'}
                  </span>
                </div>
              </div>

              <div className="device-info">
                <div className="info-row">
                  <span className="info-label">Type:</span>
                  <span className={`device-type type-${device.device_type}`}>
                    {getDeviceTypeLabel(device.device_type)}
                  </span>
                </div>
                {device.node_id && (
                  <div className="info-row">
                    <span className="info-label">Connected Node:</span>
                    <span className="info-value node-link">{device.node_id}</span>
                  </div>
                )}
                {device.ip_address && (
                  <div className="info-row">
                    <span className="info-label">IP Address:</span>
                    <span className="info-value">{device.ip_address}</span>
                  </div>
                )}
                {device.location && (
                  <div className="info-row">
                    <span className="info-label">Location:</span>
                    <span className="info-value">{device.location}</span>
                  </div>
                )}
              </div>

              <div className="sensors-section">
                <div className="sensors-header">
                  <h4>Sensors ({device.sensors?.length || 0})</h4>
                  <button 
                    className="btn btn-small btn-primary"
                    onClick={() => openAddSensorModal(device.device_id)}
                  >
                    + Add Sensor
                  </button>
                </div>
                {device.sensors && device.sensors.length > 0 ? (
                  <div className="sensors-list">
                    {device.sensors.map((sensor, idx) => (
                      <div key={idx} className="sensor-item">
                        <div className="sensor-info">
                          <span className="sensor-id">{sensor.sensor_id}</span>
                          <span className={`sensor-type type-${sensor.sensor_type}`}>
                            {sensor.sensor_type}
                          </span>
                        </div>
                        {sensor.data_rate && (
                          <div className="sensor-detail">
                            <span>Rate: {sensor.data_rate} Hz</span>
                          </div>
                        )}
                        {sensor.resolution && (
                          <div className="sensor-detail">
                            <span>Resolution: {sensor.resolution}</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="no-sensors">No sensors added yet</p>
                )}
              </div>

              {/* Node Connection & Topics */}
              {device.node_id && (
                <div className="connection-section">
                  <div className="connection-header">
                    <h4>Node Connection</h4>
                    <span className="node-badge">{device.node_id}</span>
                  </div>
                  {(device.pub_topics && device.pub_topics.length > 0) || 
                   (device.sub_topics && device.sub_topics.length > 0) ? (
                    <div className="topics-list">
                      {device.pub_topics && device.pub_topics.length > 0 && (
                        <div className="topics-group">
                          <span className="topics-label">Publish Topics:</span>
                          {device.pub_topics.map((topic, idx) => (
                            <span key={idx} className="topic-tag topic-pub">{topic}</span>
                          ))}
                        </div>
                      )}
                      {device.sub_topics && device.sub_topics.length > 0 && (
                        <div className="topics-group">
                          <span className="topics-label">Subscribe Topics:</span>
                          {device.sub_topics.map((topic, idx) => (
                            <span key={idx} className="topic-tag topic-sub">{topic}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="no-topics">No topics configured</p>
                  )}
                </div>
              )}

              <div className="device-actions">
                <button 
                  className="btn btn-secondary"
                  onClick={() => openConnectionModal(device)}
                >
                  {device.node_id ? 'Manage Connection' : 'Connect to Node'}
                </button>
                {device.node_id && (
                  <button 
                    className="btn btn-warning"
                    onClick={() => handleDisconnectDevice(device.device_id)}
                  >
                    Disconnect
                  </button>
                )}
                <button 
                  className="btn btn-danger"
                  onClick={() => handleDeleteDevice(device.device_id)}
                >
                  Delete Device
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default DevicesView;

