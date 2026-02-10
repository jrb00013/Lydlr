"""
Device Topic Generator
Automatically generates ROS2 topics for all device attributes and sensors
"""
from typing import Dict, List, Optional
from datetime import datetime


def generate_device_topics(device: Dict) -> Dict[str, List[str]]:
    """
    Generate comprehensive ROS2 topics for a device based on its type and attributes
    
    Returns:
        Dict with 'pub_topics' and 'sub_topics' lists
    """
    device_id = device.get('device_id', '')
    device_type = device.get('device_type', 'other')
    sensors = device.get('sensors', [])
    
    pub_topics = []
    sub_topics = []
    
    # Base topic prefix
    base_topic = f"/device/{device_id}"
    
    # ===== PUBLISH TOPICS (Device → Node) =====
    
    # Device status and metadata
    pub_topics.append(f"{base_topic}/status")  # Device status (active/inactive/error)
    pub_topics.append(f"{base_topic}/metadata")  # Device metadata updates
    pub_topics.append(f"{base_topic}/heartbeat")  # Periodic heartbeat
    pub_topics.append(f"{base_topic}/location")  # Location updates
    pub_topics.append(f"{base_topic}/ip_address")  # IP address updates
    
    # Device type-specific data topics
    if device_type in ['camera', 'stereo_camera', 'depth_camera', 'thermal_camera']:
        pub_topics.append(f"{base_topic}/image_raw")  # Raw image data
        pub_topics.append(f"{base_topic}/image_compressed")  # Compressed image
        pub_topics.append(f"{base_topic}/camera_info")  # Camera calibration info
        pub_topics.append(f"{base_topic}/image_metadata")  # Image metadata (timestamp, exposure, etc.)
        
    elif device_type == 'lidar':
        pub_topics.append(f"{base_topic}/points")  # Point cloud data
        pub_topics.append(f"{base_topic}/points_compressed")  # Compressed point cloud
        pub_topics.append(f"{base_topic}/scan")  # 2D scan data
        pub_topics.append(f"{base_topic}/lidar_info")  # LiDAR metadata
        
    elif device_type == 'radar':
        pub_topics.append(f"{base_topic}/detections")  # Radar detections
        pub_topics.append(f"{base_topic}/range")  # Range measurements
        pub_topics.append(f"{base_topic}/radar_info")  # Radar metadata
        
    elif device_type == 'imu':
        pub_topics.append(f"{base_topic}/imu/data")  # IMU data (accel, gyro, mag)
        pub_topics.append(f"{base_topic}/imu/magnetic_field")  # Magnetometer data
        pub_topics.append(f"{base_topic}/imu/temperature")  # IMU temperature
        
    elif device_type == 'gps':
        pub_topics.append(f"{base_topic}/gps/fix")  # GPS fix data
        pub_topics.append(f"{base_topic}/gps/velocity")  # GPS velocity
        pub_topics.append(f"{base_topic}/gps/status")  # GPS status
        
    elif device_type in ['ultrasonic_sensor', 'infrared_sensor']:
        pub_topics.append(f"{base_topic}/range")  # Range measurement
        pub_topics.append(f"{base_topic}/distance")  # Distance data
        pub_topics.append(f"{base_topic}/sensor_data")  # Raw sensor data
        
    elif device_type == 'force_torque_sensor':
        pub_topics.append(f"{base_topic}/wrench")  # Force and torque
        pub_topics.append(f"{base_topic}/force")  # Force data
        pub_topics.append(f"{base_topic}/torque")  # Torque data
        
    elif device_type == 'encoder':
        pub_topics.append(f"{base_topic}/position")  # Encoder position
        pub_topics.append(f"{base_topic}/velocity")  # Encoder velocity
        pub_topics.append(f"{base_topic}/count")  # Encoder count
        
    elif device_type == 'motor_controller':
        pub_topics.append(f"{base_topic}/motor/status")  # Motor status
        pub_topics.append(f"{base_topic}/motor/current")  # Motor current
        pub_topics.append(f"{base_topic}/motor/temperature")  # Motor temperature
        pub_topics.append(f"{base_topic}/motor/velocity")  # Motor velocity
        pub_topics.append(f"{base_topic}/motor/position")  # Motor position
        
    elif device_type in ['servo_motor', 'stepper_motor']:
        pub_topics.append(f"{base_topic}/motor/position")  # Motor position
        pub_topics.append(f"{base_topic}/motor/velocity")  # Motor velocity
        pub_topics.append(f"{base_topic}/motor/current")  # Motor current
        pub_topics.append(f"{base_topic}/motor/temperature")  # Motor temperature
        pub_topics.append(f"{base_topic}/motor/status")  # Motor status
        
    elif device_type == 'actuator':
        pub_topics.append(f"{base_topic}/actuator/position")  # Actuator position
        pub_topics.append(f"{base_topic}/actuator/force")  # Actuator force
        pub_topics.append(f"{base_topic}/actuator/status")  # Actuator status
        
    elif device_type == 'power_supply':
        pub_topics.append(f"{base_topic}/voltage")  # Voltage reading
        pub_topics.append(f"{base_topic}/current")  # Current reading
        pub_topics.append(f"{base_topic}/power")  # Power consumption
        pub_topics.append(f"{base_topic}/battery_level")  # Battery level (if applicable)
        pub_topics.append(f"{base_topic}/power_status")  # Power status
        
    elif device_type in ['edge_computer', 'gateway', 'network_switch']:
        pub_topics.append(f"{base_topic}/system_status")  # System status
        pub_topics.append(f"{base_topic}/cpu_usage")  # CPU usage
        pub_topics.append(f"{base_topic}/memory_usage")  # Memory usage
        pub_topics.append(f"{base_topic}/network_stats")  # Network statistics
        pub_topics.append(f"{base_topic}/temperature")  # System temperature
        
    # Sensor-specific topics (for each sensor attached to device)
    for sensor in sensors:
        sensor_id = sensor.get('sensor_id', '')
        sensor_type = sensor.get('sensor_type', '')
        sensor_base = f"{base_topic}/sensor/{sensor_id}"
        
        if sensor_type == 'camera':
            pub_topics.append(f"{sensor_base}/image_raw")
            pub_topics.append(f"{sensor_base}/camera_info")
        elif sensor_type == 'lidar':
            pub_topics.append(f"{sensor_base}/points")
        elif sensor_type == 'imu':
            pub_topics.append(f"{sensor_base}/data")
        elif sensor_type == 'gps':
            pub_topics.append(f"{sensor_base}/fix")
        elif sensor_type == 'ultrasonic':
            pub_topics.append(f"{sensor_base}/range")
        elif sensor_type == 'infrared':
            pub_topics.append(f"{sensor_base}/range")
        elif sensor_type == 'temperature':
            pub_topics.append(f"{sensor_base}/temperature")
        elif sensor_type == 'pressure':
            pub_topics.append(f"{sensor_base}/pressure")
        else:
            pub_topics.append(f"{sensor_base}/data")  # Generic sensor data
    
    # Device diagnostics and telemetry
    pub_topics.append(f"{base_topic}/diagnostics")  # Device diagnostics
    pub_topics.append(f"{base_topic}/telemetry")  # Device telemetry
    pub_topics.append(f"{base_topic}/errors")  # Error messages
    pub_topics.append(f"{base_topic}/warnings")  # Warning messages
    
    # ===== SUBSCRIBE TOPICS (Node → Device) =====
    
    # Device control
    sub_topics.append(f"{base_topic}/cmd/enable")  # Enable/disable device
    sub_topics.append(f"{base_topic}/cmd/reset")  # Reset device
    sub_topics.append(f"{base_topic}/cmd/configure")  # Configure device
    sub_topics.append(f"{base_topic}/cmd/shutdown")  # Shutdown device
    
    # Device type-specific control topics
    if device_type in ['camera', 'stereo_camera', 'depth_camera', 'thermal_camera']:
        sub_topics.append(f"{base_topic}/cmd/set_exposure")  # Set exposure
        sub_topics.append(f"{base_topic}/cmd/set_gain")  # Set gain
        sub_topics.append(f"{base_topic}/cmd/set_resolution")  # Set resolution
        sub_topics.append(f"{base_topic}/cmd/trigger")  # Trigger capture
        
    elif device_type == 'lidar':
        sub_topics.append(f"{base_topic}/cmd/set_scan_rate")  # Set scan rate
        sub_topics.append(f"{base_topic}/cmd/set_range")  # Set range
        sub_topics.append(f"{base_topic}/cmd/start_scan")  # Start scanning
        sub_topics.append(f"{base_topic}/cmd/stop_scan")  # Stop scanning
        
    elif device_type == 'motor_controller':
        sub_topics.append(f"{base_topic}/cmd/velocity")  # Set velocity
        sub_topics.append(f"{base_topic}/cmd/position")  # Set position
        sub_topics.append(f"{base_topic}/cmd/torque")  # Set torque
        sub_topics.append(f"{base_topic}/cmd/stop")  # Stop motor
        sub_topics.append(f"{base_topic}/cmd/brake")  # Brake motor
        
    elif device_type in ['servo_motor', 'stepper_motor']:
        sub_topics.append(f"{base_topic}/cmd/target_position")  # Set target position
        sub_topics.append(f"{base_topic}/cmd/target_velocity")  # Set target velocity
        sub_topics.append(f"{base_topic}/cmd/enable")  # Enable motor
        sub_topics.append(f"{base_topic}/cmd/disable")  # Disable motor
        
    elif device_type == 'actuator':
        sub_topics.append(f"{base_topic}/cmd/set_position")  # Set position
        sub_topics.append(f"{base_topic}/cmd/set_force")  # Set force
        sub_topics.append(f"{base_topic}/cmd/extend")  # Extend actuator
        sub_topics.append(f"{base_topic}/cmd/retract")  # Retract actuator
        
    elif device_type == 'power_supply':
        sub_topics.append(f"{base_topic}/cmd/set_voltage")  # Set voltage
        sub_topics.append(f"{base_topic}/cmd/set_current_limit")  # Set current limit
        sub_topics.append(f"{base_topic}/cmd/power_on")  # Power on
        sub_topics.append(f"{base_topic}/cmd/power_off")  # Power off
        
    # Sensor control topics
    for sensor in sensors:
        sensor_id = sensor.get('sensor_id', '')
        sensor_type = sensor.get('sensor_type', '')
        sensor_base = f"{base_topic}/sensor/{sensor_id}/cmd"
        
        sub_topics.append(f"{sensor_base}/enable")  # Enable sensor
        sub_topics.append(f"{sensor_base}/disable")  # Disable sensor
        sub_topics.append(f"{sensor_base}/configure")  # Configure sensor
        
        if sensor_type == 'camera':
            sub_topics.append(f"{sensor_base}/set_exposure")
            sub_topics.append(f"{sensor_base}/set_gain")
        elif sensor_type in ['ultrasonic', 'infrared']:
            sub_topics.append(f"{sensor_base}/set_range")
    
    # Device configuration
    sub_topics.append(f"{base_topic}/config/update")  # Update configuration
    sub_topics.append(f"{base_topic}/config/metadata")  # Update metadata
    sub_topics.append(f"{base_topic}/config/location")  # Update location
    
    return {
        "pub_topics": pub_topics,
        "sub_topics": sub_topics
    }


def get_topic_description(topic: str) -> str:
    """Get human-readable description for a topic"""
    topic_lower = topic.lower()
    
    descriptions = {
        "status": "Device status updates (active/inactive/error)",
        "metadata": "Device metadata and configuration",
        "heartbeat": "Periodic device heartbeat signal",
        "location": "Device location updates",
        "ip_address": "Device IP address updates",
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
    }
    
    for key, desc in descriptions.items():
        if key in topic_lower:
            return desc
    
    return "Device communication topic"


def filter_topics_by_device_type(topics: List[str], device_type: str) -> List[str]:
    """Filter topics to only include relevant ones for device type"""
    device_type_lower = device_type.lower()
    
    # Keep all base topics (status, metadata, etc.)
    relevant = [t for t in topics if any(base in t for base in [
        "/status", "/metadata", "/heartbeat", "/location", "/ip_address",
        "/diagnostics", "/telemetry", "/errors", "/warnings", "/cmd/"
    ])]
    
    # Add device-specific topics
    if device_type_lower in ['camera', 'stereo_camera', 'depth_camera', 'thermal_camera']:
        relevant.extend([t for t in topics if 'image' in t or 'camera' in t])
    elif device_type_lower == 'lidar':
        relevant.extend([t for t in topics if 'points' in t or 'scan' in t or 'lidar' in t])
    elif device_type_lower == 'imu':
        relevant.extend([t for t in topics if 'imu' in t])
    elif device_type_lower == 'gps':
        relevant.extend([t for t in topics if 'gps' in t])
    elif 'motor' in device_type_lower:
        relevant.extend([t for t in topics if 'motor' in t])
    elif device_type_lower == 'actuator':
        relevant.extend([t for t in topics if 'actuator' in t])
    elif device_type_lower == 'power_supply':
        relevant.extend([t for t in topics if any(x in t for x in ['voltage', 'current', 'power', 'battery'])])
    
    # Add sensor topics
    relevant.extend([t for t in topics if '/sensor/' in t])
    
    return list(set(relevant))  # Remove duplicates

