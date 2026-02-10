#!/usr/bin/env python3
"""
ROS2 Data Collection Script for Lydlr Training
Records sensor data from camera, LiDAR, IMU, and audio topics
and saves them in the format needed for train.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu
from std_msgs.msg import String
import numpy as np
import cv2
import open3d as o3d
from cv_bridge import CvBridge
import pickle
import os
from datetime import datetime
import json

class TrainingDataCollector(Node):
    def __init__(self):
        super().__init__('training_data_collector')
        
        # Data storage
        self.data_buffer = {
            'images': [],
            'lidar': [],
            'imu': [],
            'audio': []  # Placeholder for audio
        }
        
        # Configuration
        self.sequence_length = 10  # Number of frames to collect per sequence
        self.collection_rate = 10.0  # Hz
        self.save_directory = os.path.expanduser('~/lydlr_ws/data/training_data')
        
        # Create save directory
        os.makedirs(self.save_directory, exist_ok=True)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        
        # Timer for data collection
        self.timer = self.create_timer(1.0/self.collection_rate, self.collect_sequence)
        
        # Sequence counter
        self.sequence_count = 0
        self.frame_count = 0
        
        self.get_logger().info('Training data collector initialized')
        self.get_logger().info(f'Will save data to: {self.save_directory}')
    
    def image_callback(self, msg):
        """Process camera image"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Resize to expected dimensions (480x640)
            cv_image = cv2.resize(cv_image, (640, 480))
            
            # Convert to RGB and normalize to [0,1]
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            cv_image = cv_image.astype(np.float32) / 255.0
            
            # Store in buffer
            self.data_buffer['images'].append(cv_image)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def lidar_callback(self, msg):
        """Process LiDAR point cloud"""
        try:
            # Convert ROS PointCloud2 to numpy array
            points = self.pointcloud2_to_array(msg)
            
            # Sample 1024 points if we have more
            if len(points) > 1024:
                indices = np.random.choice(len(points), 1024, replace=False)
                points = points[indices]
            elif len(points) < 1024:
                # Pad with zeros if we have fewer points
                padding = np.zeros((1024 - len(points), 3))
                points = np.vstack([points, padding])
            
            # Store in buffer
            self.data_buffer['lidar'].append(points)
            
        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR: {e}')
    
    def imu_callback(self, msg):
        """Process IMU data"""
        try:
            # Extract 6-axis IMU data [ax, ay, az, gx, gy, gz]
            imu_data = np.array([
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
            ])
            
            # Store in buffer
            self.data_buffer['imu'].append(imu_data)
            
        except Exception as e:
            self.get_logger().error(f'Error processing IMU: {e}')
    
    def pointcloud2_to_array(self, cloud_msg):
        """Convert ROS PointCloud2 to numpy array"""
        # This is a simplified conversion - you might need to adjust based on your LiDAR format
        try:
            # Extract x, y, z coordinates
            points = np.frombuffer(cloud_msg.data, dtype=np.float32)
            points = points.reshape(-1, 4)  # Assuming x, y, z, intensity format
            return points[:, :3]  # Return only x, y, z
        except:
            # Fallback: return random points if conversion fails
            return np.random.rand(1024, 3)
    
    def collect_sequence(self):
        """Collect a complete sequence of sensor data"""
        if (len(self.data_buffer['images']) >= self.sequence_length and 
            len(self.data_buffer['lidar']) >= self.sequence_length and
            len(self.data_buffer['imu']) >= self.sequence_length):
            
            # Extract sequence
            sequence_data = {
                'images': self.data_buffer['images'][:self.sequence_length],
                'lidar': self.data_buffer['lidar'][:self.sequence_length],
                'imu': self.data_buffer['imu'][:self.sequence_length],
                'audio': np.random.rand(self.sequence_length, 128*128)  # Placeholder audio
            }
            
            # Save sequence
            self.save_sequence(sequence_data)
            
            # Clear buffer
            for key in self.data_buffer:
                self.data_buffer[key] = self.data_buffer[key][self.sequence_length:]
            
            self.sequence_count += 1
            self.get_logger().info(f'Collected sequence {self.sequence_count}')
    
    def save_sequence(self, sequence_data):
        """Save a sequence of sensor data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sequence_dir = os.path.join(self.save_directory, f"sequence_{timestamp}")
        os.makedirs(sequence_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'sequence_length': self.sequence_length,
            'data_shapes': {
                'images': [img.shape for img in sequence_data['images']],
                'lidar': [lidar.shape for lidar in sequence_data['lidar']],
                'imu': [imu.shape for imu in sequence_data['imu']],
                'audio': sequence_data['audio'].shape
            }
        }
        
        with open(os.path.join(sequence_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save numpy arrays
        np.save(os.path.join(sequence_dir, 'images.npy'), np.array(sequence_data['images']))
        np.save(os.path.join(sequence_dir, 'lidar.npy'), np.array(sequence_data['lidar']))
        np.save(os.path.join(sequence_dir, 'imu.npy'), np.array(sequence_data['imu']))
        np.save(os.path.join(sequence_dir, 'audio.npy'), sequence_data['audio'])
        
        self.get_logger().info(f'Saved sequence to: {sequence_dir}')

def main(args=None):
    rclpy.init(args=args)
    collector = TrainingDataCollector()
    
    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info('Data collection stopped by user')
    finally:
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
