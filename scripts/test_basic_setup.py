#!/usr/bin/env python3
"""
Basic test script to verify ROS2 setup works
"""

import os
import sys

def test_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
    
    try:
        import rclpy
        print("✓ ROS2 Python client imported successfully")
    except ImportError as e:
        print(f"✗ ROS2 import failed: {e}")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")

def test_ros2_topics():
    """Test ROS2 topic listing"""
    print("\nTesting ROS2 topics...")
    
    try:
        import rclpy
        from rclpy.node import Node
        
        # Initialize ROS2
        rclpy.init()
        node = Node('test_node')
        
        # Get topic list
        topics = node.get_topic_names_and_types()
        
        if topics:
            print("✓ ROS2 topics found:")
            for topic_name, topic_types in topics:
                print(f"  - {topic_name}: {topic_types}")
        else:
            print("ℹ No ROS2 topics found (this is normal if no nodes are running)")
        
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()
        
    except Exception as e:
        print(f"✗ ROS2 topic test failed: {e}")

def test_file_structure():
    """Test file structure"""
    print("\nTesting file structure...")
    
    # Check if key files exist (support both old and new structure)
    key_files = []
    if os.path.exists("ros2/src/lydlr_ai"):
        key_files = [
            "ros2/src/lydlr_ai/train.py",
            "ros2/src/lydlr_ai/lydlr_ai/model/compressor.py",
            "scripts/data_loader.py",
            "scripts/collect_training_data.py"
        ]
    elif os.path.exists("src/lydlr_ai"):
        key_files = [
            "src/lydlr_ai/train.py",
            "src/lydlr_ai/lydlr_ai/model/compressor.py",
            "scripts/data_loader.py",
            "scripts/collect_training_data.py"
        ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")

def main():
    """Main test function"""
    print("=== Lydlr Basic Setup Test ===\n")
    
    test_imports()
    test_ros2_topics()
    test_file_structure()
    
    print("\n=== Test Complete ===")
    print("\nNext steps:")
    # Support both old and new structure
    req_path = "ros2/src/lydlr_ai/requirements.txt" if os.path.exists("ros2/src/lydlr_ai/requirements.txt") else "src/lydlr_ai/requirements.txt"
    train_path = "ros2/src/lydlr_ai/train.py" if os.path.exists("ros2/src/lydlr_ai/train.py") else "src/lydlr_ai/train.py"
    print(f"1. Install Python dependencies: pip install -r {req_path}")
    print("2. Test data loader: python3 scripts/data_loader.py")
    print(f"3. Test training: python3 {train_path}")

if __name__ == "__main__":
    main()
