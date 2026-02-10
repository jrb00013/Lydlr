"""
ROS2 Metrics Collector Service
Subscribes to ROS2 metrics topics and forwards them to the backend API
"""
import os
import sys
import subprocess
import threading
import time
import json
import logging
import requests
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Store active collectors
active_collectors: Dict[str, subprocess.Popen] = {}


def get_ros2_command() -> Optional[str]:
    """Get ROS2 command path"""
    ros2_paths = [
        '/opt/ros/humble/bin/ros2',
        '/opt/ros/foxy/bin/ros2',
        '/usr/bin/ros2',
    ]
    
    for path in ros2_paths:
        if os.path.exists(path):
            return path
    
    import shutil
    ros2_cmd = shutil.which('ros2')
    if ros2_cmd:
        return ros2_cmd
    
    return None


def find_ros2_workspace() -> Optional[Path]:
    """Find ROS2 workspace directory"""
    possible_paths = [
        Path(__file__).parent.parent.parent,
        Path.home() / 'lydlr_ws',
        Path('/root/lydlr/lydlr_ws'),
        Path.home() / 'Documents' / 'Lydlr',
        Path.home() / 'Documents' / 'lydlr' / 'lydlr_ws',
    ]
    
    for path in possible_paths:
        if path.exists() and (path / 'src').exists() and (path / 'install').exists():
            return path
    
    current = Path.cwd()
    while current != current.parent:
        if (current / 'src').exists() and (current / 'install').exists():
            return current
        current = current.parent
    
    return None


def get_ros2_distro() -> Optional[str]:
    """Detect ROS2 distribution"""
    distros = ['humble', 'foxy', 'galactic', 'rolling']
    for distro in distros:
        setup_file = Path(f'/opt/ros/{distro}/setup.bash')
        if setup_file.exists():
            return distro
    return None


def collect_metrics_via_ros2_topic_echo(node_id: str, api_url: str = "http://localhost:8000"):
    """
    Use ros2 topic echo in a loop to collect metrics and forward to backend
    """
    ros2_cmd = get_ros2_command()
    if not ros2_cmd:
        logger.warning(f"ROS2 not found, cannot collect metrics for {node_id}")
        return
    
    workspace_dir = find_ros2_workspace()
    if not workspace_dir:
        logger.warning(f"ROS2 workspace not found, cannot collect metrics for {node_id}")
        return
    
    ros2_distro = get_ros2_distro()
    if not ros2_distro:
        logger.warning(f"ROS2 distribution not found, cannot collect metrics for {node_id}")
        return
    
    install_setup = workspace_dir / 'install' / 'setup.bash'
    ros2_setup = f'/opt/ros/{ros2_distro}/setup.bash'
    topic_name = f'/{node_id}/metrics'
    
    def collect_loop():
        logger.info(f"Starting metrics collection loop for {node_id}")
        while node_id in active_collectors:
            try:
                # Use ros2 topic echo with timeout
                cmd_str = (
                    f'source {ros2_setup} && '
                    f'source {install_setup} && '
                    f'timeout 5 {ros2_cmd} topic echo {topic_name} --once 2>/dev/null'
                )
                
                result = subprocess.run(
                    ['/bin/bash', '-c', cmd_str],
                    cwd=str(workspace_dir),
                    capture_output=True,
                    text=True,
                    timeout=6,
                    env=os.environ.copy()
                )
                
                if result.returncode == 0 and result.stdout:
                    # Parse output
                    lines = result.stdout.strip().split('\n')
                    data_line = None
                    for line in lines:
                        if 'data:' in line:
                            data_line = line
                            break
                    
                    if data_line:
                        # Extract float values
                        import re
                        floats = re.findall(r'[-+]?\d*\.\d+|\d+', data_line)
                        if len(floats) >= 5:
                            metrics_data = {
                                'node_id': node_id,
                                'compression_ratio': float(floats[0]),
                                'latency_ms': float(floats[1]),
                                'compression_level': float(floats[2]),
                                'quality_score': float(floats[3]),
                                'bandwidth_estimate': float(floats[4]),
                                'timestamp': datetime.utcnow().isoformat()
                            }
                            
                            # Forward to backend
                            try:
                                response = requests.post(
                                    f'{api_url}/api/metrics/',
                                    json=metrics_data,
                                    timeout=2
                                )
                                if response.status_code == 200:
                                    logger.debug(f"Metrics forwarded for {node_id}: {metrics_data['compression_ratio']:.2f}x")
                                else:
                                    logger.warning(f"Failed to forward metrics: {response.status_code}")
                            except Exception as e:
                                logger.warning(f"Failed to forward metrics: {e}")
                
                time.sleep(2)  # Collect every 2 seconds
                
            except subprocess.TimeoutExpired:
                # Topic might not exist yet, wait longer
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error collecting metrics for {node_id}: {e}")
                time.sleep(5)
    
    thread = threading.Thread(target=collect_loop, daemon=True)
    thread.start()
    active_collectors[node_id] = thread  # Store thread reference
    logger.info(f"Started metrics collector for node {node_id}")


def start_metrics_collector(node_id: str, api_url: str = "http://localhost:8000"):
    """Start metrics collection for a node"""
    if node_id in active_collectors:
        logger.info(f"Metrics collector for {node_id} already running")
        return
    
    collect_metrics_via_ros2_topic_echo(node_id, api_url)


def stop_metrics_collector(node_id: str):
    """Stop metrics collection for a node"""
    if node_id in active_collectors:
        del active_collectors[node_id]
        logger.info(f"Stopped metrics collector for {node_id}")


def stop_all_collectors():
    """Stop all metrics collectors"""
    for node_id in list(active_collectors.keys()):
        stop_metrics_collector(node_id)

