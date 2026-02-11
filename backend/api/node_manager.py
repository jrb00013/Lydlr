"""
Node Process Manager
Manages actual node process execution, logging, and tracing
"""
import os
import subprocess
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Store running processes
running_processes: Dict[str, subprocess.Popen] = {}
process_logs: Dict[str, list] = {}
process_info: Dict[str, Dict[str, Any]] = {}


def get_ros2_distro() -> Optional[str]:
    """Detect ROS2 distribution - check in Docker container if available, otherwise local"""
    # First, try to check in Docker container if available
    if is_docker_available():
        ros2_container = get_ros2_container()
        try:
            # Check if container is running
            result = subprocess.run(
                ['docker', 'ps', '--filter', f'name={ros2_container}', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if ros2_container in result.stdout:
                # Container is running, check for ROS2 inside it
                distros = ['humble', 'foxy', 'galactic', 'rolling']
                for distro in distros:
                    check_result = subprocess.run(
                        ['docker', 'exec', ros2_container, 'test', '-f', f'/opt/ros/{distro}/setup.bash'],
                        capture_output=True,
                        timeout=5
                    )
                    if check_result.returncode == 0:
                        logger.info(f"Found ROS2 {distro} in container {ros2_container}")
                        return distro
        except Exception as e:
            logger.debug(f"Could not check ROS2 in container: {e}")
    
    # Fallback: check local filesystem (for local development)
    distros = ['humble', 'foxy', 'galactic', 'rolling']
    for distro in distros:
        setup_file = Path(f'/opt/ros/{distro}/setup.bash')
        if setup_file.exists():
            logger.info(f"Found ROS2 {distro} locally")
            return distro
    
    return None


def get_ros2_container() -> Optional[str]:
    """Get ROS2 container name"""
    return os.getenv('ROS2_CONTAINER', 'lydlr-ros2')


def is_docker_available() -> bool:
    """Check if Docker is available"""
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def find_ros2_workspace() -> Optional[Path]:
    """Find ROS2 workspace directory - check in Docker container if available, otherwise local"""
    # First, try to check in Docker container if available
    if is_docker_available():
        ros2_container = get_ros2_container()
        try:
            # Check if container is running
            result = subprocess.run(
                ['docker', 'ps', '--filter', f'name={ros2_container}', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if ros2_container in result.stdout:
                # Container is running, check for workspace inside it
                # Check if /app/src exists in container (standard ROS2 workspace structure)
                check_result = subprocess.run(
                    ['docker', 'exec', ros2_container, 'test', '-d', '/app/src'],
                    capture_output=True,
                    timeout=5
                )
                if check_result.returncode == 0:
                    # Found workspace in container at /app
                    logger.info(f"Found ROS2 workspace at /app in container {ros2_container}")
                    return Path("/app")  # This will be used with docker exec
                else:
                    logger.warning(f"Container {ros2_container} is running but /app/src not found")
            else:
                logger.warning(f"Container {ros2_container} is not running. Available containers: {result.stdout}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout checking container {ros2_container}")
        except Exception as e:
            logger.warning(f"Could not check workspace in container {ros2_container}: {e}")
    
    # Fallback: check local filesystem (for local development)
    container_workspace = os.getenv('ROS2_WORKSPACE', '/app')
    if Path(container_workspace).exists():
        # Check if it's a ROS2 workspace
        if (Path(container_workspace) / 'src').exists():
            return Path(container_workspace)
    
    # Check common locations (support both old and new structure)
    possible_paths = [
        Path('/app'),  # Container path
        Path('/app/src').parent if Path('/app/src').exists() else None,
        # For local development, check ros2/ directory first (new structure)
        Path(__file__).parent.parent.parent / 'ros2',  # Project root/ros2
        Path(__file__).parent.parent.parent,  # Project root (old structure)
        Path.home() / 'lydlr_ws',
        Path('/root/lydlr/lydlr_ws'),
    ]
    
    for path in possible_paths:
        if path and path.exists() and (path / 'src').exists():
            # Check for install directory (might not exist if not built yet)
            if (path / 'install').exists() or (path / 'src').exists():
                logger.info(f"Found ROS2 workspace locally at {path}")
                return path
    
    return None


def get_ros2_command() -> Optional[str]:
    """Get ROS2 command path"""
    # Check common ROS2 locations
    ros2_paths = [
        '/opt/ros/humble/bin/ros2',
        '/opt/ros/foxy/bin/ros2',
        '/usr/bin/ros2',
    ]
    
    for path in ros2_paths:
        if os.path.exists(path):
            return path
    
    # Try to find in PATH
    import shutil
    ros2_cmd = shutil.which('ros2')
    if ros2_cmd:
        return ros2_cmd
    
    return None


def build_ros2_env_command(workspace_dir: Path, ros2_distro: str, node_id: str, use_docker: bool = False) -> str:
    """Build bash command that sets up ROS2 environment and runs node"""
    # Setup environment variables
    if use_docker:
        # Container paths
        src_path = '/app/src'
        install_path = '/app/install'
        venv_path = '/app/.venv'
    else:
        src_path = str(workspace_dir / 'src')
        install_path = str(workspace_dir / 'install')
        venv_path = str(workspace_dir / '.venv')
    
    # Find Python version in venv (if exists)
    python_version = None
    if Path(venv_path).exists():
        # Try to detect Python version from venv
        for py_ver in ['python3.10', 'python3.11', 'python3.12', 'python3.9']:
            site_packages = Path(venv_path) / 'lib' / py_ver / 'site-packages'
            if site_packages.exists():
                python_version = py_ver
                break
    
    # Build PYTHONPATH
    pythonpath_parts = [src_path]
    if python_version and Path(venv_path).exists():
        pythonpath_parts.append(f'{venv_path}/lib/{python_version}/site-packages')
    pythonpath = ':'.join(pythonpath_parts)
    
    # Build the command
    cmd_parts = [
        # Set PYTHONPATH
        f'export PYTHONPATH="$PYTHONPATH:{pythonpath}"',
        # Set XDG_RUNTIME_DIR
        'export XDG_RUNTIME_DIR=/tmp/runtime-root',
        'mkdir -p "$XDG_RUNTIME_DIR"',
        'chmod 700 "$XDG_RUNTIME_DIR"',
        # Setup Xvfb for headless display (if needed)
        'if ! pgrep -x Xvfb > /dev/null; then',
        '  Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &',
        '  sleep 1',
        'fi',
        'export DISPLAY=:99',
        # Source ROS2 setup
        f'source /opt/ros/{ros2_distro}/setup.bash',
        # Source workspace setup (if install exists)
        f'if [ -f {install_path}/setup.bash ]; then source {install_path}/setup.bash; fi',
        # Set node ID
        f'export NODE_ID={node_id}',
        # Run the node
        'ros2 run lydlr_ai edge_compressor_node'
    ]
    
    return ' && '.join(cmd_parts)


def get_log_directory() -> Path:
    """Get directory for node logs"""
    log_dir = Path('/tmp/lydlr_nodes')
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def start_node(node_id: str, model_version: Optional[str] = None) -> Dict[str, Any]:
    """
    Start a node process
    
    Args:
        node_id: Node identifier
        model_version: Optional model version to deploy
    
    Returns:
        Dict with status, pid, and log_path
    """
    if node_id in running_processes:
        process = running_processes[node_id]
        if process.poll() is None:  # Process is still running
            return {
                "status": "already_running",
                "pid": process.pid,
                "message": f"Node {node_id} is already running"
            }
        else:
            # Process died, remove it
            del running_processes[node_id]
    
    # Check if we should use Docker to execute in ros2-runtime container
    use_docker = is_docker_available()
    ros2_container = get_ros2_container()
    
    # Detect ROS2 distribution
    ros2_distro = get_ros2_distro()
    if not ros2_distro:
        return {
            "status": "error",
            "error": "ROS2 not found. Please ensure ROS2 is installed (e.g., /opt/ros/humble/setup.bash)"
        }
    
    # Find ROS2 workspace
    workspace_dir = find_ros2_workspace()
    
    # If workspace_dir is /app, it's a container workspace - ensure Docker is used
    if workspace_dir and str(workspace_dir) == '/app':
        # Force Docker usage for container workspace
        if not use_docker:
            logger.warning("Workspace is in container (/app) but Docker not detected. Re-checking Docker availability...")
            use_docker = is_docker_available()
            if not use_docker:
                return {
                    "status": "error",
                    "error": "Workspace is in Docker container but Docker is not available. Please ensure Docker is installed and accessible.",
                    "diagnostic": {
                        "workspace_path": str(workspace_dir),
                        "docker_required": True
                    }
                }
        if not ros2_container:
            ros2_container = get_ros2_container()
        logger.info(f"Detected container workspace at /app, using Docker container: {ros2_container}")
    
    if not workspace_dir:
        # If Docker is available, try to verify workspace exists in container
        if use_docker and ros2_container:
            try:
                # Check if container is actually running
                ps_result = subprocess.run(
                    ['docker', 'ps', '--filter', f'name={ros2_container}', '--format', '{{.Names}}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                container_running = ros2_container in ps_result.stdout
                
                if not container_running:
                    return {
                        "status": "error",
                        "error": f"ROS2 container '{ros2_container}' is not running. Please start it with: docker-compose up ros2-runtime",
                        "diagnostic": {
                            "container_name": ros2_container,
                            "container_running": False,
                            "available_containers": ps_result.stdout.strip().split('\n') if ps_result.returncode == 0 else []
                        }
                    }
                
                # Check if workspace exists in container
                result = subprocess.run(
                    ['docker', 'exec', ros2_container, 'test', '-d', '/app/src'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Workspace exists in container, use /app as workspace
                    workspace_dir = Path('/app')
                    logger.info("Found workspace in container, using /app")
                else:
                    # Get more diagnostic info
                    ls_result = subprocess.run(
                        ['docker', 'exec', ros2_container, 'ls', '-la', '/app'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    app_contents = ls_result.stdout if ls_result.returncode == 0 else "Could not list /app"
                    
                    return {
                        "status": "error",
                        "error": f"ROS2 workspace not found in container '{ros2_container}'. /app/src directory does not exist.",
                        "diagnostic": {
                            "container_name": ros2_container,
                            "container_running": True,
                            "/app_contents": app_contents,
                            "suggestion": "The ros2-runtime container may not have the workspace mounted correctly. Check docker-compose.yml volumes."
                        }
                    }
            except subprocess.TimeoutExpired:
                return {
                    "status": "error",
                    "error": f"Timeout checking container '{ros2_container}'. Container may be unresponsive.",
                    "diagnostic": {
                        "container_name": ros2_container,
                        "timeout": True
                    }
                }
            except Exception as e:
                logger.warning(f"Could not verify workspace in container: {e}")
                return {
                    "status": "error",
                    "error": f"ROS2 workspace not found. Could not verify container workspace: {str(e)}",
                    "diagnostic": {
                        "container_name": ros2_container,
                        "exception": str(e),
                        "exception_type": type(e).__name__
                    }
                }
        else:
            return {
                "status": "error",
                "error": "ROS2 workspace not found. Please ensure the workspace is built and install/setup.bash exists.",
                "diagnostic": {
                    "docker_available": use_docker,
                    "container_name": ros2_container if use_docker else None,
                    "suggestion": "If using Docker, ensure the ros2-runtime container is running and has the workspace mounted."
                }
            }
    
    # Check if workspace is built (check in container if using Docker)
    # If workspace_dir is /app, it's definitely in a container, so use_docker should be True
    if use_docker and ros2_container:
        # Check if install/setup.bash exists in container
        try:
            result = subprocess.run(
                ['docker', 'exec', ros2_container, 'test', '-f', '/app/install/setup.bash'],
                capture_output=True,
                timeout=5
            )
            install_exists = result.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to check install/setup.bash in container: {e}")
            install_exists = False
        
        if not install_exists:
            # Try to build the workspace in the container
            logger.info(f"Workspace not built in container, attempting to build...")
            try:
                build_cmd = (
                    f'source /opt/ros/{ros2_distro}/setup.bash && '
                    f'cd /app && '
                    f'colcon build --packages-select lydlr_ai'
                )
                build_result = subprocess.run(
                    ['docker', 'exec', ros2_container, '/bin/bash', '-c', build_cmd],
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout for build
                )
                if build_result.returncode != 0:
                    logger.error(f"Build failed: {build_result.stderr}")
                    return {
                        "status": "error",
                        "error": f"Workspace build failed in container. Please check logs: {build_result.stderr[:200]}"
                    }
                logger.info("Workspace built successfully in container")
            except subprocess.TimeoutExpired:
                return {
                    "status": "error",
                    "error": "Workspace build timed out. Please build manually: docker exec lydlr-ros2 bash -c 'source /opt/ros/humble/setup.bash && cd /app && colcon build --symlink-install'"
                }
            except Exception as e:
                logger.error(f"Failed to build workspace: {e}")
                return {
                    "status": "error",
                    "error": f"Failed to build workspace: {str(e)}"
                }
    elif workspace_dir:
        # Local check (only if not using container)
        install_setup = workspace_dir / 'install' / 'setup.bash'
        if not install_setup.exists():
            return {
                "status": "error",
                "error": f"Workspace not built. Please run 'colcon build' in {workspace_dir}"
            }
    
    # Create log file
    log_dir = get_log_directory()
    log_file = log_dir / f"{node_id}.log"
    
    try:
        # use_docker and ros2_container already set above
        if use_docker:
            # Check if container exists and is running
            try:
                result = subprocess.run(
                    ['docker', 'ps', '--filter', f'name={ros2_container}', '--format', '{{.Names}}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                container_running = ros2_container in result.stdout
            except Exception:
                container_running = False
            
            if not container_running:
                logger.warning(f"ROS2 container {ros2_container} not running, trying local execution")
                use_docker = False
        
        # Build the command that sets up ROS2 environment
        ros2_cmd = build_ros2_env_command(workspace_dir, ros2_distro, node_id, use_docker=use_docker)
        
        logger.info(f"Starting node {node_id} in workspace {workspace_dir} with ROS2 {ros2_distro} (docker={use_docker})")
        
        # Start the node process with proper ROS2 environment
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Node {node_id} started at {datetime.utcnow().isoformat()}\n")
            f.write(f"Workspace: {workspace_dir}\n")
            f.write(f"ROS2 Distro: {ros2_distro}\n")
            f.write(f"Docker: {use_docker}\n")
            f.write(f"Command: {ros2_cmd[:200]}...\n")  # Log first 200 chars of command
            f.write(f"{'='*80}\n")
            f.flush()
            
            if use_docker:
                # Execute command in ROS2 container
                docker_cmd = [
                    'docker', 'exec', '-d',  # -d for detached
                    '-e', f'NODE_ID={node_id}',
                    '-w', '/app',  # Working directory in container
                    ros2_container,
                    '/bin/bash', '-c', ros2_cmd
                ]
                
                process = subprocess.Popen(
                    docker_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy()
                )
                
                # Get PID from container
                time.sleep(0.5)  # Wait a bit for process to start
                try:
                    pid_result = subprocess.run(
                        ['docker', 'exec', ros2_container, 'pgrep', '-f', f'edge_compressor_node.*{node_id}'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if pid_result.returncode == 0 and pid_result.stdout.strip():
                        actual_pid = int(pid_result.stdout.strip().split('\n')[0])
                    else:
                        actual_pid = process.pid
                except Exception:
                    actual_pid = process.pid
            else:
                # Local execution (fallback)
                process = subprocess.Popen(
                    ['/bin/bash', '-c', ros2_cmd],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=str(workspace_dir),
                    env=os.environ.copy(),
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                )
                actual_pid = process.pid
        
        running_processes[node_id] = process
        process_info[node_id] = {
            "pid": actual_pid,
            "docker_pid": process.pid if use_docker else None,
            "started_at": datetime.utcnow().isoformat(),
            "log_file": str(log_file),
            "model_version": model_version,
            "status": "running",
            "container": ros2_container if use_docker else None
        }
        process_logs[node_id] = []
        
        logger.info(f"Started node {node_id} with PID {actual_pid} (docker={use_docker})")
        
        # Deploy model if specified (after a delay to let node start)
        if model_version:
            import threading
            def delayed_deploy():
                time.sleep(3)  # Wait for node to start
                deploy_model_to_node(node_id, model_version)
            threading.Thread(target=delayed_deploy, daemon=True).start()
        
        return {
            "status": "started",
            "pid": actual_pid,
            "log_file": str(log_file),
            "message": f"Node {node_id} started successfully in {'Docker container' if use_docker else 'local environment'}",
            "container": ros2_container if use_docker else None
        }
    
    except Exception as e:
        logger.error(f"Failed to start node {node_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def stop_node(node_id: str) -> Dict[str, Any]:
    """Stop a node process"""
    if node_id not in running_processes:
        # Try to stop in Docker container if not in local processes
        if is_docker_available():
            ros2_container = get_ros2_container()
            try:
                # Kill process in container by NODE_ID
                result = subprocess.run(
                    ['docker', 'exec', ros2_container, 'pkill', '-f', f'edge_compressor_node.*{node_id}'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info(f"Stopped node {node_id} in Docker container")
                    return {
                        "status": "stopped",
                        "message": f"Node {node_id} stopped successfully"
                    }
            except Exception as e:
                logger.warning(f"Failed to stop node {node_id} in Docker: {e}")
        
        return {
            "status": "not_running",
            "message": f"Node {node_id} is not running"
        }
    
    process = running_processes[node_id]
    process_info_data = process_info.get(node_id, {})
    use_docker = process_info_data.get('container') is not None
    ros2_container = process_info_data.get('container')
    
    try:
        if use_docker and ros2_container:
            # Stop process in Docker container
            pid = process_info_data.get('pid')
            if pid:
                try:
                    subprocess.run(
                        ['docker', 'exec', ros2_container, 'kill', str(pid)],
                        timeout=5
                    )
                except Exception:
                    # Try pkill as fallback
                    subprocess.run(
                        ['docker', 'exec', ros2_container, 'pkill', '-f', f'edge_compressor_node.*{node_id}'],
                        timeout=5
                    )
        else:
            # Try graceful shutdown first
            process.terminate()
            
            # Wait up to 5 seconds for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                process.kill()
                process.wait()
        
        # Log shutdown
        log_dir = get_log_directory()
        log_file = log_dir / f"{node_id}.log"
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Node {node_id} stopped at {datetime.utcnow().isoformat()}\n")
            f.write(f"{'='*80}\n")
        
        del running_processes[node_id]
        if node_id in process_info:
            process_info[node_id]["status"] = "stopped"
            process_info[node_id]["stopped_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Stopped node {node_id}")
        
        return {
            "status": "stopped",
            "message": f"Node {node_id} stopped successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to stop node {node_id}: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def restart_node(node_id: str, model_version: Optional[str] = None) -> Dict[str, Any]:
    """Restart a node process"""
    # Stop first
    stop_result = stop_node(node_id)
    if stop_result.get("status") == "error" and stop_result.get("status") != "not_running":
        return stop_result
    
    # Wait a bit
    import time
    time.sleep(1)
    
    # Start again
    return start_node(node_id, model_version)


def get_node_status(node_id: str) -> Dict[str, Any]:
    """Get current status of a node process"""
    info = process_info.get(node_id, {})
    container = info.get('container')
    
    # Check if running in Docker - use process name check which is more reliable
    if container and is_docker_available():
        try:
            # Check if process exists in container by name (more reliable than PID)
            result = subprocess.run(
                ['docker', 'exec', container, 'pgrep', '-f', f'edge_compressor_node.*{node_id}'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                # Process is running - get the PID
                pid = int(result.stdout.strip().split('\n')[0])
                # Update process_info with current PID in case it changed
                if node_id in process_info:
                    process_info[node_id]['pid'] = pid
                return {
                    "status": "running",
                    "pid": pid,
                    "started_at": info.get("started_at"),
                    "log_file": info.get("log_file"),
                    "model_version": info.get("model_version"),
                    "container": container
                }
        except Exception as e:
            logger.debug(f"Failed to check node {node_id} status in Docker: {e}")
    
    # Check local processes
    if node_id in running_processes:
        process = running_processes[node_id]
        poll_result = process.poll()
        
        if poll_result is None:
            # Process is running
            return {
                "status": "running",
                "pid": info.get('pid', process.pid),
                "started_at": info.get("started_at"),
                "log_file": info.get("log_file"),
                "model_version": info.get("model_version")
            }
        else:
            # Process has exited
            del running_processes[node_id]
            if node_id in process_info:
                process_info[node_id]["status"] = "stopped"
            return {
                "status": "stopped",
                "exit_code": poll_result,
                "pid": None
            }
    
    # If we have container info but process not found, it might have stopped
    if container and is_docker_available():
        # Double-check by looking for any edge_compressor_node process
        try:
            result = subprocess.run(
                ['docker', 'exec', container, 'pgrep', '-f', f'edge_compressor_node.*{node_id}'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                # Found it! Update process_info
                pid = int(result.stdout.strip().split('\n')[0])
                if node_id not in process_info:
                    process_info[node_id] = {}
                process_info[node_id].update({
                    "pid": pid,
                    "container": container,
                    "status": "running"
                })
                return {
                    "status": "running",
                    "pid": pid,
                    "container": container
                }
        except Exception:
            pass
    
    return {
        "status": "not_running",
        "pid": None
    }


def get_node_logs(node_id: str, lines: int = 100) -> list:
    """Get recent logs for a node"""
    log_dir = get_log_directory()
    log_file = log_dir / f"{node_id}.log"
    
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            return all_lines[-lines:] if len(all_lines) > lines else all_lines
    except Exception as e:
        logger.error(f"Failed to read logs for {node_id}: {e}")
        return []


def deploy_model_to_node(node_id: str, model_version: str) -> bool:
    """Deploy a model to a running node via ROS2 topic"""
    try:
        # Detect ROS2 distribution
        ros2_distro = get_ros2_distro()
        if not ros2_distro:
            logger.warning("ROS2 not found, cannot deploy model")
            return False
        
        # Check if we should use Docker
        use_docker = is_docker_available()
        ros2_container = get_ros2_container()
        
        if use_docker:
            # Build command for Docker container
            cmd_str = (
                f'source /opt/ros/{ros2_distro}/setup.bash && '
                f'source /app/install/setup.bash && '
                f'ros2 topic pub --once /{node_id}/model/deploy std_msgs/String "data: \'{model_version}\'"'
            )
            
            result = subprocess.run(
                ['docker', 'exec', ros2_container, '/bin/bash', '-c', cmd_str],
                capture_output=True,
                text=True,
                timeout=10
            )
        else:
            # Local execution
            workspace_dir = find_ros2_workspace()
            if not workspace_dir:
                logger.warning("ROS2 workspace not found, cannot deploy model")
                return False
            
            install_setup = workspace_dir / 'install' / 'setup.bash'
            ros2_setup = f'/opt/ros/{ros2_distro}/setup.bash'
            
            cmd_str = (
                f'source {ros2_setup} && '
                f'source {install_setup} && '
                f'ros2 topic pub --once /{node_id}/model/deploy std_msgs/String "data: \'{model_version}\'"'
            )
            
            result = subprocess.run(
                ['/bin/bash', '-c', cmd_str],
                cwd=str(workspace_dir),
                capture_output=True,
                text=True,
                timeout=10,
                env=os.environ.copy()
            )
        
        if result.returncode == 0:
            logger.info(f"Deployed model {model_version} to {node_id}")
            return True
        else:
            logger.error(f"Failed to deploy model: {result.stderr}")
            return False
    
    except Exception as e:
        logger.error(f"Error deploying model to {node_id}: {e}")
        return False


def get_all_running_nodes() -> Dict[str, Dict[str, Any]]:
    """Get status of all running nodes"""
    status = {}
    for node_id in list(running_processes.keys()):
        process = running_processes[node_id]
        poll_result = process.poll()
        if poll_result is None:
            info = process_info.get(node_id, {})
            status[node_id] = {
                "status": "running",
                "pid": process.pid,
                "started_at": info.get("started_at"),
                "model_version": info.get("model_version")
            }
        else:
            # Process died, clean up
            del running_processes[node_id]
    
    return status

