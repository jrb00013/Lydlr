"""
Workspace management views
"""
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from rest_framework.response import Response
from rest_framework import status

from backend.api.views.base import AsyncAPIView, ensure_db_connection
from backend.api.node_manager import (
    find_ros2_workspace, get_ros2_distro,
    is_docker_available, get_ros2_container
)

logger = logging.getLogger(__name__)


class DiagnosticView(AsyncAPIView):
    """Diagnostic endpoint to test ROS2 and workspace detection"""
    
    async def get(self, request):
        """Run diagnostics and return detailed information"""
        diagnostics = {
            "docker": {},
            "ros2": {},
            "workspace": {},
            "container": {},
            "errors": []
        }
        
        # Test Docker availability
        docker_available = is_docker_available()
        diagnostics["docker"]["available"] = docker_available
        
        if docker_available:
            try:
                # Get Docker version
                result = subprocess.run(
                    ['docker', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                diagnostics["docker"]["version"] = result.stdout.strip() if result.returncode == 0 else "unknown"
            except Exception as e:
                diagnostics["docker"]["error"] = str(e)
                diagnostics["errors"].append(f"Docker version check failed: {e}")
        
        # Test ROS2 container
        ros2_container = get_ros2_container()
        diagnostics["container"]["name"] = ros2_container
        diagnostics["container"]["running"] = False
        
        if docker_available and ros2_container:
            try:
                # Check if container exists
                result = subprocess.run(
                    ['docker', 'ps', '-a', '--filter', f'name={ros2_container}', '--format', '{{.Names}} {{.Status}}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if ros2_container in result.stdout:
                    diagnostics["container"]["exists"] = True
                    diagnostics["container"]["status"] = result.stdout.strip()
                    diagnostics["container"]["running"] = "Up" in result.stdout
                    
                    if diagnostics["container"]["running"]:
                        # Test ROS2 in container
                        ros2_check = subprocess.run(
                            ['docker', 'exec', ros2_container, 'test', '-f', '/opt/ros/humble/setup.bash'],
                            capture_output=True,
                            timeout=5
                        )
                        diagnostics["ros2"]["in_container"] = ros2_check.returncode == 0
                        
                        # Test workspace in container
                        src_check = subprocess.run(
                            ['docker', 'exec', ros2_container, 'test', '-d', '/app/src'],
                            capture_output=True,
                            timeout=5
                        )
                        diagnostics["workspace"]["src_exists"] = src_check.returncode == 0
                        
                        install_check = subprocess.run(
                            ['docker', 'exec', ros2_container, 'test', '-f', '/app/install/setup.bash'],
                            capture_output=True,
                            timeout=5
                        )
                        diagnostics["workspace"]["install_exists"] = install_check.returncode == 0
                        
                        # List directory contents
                        ls_result = subprocess.run(
                            ['docker', 'exec', ros2_container, 'ls', '-la', '/app'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        diagnostics["workspace"]["app_contents"] = ls_result.stdout if ls_result.returncode == 0 else None
                        
                        # Check src directory
                        src_ls = subprocess.run(
                            ['docker', 'exec', ros2_container, 'ls', '-la', '/app/src'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        diagnostics["workspace"]["src_contents"] = src_ls.stdout if src_ls.returncode == 0 else None
                else:
                    diagnostics["container"]["exists"] = False
                    # List all containers
                    all_containers = subprocess.run(
                        ['docker', 'ps', '-a', '--format', '{{.Names}}'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    diagnostics["container"]["all_containers"] = all_containers.stdout.strip().split('\n') if all_containers.returncode == 0 else []
            except Exception as e:
                diagnostics["container"]["error"] = str(e)
                diagnostics["errors"].append(f"Container check failed: {e}")
        
        # Test ROS2 detection
        ros2_distro = get_ros2_distro()
        diagnostics["ros2"]["detected"] = ros2_distro is not None
        diagnostics["ros2"]["distro"] = ros2_distro
        
        # Test workspace detection
        workspace_dir = find_ros2_workspace()
        diagnostics["workspace"]["detected"] = workspace_dir is not None
        diagnostics["workspace"]["path"] = str(workspace_dir) if workspace_dir else None
        
        # Test local ROS2 (if not using Docker)
        if not docker_available:
            for distro in ['humble', 'foxy', 'galactic', 'rolling']:
                setup_file = Path(f'/opt/ros/{distro}/setup.bash')
                if setup_file.exists():
                    diagnostics["ros2"]["local_path"] = str(setup_file)
                    break
        
        return Response(diagnostics)


class OrchestrationStatusView(AsyncAPIView):
    """Orchestration status endpoint"""
    
    async def get(self, request):
        """Get orchestration status"""
        try:
            db = await ensure_db_connection()
            db_connected = db is not None
        except Exception:
            db_connected = False
        
        status_info = {
            "status": "operational",
            "docker_available": is_docker_available(),
            "ros2_container": get_ros2_container() if is_docker_available() else None,
            "database_connected": db_connected,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return Response(status_info)


class WorkspaceView(AsyncAPIView):
    """Workspace management - build, status, info"""
    
    async def get(self, request):
        """Get workspace status and info with diagnostics"""
        workspace_dir = find_ros2_workspace()
        ros2_distro = get_ros2_distro()
        use_docker = is_docker_available()
        ros2_container = get_ros2_container() if use_docker else None
        
        workspace_info = {
            "workspace_path": str(workspace_dir) if workspace_dir else None,
            "ros2_distro": ros2_distro,
            "use_docker": use_docker,
            "container": ros2_container,
            "is_built": False,
            "has_src": False,
            "has_install": False,
            "packages": [],
            "diagnostics": {}
        }
        
        # Add diagnostic information
        diagnostics = {}
        diagnostics["docker_available"] = use_docker
        diagnostics["container_name"] = ros2_container
        diagnostics["container_running"] = False
        
        if use_docker and ros2_container:
            try:
                result = subprocess.run(
                    ['docker', 'ps', '--filter', f'name={ros2_container}', '--format', '{{.Names}}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                diagnostics["container_running"] = ros2_container in result.stdout
                if diagnostics["container_running"]:
                    # Check what's in /app
                    ls_result = subprocess.run(
                        ['docker', 'exec', ros2_container, 'ls', '-la', '/app'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    diagnostics["app_contents"] = ls_result.stdout if ls_result.returncode == 0 else "Error listing /app"
                    
                    # Check src
                    src_check = subprocess.run(
                        ['docker', 'exec', ros2_container, 'test', '-d', '/app/src'],
                        capture_output=True,
                        timeout=5
                    )
                    diagnostics["src_exists"] = src_check.returncode == 0
            except Exception as e:
                diagnostics["error"] = str(e)
        
        workspace_info["diagnostics"] = diagnostics
        
        if workspace_dir:
            workspace_info["has_src"] = (workspace_dir / 'src').exists()
            workspace_info["has_install"] = (workspace_dir / 'install').exists()
            
            # Check if built (check in container if using Docker)
            if use_docker and ros2_container:
                try:
                    result = subprocess.run(
                        ['docker', 'exec', ros2_container, 'test', '-f', '/app/install/setup.bash'],
                        capture_output=True,
                        timeout=5
                    )
                    workspace_info["is_built"] = result.returncode == 0
                    
                    # Get packages
                    if workspace_info["is_built"]:
                        pkg_result = subprocess.run(
                            ['docker', 'exec', ros2_container, 'find', '/app/src', '-name', 'package.xml', '-type', 'f'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if pkg_result.returncode == 0:
                            packages = []
                            for pkg_path in pkg_result.stdout.strip().split('\n'):
                                if pkg_path:
                                    pkg_name = Path(pkg_path).parent.name
                                    packages.append(pkg_name)
                            workspace_info["packages"] = packages
                except Exception as e:
                    logger.warning(f"Failed to check workspace in container: {e}")
            else:
                install_setup = workspace_dir / 'install' / 'setup.bash'
                workspace_info["is_built"] = install_setup.exists()
                
                # Get packages
                if (workspace_dir / 'src').exists():
                    src_dir = workspace_dir / 'src'
                    packages = []
                    for pkg_dir in src_dir.iterdir():
                        if (pkg_dir / 'package.xml').exists() or (pkg_dir / 'setup.py').exists():
                            packages.append(pkg_dir.name)
                    workspace_info["packages"] = packages
        
        return Response(workspace_info)
    
    async def post(self, request):
        """Build workspace"""
        action = request.data.get('action', 'build')
        workspace_dir = find_ros2_workspace()
        ros2_distro = get_ros2_distro()
        use_docker = is_docker_available()
        ros2_container = get_ros2_container() if use_docker else None
        
        if not workspace_dir:
            return Response(
                {"status": "error", "error": "Workspace not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        if not ros2_distro:
            return Response(
                {"status": "error", "error": "ROS2 not found"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        if action == 'build':
            # Build workspace
            if use_docker and ros2_container:
                # Build in container
                build_cmd = (
                    f'source /opt/ros/{ros2_distro}/setup.bash && '
                    f'cd /app && '
                    f'colcon build --packages-select lydlr_ai 2>&1'
                )
                
                try:
                    # Run build and capture output
                    process = subprocess.Popen(
                        ['docker', 'exec', ros2_container, '/bin/bash', '-c', build_cmd],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    
                    # Read output in real-time
                    output_lines = []
                    for line in process.stdout:
                        output_lines.append(line.strip())
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        return Response({
                            "status": "success",
                            "message": "Workspace built successfully",
                            "output": output_lines[-50:]  # Last 50 lines
                        })
                    else:
                        return Response({
                            "status": "error",
                            "error": "Build failed",
                            "output": output_lines[-50:]
                        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                        
                except Exception as e:
                    logger.error(f"Build failed: {e}")
                    return Response(
                        {"status": "error", "error": str(e)},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
            else:
                # Local build
                build_cmd = (
                    f'source /opt/ros/{ros2_distro}/setup.bash && '
                    f'cd {workspace_dir} && '
                    f'colcon build --packages-select lydlr_ai 2>&1'
                )
                
                try:
                    result = subprocess.run(
                        ['/bin/bash', '-c', build_cmd],
                        cwd=str(workspace_dir),
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    
                    if result.returncode == 0:
                        return Response({
                            "status": "success",
                            "message": "Workspace built successfully",
                            "output": result.stdout.split('\n')[-50:]
                        })
                    else:
                        return Response({
                            "status": "error",
                            "error": "Build failed",
                            "output": result.stderr.split('\n')[-50:] if result.stderr else result.stdout.split('\n')[-50:]
                        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                        
                except subprocess.TimeoutExpired:
                    return Response(
                        {"status": "error", "error": "Build timed out"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                except Exception as e:
                    logger.error(f"Build failed: {e}")
                    return Response(
                        {"status": "error", "error": str(e)},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
        
        return Response(
            {"status": "error", "error": f"Unknown action: {action}"},
            status=status.HTTP_400_BAD_REQUEST
        )

