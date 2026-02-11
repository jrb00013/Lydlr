#!/usr/bin/env python3
# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# Python-based Master Launcher for Lydlr System
# - More robust than bash script
# - Better error handling
# - Cross-platform support

import os
import sys
import subprocess
import time
import signal
import json
from pathlib import Path
from typing import List, Dict
import argparse

class LydlrLauncher:
    def __init__(self, num_nodes=2, model_version="vv1.0"):
        self.script_dir = Path(__file__).parent.absolute()
        self.workspace_dir = self.script_dir
        self.src_dir = self.workspace_dir / "src" / "lydlr_ai"
        self.model_dir = self.src_dir / "models"
        self.num_nodes = num_nodes
        self.model_version = model_version
        self.processes: List[subprocess.Popen] = []
        
    def print_status(self, message):
        print(f"✅ {message}")
    
    def print_error(self, message):
        print(f"❌ {message}")
    
    def print_info(self, message):
        print(f"ℹ️  {message}")
    
    def check_ros2(self):
        """Check if ROS2 is available"""
        self.print_info("Checking ROS2 installation...")
        ros2_setup = Path("/opt/ros/humble/setup.bash")
        if ros2_setup.exists():
            self.print_status("ROS2 Humble found")
            return True
        else:
            self.print_error("ROS2 Humble not found")
            return False
    
    def setup_python_env(self):
        """Setup Python virtual environment"""
        self.print_info("Setting up Python environment...")
        venv_dir = self.workspace_dir / ".venv"
        
        if not venv_dir.exists():
            self.print_info("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        
        # Activate venv (for subprocesses)
        if sys.platform == "win32":
            python_exe = venv_dir / "Scripts" / "python.exe"
            pip_exe = venv_dir / "Scripts" / "pip.exe"
        else:
            python_exe = venv_dir / "bin" / "python"
            pip_exe = venv_dir / "bin" / "pip"
        
        # Check dependencies
        try:
            import torch
            self.print_status("Dependencies ready")
        except ImportError:
            self.print_info("Installing dependencies...")
            subprocess.run([str(pip_exe), "install", "torch", "torchvision", "torchaudio", 
                          "numpy", "tqdm", "lpips", "scikit-image", "psutil", "open3d", 
                          "--quiet"], check=True)
            self.print_status("Dependencies installed")
        
        return python_exe, pip_exe
    
    def build_ros2_package(self):
        """Build ROS2 package"""
        self.print_info("Building ROS2 package...")
        os.chdir(self.workspace_dir)
        
        # Source ROS2 and build
        build_cmd = [
            "bash", "-c",
            "source /opt/ros/humble/setup.bash && "
            "colcon build --packages-select lydlr_ai"
        ]
        
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            self.print_status("Package built successfully")
            return True
        else:
            self.print_error(f"Build failed: {result.stderr}")
            return False
    
    def check_models(self):
        """Check if models exist"""
        self.print_info("Checking trained models...")
        model_file = self.model_dir / f"compressor_{self.model_version}.pth"
        
        if model_file.exists():
            self.print_status(f"Model found: {model_file.name}")
            return True
        else:
            self.print_error(f"Model not found: {model_file}")
            self.print_info("Available models:")
            for pth_file in self.model_dir.glob("*.pth"):
                print(f"  - {pth_file.name}")
            return False
    
    def deploy_models(self):
        """Deploy models to node directories"""
        self.print_info("Deploying models to nodes...")
        
        for i in range(self.num_nodes):
            node_id = f"node_{i}"
            node_model_dir = self.model_dir / node_id
            node_model_dir.mkdir(parents=True, exist_ok=True)
            
            model_file = node_model_dir / f"compressor_{self.model_version}.pth"
            metadata_file = node_model_dir / f"metadata_{self.model_version}.json"
            
            if not model_file.exists():
                src_model = self.model_dir / f"compressor_{self.model_version}.pth"
                src_metadata = self.model_dir / f"metadata_{self.model_version}.json"
                
                if src_model.exists():
                    import shutil
                    shutil.copy(src_model, model_file)
                    if src_metadata.exists():
                        shutil.copy(src_metadata, metadata_file)
                    self.print_status(f"Model deployed to {node_id}")
                else:
                    self.print_error(f"Source model not found: {src_model}")
            else:
                self.print_status(f"Model already deployed to {node_id}")
    
    def setup_display(self):
        """Setup display for WSL/headless"""
        if "WSL" in os.uname().release or "microsoft" in os.uname().release.lower():
            self.print_info("Setting up display for WSL...")
            os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"
            os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)
            os.chmod(os.environ["XDG_RUNTIME_DIR"], 0o700)
            
            # Check if Xvfb is running
            result = subprocess.run(["pgrep", "-x", "Xvfb"], capture_output=True)
            if result.returncode != 0:
                subprocess.Popen(["Xvfb", ":99", "-screen", "0", "1024x768x24"], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(1)
            
            os.environ["DISPLAY"] = ":99"
            self.print_status("Display configured")
    
    def launch_node(self, command: List[str], name: str, log_file: str = None):
        """Launch a ROS2 node"""
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.src_dir}:{env.get('PYTHONPATH', '')}"
        
        if log_file:
            log_path = Path("/tmp") / log_file
            with open(log_path, "w") as f:
                process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
        else:
            process = subprocess.Popen(command, env=env)
        
        self.processes.append(process)
        return process
    
    def launch_system(self):
        """Launch all system components"""
        self.print_info("Launching Lydlr system...")
        print("")
        
        # Synthetic publisher
        self.print_info("Launching synthetic data publisher...")
        self.launch_node(
            ["ros2", "run", "lydlr_ai", "synthetic_multimodal_publisher"],
            "synthetic_publisher",
            "lydlr_synthetic.log"
        )
        time.sleep(2)
        self.print_status("Synthetic publisher started")
        
        # Edge nodes
        for i in range(self.num_nodes):
            node_id = f"node_{i}"
            self.print_info(f"Launching edge node: {node_id}...")
            env = os.environ.copy()
            env["NODE_ID"] = node_id
            env["PYTHONPATH"] = f"{self.src_dir}:{env.get('PYTHONPATH', '')}"
            
            log_path = Path("/tmp") / f"lydlr_{node_id}.log"
            with open(log_path, "w") as f:
                process = subprocess.Popen(
                    ["ros2", "run", "lydlr_ai", "edge_compressor_node"],
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
            self.processes.append(process)
            time.sleep(1)
            self.print_status(f"{node_id} started (PID: {process.pid})")
        
        # Deployment manager
        self.print_info("Launching model deployment manager...")
        self.launch_node(
            ["ros2", "run", "lydlr_ai", "model_deployment_manager"],
            "deployment_manager",
            "lydlr_deployment.log"
        )
        time.sleep(2)
        self.print_status("Deployment manager started")
        
        # Coordinator
        self.print_info("Launching distributed coordinator...")
        self.launch_node(
            ["ros2", "run", "lydlr_ai", "distributed_coordinator"],
            "coordinator",
            "lydlr_coordinator.log"
        )
        time.sleep(2)
        self.print_status("Coordinator started")
        
        # Deploy models
        self.print_info("Deploying models to nodes...")
        time.sleep(3)
        
        for i in range(self.num_nodes):
            node_id = f"node_{i}"
            subprocess.run(
                ["ros2", "topic", "pub", "--once", f"/{node_id}/model/deploy", 
                 "std_msgs/String", f"data: '{self.model_version}'"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1)
            self.print_status(f"Model deployed to {node_id}")
    
    def monitor_performance(self):
        """Monitor system performance"""
        print("")
        print("=" * 50)
        print("📊 Performance Monitoring")
        print("=" * 50)
        print("")
        self.print_info("System is running! Monitoring performance...")
        self.print_info("Press Ctrl+C to stop all nodes")
        print("")
        
        while True:
            try:
                # Get metrics
                for i in range(self.num_nodes):
                    node_id = f"node_{i}"
                    try:
                        result = subprocess.run(
                            ["timeout", "1", "ros2", "topic", "echo", 
                             f"/{node_id}/metrics", "--once"],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        
                        if result.returncode == 0 and result.stdout:
                            lines = result.stdout.strip().split('\n')
                            data_line = [l for l in lines if 'data:' in l]
                            if data_line:
                                print(f"{node_id} metrics: {data_line[0]}")
                    except:
                        pass
                
                time.sleep(5)
            except KeyboardInterrupt:
                break
    
    def cleanup(self):
        """Stop all processes"""
        self.print_info("Shutting down Lydlr system...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        self.print_status("All nodes stopped")
    
    def run(self):
        """Main execution"""
        print("=" * 50)
        print("🚀 Lydlr Revolutionary System Launcher")
        print("=" * 50)
        print("")
        
        # Setup signal handler
        def signal_handler(sig, frame):
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Setup steps
            if not self.check_ros2():
                return False
            
            self.setup_python_env()
            if not self.build_ros2_package():
                return False
            
            if not self.check_models():
                return False
            
            self.deploy_models()
            self.setup_display()
            
            # Launch
            self.launch_system()
            
            # Monitor
            self.monitor_performance()
            
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Launch Lydlr Revolutionary Compression System")
    parser.add_argument("--num-nodes", type=int, default=2, help="Number of edge nodes")
    parser.add_argument("--model-version", type=str, default="vv1.0", help="Model version to deploy")
    
    args = parser.parse_args()
    
    launcher = LydlrLauncher(num_nodes=args.num_nodes, model_version=args.model_version)
    success = launcher.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

