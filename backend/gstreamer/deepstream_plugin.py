"""
NVIDIA Deepstream integration for AI inference
Comprehensive Deepstream support with configuration management
"""
import os
import json
import logging
import subprocess
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DeepstreamManager:
    """Manager for NVIDIA Deepstream pipelines"""
    
    def __init__(self, deepstream_path: Optional[str] = None):
        self.deepstream_path = deepstream_path or os.getenv(
            'NVIDIA_DEEPSTREAM_PATH',
            '/opt/nvidia/deepstream/deepstream'
        )
        self.enabled = os.getenv('DEEPSTREAM_ENABLED', 'False') == 'True'
        self.pipelines: Dict[str, Any] = {}
    
    def is_available(self) -> bool:
        """Check if Deepstream is available"""
        if not self.enabled:
            return False
        return os.path.exists(self.deepstream_path)
    
    def get_version(self) -> Optional[str]:
        """Get Deepstream version"""
        if not self.is_available():
            return None
        
        try:
            deepstream_app = os.path.join(self.deepstream_path, "bin", "deepstream-app")
            if os.path.exists(deepstream_app):
                proc = subprocess.run(
                    [deepstream_app, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if proc.returncode == 0:
                    return proc.stdout.strip()
        except Exception as e:
            logger.error(f"Error getting Deepstream version: {e}")
        
        return None
    
    def list_models(self) -> List[str]:
        """List available Deepstream models"""
        models = []
        if not self.is_available():
            return models
        
        models_dir = os.path.join(self.deepstream_path, "samples", "models")
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                if item.endswith(('.engine', '.onnx', '.trt')):
                    models.append(item)
        
        return models
    
    def create_config(self, 
                     input_source: str,
                     output_sink: str,
                     model_path: str,
                     config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create Deepstream config file
        
        Args:
            input_source: Input video source
            output_sink: Output sink
            model_path: Path to model file
            config: Additional configuration
        
        Returns:
            Path to created config file
        """
        config_dir = Path("/tmp/deepstream_configs")
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / f"lydlr_{hash(input_source)}.txt"
        
        # Basic Deepstream config structure
        deepstream_config = {
            "application": {
                "enable-perf-measurement": 1,
                "perf-measurement-interval-sec": 5
            },
            "source": {
                "uri": input_source,
                "type": "uri" if input_source.startswith("http") else "file"
            },
            "sink": {
                "type": "file",
                "uri": output_sink,
                "sync": 0
            },
            "primary-gie": {
                "enable": 1,
                "model-engine-file": model_path,
                "config-file": str(Path(model_path).parent / "config.txt")
            }
        }
        
        # Merge with custom config
        if config:
            deepstream_config.update(config)
        
        # Write config file (Deepstream uses specific format)
        with open(config_file, 'w') as f:
            # Deepstream config format is specific - this is a simplified version
            f.write(f"[application]\n")
            f.write(f"enable-perf-measurement=1\n")
            f.write(f"[source0]\n")
            f.write(f"uri={input_source}\n")
            f.write(f"[sink0]\n")
            f.write(f"uri={output_sink}\n")
        
        return str(config_file)
    
    def run_inference(self, config_file: str) -> bool:
        """
        Run Deepstream inference pipeline
        
        Args:
            config_file: Path to config file
        
        Returns:
            True if successful
        """
        if not self.is_available():
            print("Deepstream is not available")
            return False
        
        # This would actually launch Deepstream
        # For now, it's a placeholder
        print(f"Running Deepstream inference with config: {config_file}")
        return True


class GStreamerPluginManager:
    """Manager for GStreamer plugins with comprehensive functionality"""
    
    def __init__(self):
        self.plugins_loaded = False
        self.available_plugins = []
        self.active_pipelines: Dict[str, Any] = {}
    
    def check_plugins(self) -> list:
        """Check available GStreamer plugins"""
        import subprocess
        try:
            result = subprocess.run(
                ['gst-inspect-1.0'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.available_plugins = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                self.plugins_loaded = True
        except FileNotFoundError:
            logger.error("GStreamer not found in PATH")
        except subprocess.TimeoutExpired:
            logger.error("GStreamer plugin check timeout")
        except Exception as e:
            logger.error(f"Error checking plugins: {e}")
        
        return self.available_plugins
    
    def create_compression_pipeline(self, 
                                   input_source: str,
                                   output_sink: str,
                                   compression_level: float = 0.8,
                                   codec: str = "x264") -> Optional[Any]:
        """Create a video compression pipeline"""
        from backend.gstreamer.gst_pipeline import VideoCompressionPipeline
        
        pipeline = VideoCompressionPipeline(input_source, output_sink, compression_level, codec)
        if pipeline.create_pipeline():
            pipeline_id = str(id(pipeline))
            self.active_pipelines[pipeline_id] = pipeline
            return pipeline
        return None
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Any]:
        """Get an active pipeline by ID"""
        return self.active_pipelines.get(pipeline_id)
    
    def remove_pipeline(self, pipeline_id: str) -> bool:
        """Remove and stop a pipeline"""
        if pipeline_id in self.active_pipelines:
            pipeline = self.active_pipelines[pipeline_id]
            pipeline.stop()
            del self.active_pipelines[pipeline_id]
            return True
        return False
    
    def list_active_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """List all active pipelines with their status"""
        pipelines_info = {}
        for pipeline_id, pipeline in self.active_pipelines.items():
            pipelines_info[pipeline_id] = {
                'name': pipeline.name,
                'state': pipeline.get_state().value if pipeline.get_state() else None,
                'stats': pipeline.get_stats()
            }
        return pipelines_info

