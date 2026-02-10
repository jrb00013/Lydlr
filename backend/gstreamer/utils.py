"""
GStreamer utilities and helper functions
Comprehensive diagnostics, health checks, and pipeline management
"""
import os
import subprocess
import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class GStreamerDiagnostics:
    """Comprehensive GStreamer diagnostics and health checks"""
    
    @staticmethod
    def check_gstreamer_installation() -> Dict[str, Any]:
        """Check if GStreamer is properly installed"""
        result = {
            'installed': False,
            'version': None,
            'plugins_path': None,
            'errors': []
        }
        
        try:
            # Check gst-launch-1.0
            proc = subprocess.run(
                ['gst-launch-1.0', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if proc.returncode == 0:
                result['installed'] = True
                result['version'] = proc.stdout.strip()
        except FileNotFoundError:
            result['errors'].append("gst-launch-1.0 not found in PATH")
        except subprocess.TimeoutExpired:
            result['errors'].append("gst-launch-1.0 timeout")
        except Exception as e:
            result['errors'].append(f"Error checking GStreamer: {str(e)}")
        
        # Check plugins path
        possible_paths = [
            '/usr/lib/x86_64-linux-gnu/gstreamer-1.0',
            '/usr/lib/gstreamer-1.0',
            '/usr/local/lib/gstreamer-1.0'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                result['plugins_path'] = path
                break
        
        return result
    
    @staticmethod
    def list_available_plugins() -> List[str]:
        """List all available GStreamer plugins"""
        plugins = []
        try:
            proc = subprocess.run(
                ['gst-inspect-1.0'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if proc.returncode == 0:
                plugins = [line.strip() for line in proc.stdout.split('\n') if line.strip()]
        except Exception as e:
            logger.error(f"Error listing plugins: {e}")
        
        return plugins
    
    @staticmethod
    def check_plugin_availability(plugin_name: str) -> Dict[str, Any]:
        """Check if a specific plugin is available"""
        result = {
            'available': False,
            'description': None,
            'elements': []
        }
        
        try:
            proc = subprocess.run(
                ['gst-inspect-1.0', plugin_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            if proc.returncode == 0:
                result['available'] = True
                result['description'] = proc.stdout[:200]  # First 200 chars
                # Extract element names
                lines = proc.stdout.split('\n')
                for line in lines:
                    if 'Factory Details:' in line or 'Element Details:' in line:
                        # Try to extract element name
                        pass
        except FileNotFoundError:
            result['available'] = False
        except Exception as e:
            logger.error(f"Error checking plugin {plugin_name}: {e}")
        
        return result
    
    @staticmethod
    def test_pipeline(pipeline_string: str) -> Dict[str, Any]:
        """Test a GStreamer pipeline string"""
        result = {
            'valid': False,
            'error': None,
            'warnings': []
        }
        
        try:
            # Use gst-launch-1.0 --check to validate pipeline
            proc = subprocess.run(
                ['gst-launch-1.0', '--check', pipeline_string],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if proc.returncode == 0:
                result['valid'] = True
            else:
                result['error'] = proc.stderr
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get GStreamer system information"""
        info = {
            'gstreamer_version': None,
            'plugins_count': 0,
            'codecs_available': [],
            'formats_available': [],
            'devices_available': []
        }
        
        try:
            # Get version
            proc = subprocess.run(
                ['gst-launch-1.0', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if proc.returncode == 0:
                info['gstreamer_version'] = proc.stdout.strip()
        except Exception:
            pass
        
        # Count plugins
        plugins = GStreamerDiagnostics.list_available_plugins()
        info['plugins_count'] = len(plugins)
        
        # Check for common codecs
        codecs = ['x264', 'x265', 'vp8', 'vp9', 'h264', 'h265', 'mpeg2video']
        for codec in codecs:
            plugin_check = GStreamerDiagnostics.check_plugin_availability(codec)
            if plugin_check['available']:
                info['codecs_available'].append(codec)
        
        # Check for video devices
        try:
            proc = subprocess.run(
                ['v4l2-ctl', '--list-devices'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if proc.returncode == 0:
                # Parse devices
                lines = proc.stdout.split('\n')
                for line in lines:
                    if '/dev/video' in line:
                        info['devices_available'].append(line.strip())
        except Exception:
            pass
        
        return info


class PipelineBuilder:
    """Builder class for creating GStreamer pipelines"""
    
    def __init__(self):
        self.elements = []
        self.caps = []
    
    def add_source(self, source_type: str, location: str) -> 'PipelineBuilder':
        """Add source element"""
        if source_type == "file":
            self.elements.append(f"filesrc location={location}")
        elif source_type == "rtsp":
            self.elements.append(f"rtspsrc location={location}")
        elif source_type == "http":
            self.elements.append(f"souphttpsrc location={location}")
        elif source_type == "v4l2":
            self.elements.append(f"v4l2src device={location}")
        elif source_type == "test":
            self.elements.append(f"videotestsrc")
        return self
    
    def add_decoder(self, decoder_type: str = "auto") -> 'PipelineBuilder':
        """Add decoder element"""
        if decoder_type == "auto":
            self.elements.append("decodebin")
        else:
            self.elements.append(f"{decoder_type}dec")
        return self
    
    def add_video_convert(self) -> 'PipelineBuilder':
        """Add video converter"""
        self.elements.append("videoconvert")
        return self
    
    def add_video_scale(self, width: Optional[int] = None, height: Optional[int] = None) -> 'PipelineBuilder':
        """Add video scaler"""
        self.elements.append("videoscale")
        if width and height:
            self.caps.append(f"video/x-raw,width={width},height={height}")
        return self
    
    def add_encoder(self, encoder: str, bitrate: int = 1000, **kwargs) -> 'PipelineBuilder':
        """Add encoder element"""
        encoder_map = {
            "h264": "x264enc",
            "h265": "x265enc",
            "vp8": "vp8enc",
            "vp9": "vp9enc"
        }
        
        enc_name = encoder_map.get(encoder.lower(), encoder)
        params = [f"{k}={v}" for k, v in kwargs.items()]
        params_str = " ".join(params) if params else ""
        
        if encoder.lower() in ["h264", "x264"]:
            self.elements.append(f"x264enc bitrate={bitrate} {params_str}".strip())
        elif encoder.lower() in ["h265", "x265"]:
            self.elements.append(f"x265enc bitrate={bitrate} {params_str}".strip())
        elif encoder.lower() == "vp8":
            self.elements.append(f"vp8enc target-bitrate={bitrate * 1000} {params_str}".strip())
        elif encoder.lower() == "vp9":
            self.elements.append(f"vp9enc target-bitrate={bitrate * 1000} {params_str}".strip())
        
        return self
    
    def add_sink(self, sink_type: str, location: str) -> 'PipelineBuilder':
        """Add sink element"""
        if sink_type == "file":
            self.elements.append(f"filesink location={location}")
        elif sink_type == "rtsp":
            self.elements.append(f"rtspclientsink location={location}")
        elif sink_type == "udp":
            self.elements.append(f"udpsink host={location.split(':')[0]} port={location.split(':')[1]}")
        elif sink_type == "rtp":
            self.elements.append(f"rtph264pay ! udpsink host={location.split(':')[0]} port={location.split(':')[1]}")
        return self
    
    def build(self) -> str:
        """Build pipeline string"""
        pipeline = " ! ".join(self.elements)
        if self.caps:
            # Insert caps at appropriate position
            caps_str = " ! ".join(self.caps)
            # Simple insertion - could be improved
            pipeline = pipeline.replace("videoscale", f"videoscale ! {caps_str}")
        return pipeline


def create_compression_pipeline_string(
    input_source: str,
    output_sink: str,
    codec: str = "h264",
    bitrate: int = 1000,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> str:
    """Create a compression pipeline string"""
    builder = PipelineBuilder()
    builder.add_source("file", input_source)
    builder.add_decoder()
    builder.add_video_convert()
    if width and height:
        builder.add_video_scale(width, height)
    else:
        builder.add_video_scale()
    builder.add_encoder(codec, bitrate, speed_preset="ultrafast", tune="zerolatency")
    builder.add_sink("file", output_sink)
    return builder.build()

