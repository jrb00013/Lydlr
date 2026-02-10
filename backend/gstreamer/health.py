"""
GStreamer health check and monitoring
"""
import logging
import time
from typing import Dict, Any, Optional
from django.conf import settings

from backend.gstreamer.utils import GStreamerDiagnostics
from backend.gstreamer.deepstream_plugin import DeepstreamManager

logger = logging.getLogger(__name__)


class GStreamerHealthCheck:
    """Comprehensive health check for GStreamer system"""
    
    def __init__(self):
        self.diagnostics = GStreamerDiagnostics()
        self.deepstream_manager = DeepstreamManager()
        self.last_check_time: Optional[float] = None
        self.cached_status: Optional[Dict[str, Any]] = None
        self.cache_ttl = 60  # Cache for 60 seconds
    
    def get_health_status(self, use_cache: bool = True) -> Dict[str, Any]:
        """Get comprehensive health status"""
        current_time = time.time()
        
        # Use cache if available and fresh
        if use_cache and self.cached_status and self.last_check_time:
            if current_time - self.last_check_time < self.cache_ttl:
                return self.cached_status
        
        # Perform fresh health check
        status = {
            'healthy': True,
            'timestamp': current_time,
            'gstreamer': {},
            'deepstream': {},
            'issues': []
        }
        
        # Check GStreamer installation
        gst_install = self.diagnostics.check_gstreamer_installation()
        status['gstreamer']['installed'] = gst_install['installed']
        status['gstreamer']['version'] = gst_install.get('version')
        status['gstreamer']['plugins_path'] = gst_install.get('plugins_path')
        
        if not gst_install['installed']:
            status['healthy'] = False
            status['issues'].append("GStreamer not properly installed")
            if gst_install.get('errors'):
                status['issues'].extend(gst_install['errors'])
        
        # Check system info
        system_info = self.diagnostics.get_system_info()
        status['gstreamer']['system_info'] = system_info
        
        # Check Deepstream
        status['deepstream']['enabled'] = settings.DEEPSTREAM_ENABLED
        status['deepstream']['available'] = self.deepstream_manager.is_available()
        
        if settings.DEEPSTREAM_ENABLED and not self.deepstream_manager.is_available():
            status['issues'].append("Deepstream enabled but not available")
            status['healthy'] = False
        
        if status['deepstream']['available']:
            status['deepstream']['version'] = self.deepstream_manager.get_version()
            status['deepstream']['models'] = self.deepstream_manager.list_models()
        
        # Cache the result
        self.cached_status = status
        self.last_check_time = current_time
        
        return status
    
    def quick_check(self) -> bool:
        """Quick health check - returns True if system is healthy"""
        status = self.get_health_status()
        return status['healthy']
    
    def get_detailed_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics information"""
        diagnostics = {
            'gstreamer_installation': self.diagnostics.check_gstreamer_installation(),
            'system_info': self.diagnostics.get_system_info(),
            'plugins': self.diagnostics.list_available_plugins()[:50],  # First 50
            'deepstream': {
                'enabled': settings.DEEPSTREAM_ENABLED,
                'available': self.deepstream_manager.is_available(),
                'version': self.deepstream_manager.get_version() if self.deepstream_manager.is_available() else None,
                'models': self.deepstream_manager.list_models() if self.deepstream_manager.is_available() else []
            },
            'configuration': {
                'plugins_path': settings.GSTREAMER_PLUGINS_PATH,
                'deepstream_path': settings.NVIDIA_DEEPSTREAM_PATH if settings.DEEPSTREAM_ENABLED else None
            }
        }
        
        return diagnostics

