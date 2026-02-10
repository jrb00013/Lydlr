"""
GStreamer/Deepstream API views
Comprehensive API endpoints for GStreamer operations
"""
import logging
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from django.conf import settings

from backend.gstreamer.gst_pipeline import (
    GStreamerPipeline, 
    VideoCompressionPipeline,
    RTSPStreamPipeline,
    HLSPipeline,
    WebRTCPipeline
)
from backend.gstreamer.deepstream_plugin import DeepstreamManager, GStreamerPluginManager
from backend.gstreamer.utils import GStreamerDiagnostics, PipelineBuilder
from backend.gstreamer.health import GStreamerHealthCheck

logger = logging.getLogger(__name__)


@api_view(['GET'])
def gstreamer_status(request):
    """Comprehensive GStreamer status check"""
    diagnostics = GStreamerDiagnostics()
    installation = diagnostics.check_gstreamer_installation()
    system_info = diagnostics.get_system_info()
    
    return Response({
        "installation": installation,
        "system_info": system_info,
        "plugins_path": settings.GSTREAMER_PLUGINS_PATH,
        "deepstream_enabled": settings.DEEPSTREAM_ENABLED,
        "deepstream_path": settings.NVIDIA_DEEPSTREAM_PATH if settings.DEEPSTREAM_ENABLED else None
    })


@api_view(['GET'])
def gstreamer_plugins(request):
    """List all available GStreamer plugins"""
    diagnostics = GStreamerDiagnostics()
    plugins = diagnostics.list_available_plugins()
    
    return Response({
        "count": len(plugins),
        "plugins": plugins[:100]  # First 100 plugins
    })


@api_view(['GET'])
def gstreamer_plugin_detail(request, plugin_name):
    """Get details about a specific plugin"""
    diagnostics = GStreamerDiagnostics()
    plugin_info = diagnostics.check_plugin_availability(plugin_name)
    
    if not plugin_info['available']:
        return Response(
            {"error": f"Plugin {plugin_name} not found"},
            status=status.HTTP_404_NOT_FOUND
        )
    
    return Response(plugin_info)


@api_view(['POST'])
def test_pipeline(request):
    """Test a GStreamer pipeline string"""
    pipeline_string = request.data.get('pipeline_string')
    
    if not pipeline_string:
        return Response(
            {"error": "pipeline_string is required"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    diagnostics = GStreamerDiagnostics()
    result = diagnostics.test_pipeline(pipeline_string)
    
    return Response(result)


@api_view(['GET'])
def deepstream_status(request):
    """Check Deepstream status"""
    manager = DeepstreamManager()
    
    return Response({
        "enabled": settings.DEEPSTREAM_ENABLED,
        "available": manager.is_available(),
        "path": manager.deepstream_path,
        "version": manager.get_version() if manager.is_available() else None,
        "models": manager.list_models() if manager.is_available() else []
    })


@api_view(['GET'])
def gstreamer_health(request):
    """Comprehensive GStreamer health check"""
    health_check = GStreamerHealthCheck()
    health_status = health_check.get_health_status()
    
    status_code = status.HTTP_200_OK if health_status['healthy'] else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return Response(health_status, status=status_code)


@api_view(['GET'])
def gstreamer_diagnostics(request):
    """Get detailed GStreamer diagnostics"""
    health_check = GStreamerHealthCheck()
    diagnostics = health_check.get_detailed_diagnostics()
    
    return Response(diagnostics)


class CompressionPipelineView(APIView):
    """Create and manage compression pipelines"""
    
    def post(self, request):
        """Create a compression pipeline"""
        input_source = request.data.get('input_source')
        output_sink = request.data.get('output_sink')
        compression_level = float(request.data.get('compression_level', 0.8))
        codec = request.data.get('codec', 'x264')
        
        if not input_source or not output_sink:
            return Response(
                {"error": "input_source and output_sink are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            pipeline = VideoCompressionPipeline(
                input_source,
                output_sink,
                compression_level,
                codec
            )
            
            if pipeline.create_pipeline():
                return Response({
                    "status": "created",
                    "pipeline_id": id(pipeline),
                    "pipeline_name": pipeline.name,
                    "pipeline_string": pipeline.pipeline_string
                })
            else:
                return Response(
                    {"error": "Failed to create pipeline"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except Exception as e:
            logger.error(f"Error creating compression pipeline: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RTSPPipelineView(APIView):
    """Create and manage RTSP streaming pipelines"""
    
    def post(self, request):
        """Create an RTSP streaming pipeline"""
        input_source = request.data.get('input_source')
        rtsp_port = int(request.data.get('rtsp_port', 8554))
        mount_point = request.data.get('mount_point', '/test')
        codec = request.data.get('codec', 'h264')
        
        if not input_source:
            return Response(
                {"error": "input_source is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            pipeline = RTSPStreamPipeline(
                input_source,
                rtsp_port,
                mount_point,
                codec
            )
            
            if pipeline.create_pipeline():
                return Response({
                    "status": "created",
                    "pipeline_id": id(pipeline),
                    "pipeline_name": pipeline.name,
                    "rtsp_url": f"rtsp://localhost:{rtsp_port}{mount_point}"
                })
            else:
                return Response(
                    {"error": "Failed to create RTSP pipeline"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except Exception as e:
            logger.error(f"Error creating RTSP pipeline: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HLSPipelineView(APIView):
    """Create and manage HLS streaming pipelines"""
    
    def post(self, request):
        """Create an HLS streaming pipeline"""
        input_source = request.data.get('input_source')
        output_dir = request.data.get('output_dir')
        playlist_name = request.data.get('playlist_name', 'playlist.m3u8')
        segment_duration = int(request.data.get('segment_duration', 2))
        
        if not input_source or not output_dir:
            return Response(
                {"error": "input_source and output_dir are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            pipeline = HLSPipeline(
                input_source,
                output_dir,
                playlist_name,
                segment_duration
            )
            
            if pipeline.create_pipeline():
                return Response({
                    "status": "created",
                    "pipeline_id": id(pipeline),
                    "pipeline_name": pipeline.name,
                    "playlist_url": f"{output_dir}/{playlist_name}"
                })
            else:
                return Response(
                    {"error": "Failed to create HLS pipeline"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except Exception as e:
            logger.error(f"Error creating HLS pipeline: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class WebRTCPipelineView(APIView):
    """Create and manage WebRTC streaming pipelines"""
    
    def post(self, request):
        """Create a WebRTC streaming pipeline"""
        input_source = request.data.get('input_source')
        stun_server = request.data.get('stun_server', 'stun://stun.l.google.com:19302')
        
        if not input_source:
            return Response(
                {"error": "input_source is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            pipeline = WebRTCPipeline(input_source, stun_server)
            
            if pipeline.create_pipeline():
                return Response({
                    "status": "created",
                    "pipeline_id": id(pipeline),
                    "pipeline_name": pipeline.name,
                    "stun_server": stun_server
                })
            else:
                return Response(
                    {"error": "Failed to create WebRTC pipeline"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        except Exception as e:
            logger.error(f"Error creating WebRTC pipeline: {e}")
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class DeepstreamInferenceView(APIView):
    """Run Deepstream inference"""
    
    def post(self, request):
        """Create and run Deepstream inference"""
        manager = DeepstreamManager()
        
        if not manager.is_available():
            return Response(
                {"error": "Deepstream is not available"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        input_source = request.data.get('input_source')
        output_sink = request.data.get('output_sink')
        model_path = request.data.get('model_path')
        
        if not all([input_source, output_sink, model_path]):
            return Response(
                {"error": "input_source, output_sink, and model_path are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        config_file = manager.create_config(
            input_source,
            output_sink,
            model_path,
            request.data.get('config')
        )
        
        success = manager.run_inference(config_file)
        
        if success:
            return Response({
                "status": "running",
                "config_file": config_file
            })
        else:
            return Response(
                {"error": "Failed to start inference"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

